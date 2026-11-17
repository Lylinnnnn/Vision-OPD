"""
Qwen3-VL-8B 推理脚本（基于 HuggingFace transformers）
功能: 对每个问题进行多次采样，收集生成文本及 top-k logits，
     用于后续的 RLSF-V 训练数据构建。
支持多 GPU 分块并行推理（通过 --num-chunks / --chunk-idx 参数）。

与 qwen25_infer.py (Qwen2.5-VL) 的区别:
  - 使用 Qwen3VLForConditionalGeneration 替代 Qwen2_5_VLForConditionalGeneration
  - 使用 processor.apply_chat_template(tokenize=True, return_dict=True) 一步完成输入构造,
    无需 qwen_vl_utils.process_vision_info

输出格式与 qwen25_infer.py 完全一致：
  - JSONL 文件: 每行包含 question_id, image_path, prompt, text, answer_id, metadata
  - PT 文件: {answer_id: {"topk_values": Tensor, "generated_tokens": Tensor}}
"""

import argparse
import json
import math
import os
import random

import numpy as np
import shortuuid
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ==================== 工具函数 ====================

def setup_seeds(seed: int) -> None:
    """固定随机种子以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_questions(question_file: str) -> list:
    """从 JSONL 文件中逐行加载问题数据。"""
    questions = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipped invalid json line: {line[:50]}...")
    return questions


def get_chunk_manual(lst: list, num_chunks: int, chunk_idx: int) -> list:
    """将列表按 chunk 数量均匀分块，返回第 chunk_idx 个分块。"""
    if not lst:
        return []
    chunk_size = math.ceil(len(lst) / num_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(lst))
    return lst[start_idx:end_idx]


def get_id_from_sample(sample: dict) -> str:
    """从样本中提取唯一标识符。"""
    if "question_id" in sample:
        return sample["question_id"]
    elif "id" in sample:
        return sample["id"]
    elif "image_id" in sample:
        return sample["image_id"]
    else:
        return shortuuid.uuid()


def get_question_from_sample(sample: dict) -> str:
    """从样本中提取问题文本。"""
    if "text" in sample:
        return sample["text"]
    elif "query" in sample:
        return sample["query"]
    elif "question" in sample:
        return sample["question"]
    return ""


def get_image_path_from_sample(sample: dict) -> str | None:
    """从样本中提取图像路径。"""
    for key in ("image", "image_path", "image_file"):
        if key in sample:
            return sample[key]
    return None


def load_image(image_path: str) -> Image.Image:
    """加载图像文件并转换为 RGB 格式。"""
    return Image.open(image_path).convert("RGB")


def build_messages(question: str, full_image_path: str | None) -> list:
    """构造 Qwen3-VL 格式的消息列表。"""
    content_parts = []
    if full_image_path:
        content_parts.append({"type": "image", "image": f"{full_image_path}"})
    content_parts.append({"type": "text", "text": question})
    return [{"role": "user", "content": content_parts}]


# ==================== 单样本 fallback ====================

def process_single_sample(
    line: dict,
    sample_i: int,
    args,
    processor,
    model,
    answers_file: str,
    all_logits_data: dict,
) -> None:
    """
    单样本推理的 fallback 函数。
    当 batch 推理出错时，回退到逐条处理以保证鲁棒性。
    """
    idx = get_id_from_sample(line)
    question = get_question_from_sample(line)
    image_path = get_image_path_from_sample(line)

    # 解析完整图像路径
    full_image_path = image_path
    if image_path:
        if hasattr(args, "image_folder") and args.image_folder:
            full_image_path = os.path.join(args.image_folder, image_path)
        try:
            _ = load_image(full_image_path)  # 验证图像可加载
        except Exception as e:
            print(f"Error loading image {full_image_path}: {e}")
            return

    # 构造 Qwen3-VL 消息并用 processor 一步完成输入构造
    messages = build_messages(question, full_image_path if image_path else None)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # 模型推理
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=(args.temperature > 0),
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # 解码生成结果（仅保留新生成的 token）
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output.sequences[0, input_len:]
    text_output = processor.decode(generated_ids, skip_special_tokens=True).strip()
    ans_id = shortuuid.uuid()

    # 提取每个生成 step 的 top-k logits
    topk_values_list = []
    for step_score in output.scores:
        step_score = step_score.squeeze(0)  # [vocab_size]
        vals, _ = torch.topk(step_score, k=args.top_k_logits)
        topk_values_list.append(vals.float().cpu())

    if topk_values_list:
        all_logits_data[ans_id] = {
            "topk_values": torch.stack(topk_values_list),   # [num_tokens, top_k]
            "generated_tokens": generated_ids.cpu(),          # [num_tokens]
        }

    # 写入结果
    with open(answers_file, "a", encoding="utf-8") as ans_file:
        ans_file.write(json.dumps({
            "question_id": idx,
            "image_path": image_path,
            "prompt": question,
            "text": text_output,
            "answer_id": ans_id,
            "metadata": {"sample_idx": sample_i},
        }, ensure_ascii=False) + "\n")


# ==================== 主推理流程 ====================

def eval_model(args):
    """主推理流程：加载模型 -> 遍历问题 -> batch 采样 -> 保存结果。"""
    setup_seeds(args.seed)

    print(f"Loading model: {args.model_name}...")

    # 加载 Qwen3-VL 模型和处理器
    # 使用 bfloat16 而非 float16，避免 logits 溢出产生 inf/nan 导致采样失败
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        args.model_name, min_pixels=min_pixels, max_pixels=max_pixels
    )

    print(f"Loading questions from {args.question_file}...")
    questions = load_questions(args.question_file)
    questions = get_chunk_manual(questions, args.num_chunks, args.chunk_idx)
    print(f"Processing chunk {args.chunk_idx}/{args.num_chunks}: {len(questions)} questions.")

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # logits 保存路径（每个 chunk 独立保存，最终由 shell 脚本合并）
    merged_logits_file = os.path.join(
        os.path.dirname(answers_file),
        f"all_logits_chunk{args.chunk_idx}_merged.pt",
    )

    # 每次运行从头开始，清空已有输出文件
    open(answers_file, "w").close()

    # 存储所有 logits 数据，最终统一写入磁盘
    all_logits_data = {}

    # 逐个问题遍历，每个问题内部进行 batch 多次采样
    for line in tqdm(questions, desc="Processing questions"):
        idx = get_id_from_sample(line)
        question = get_question_from_sample(line)
        image_path = get_image_path_from_sample(line)

        # --- 解析完整图像路径 ---
        full_image_path = image_path
        if image_path:
            if hasattr(args, "image_folder") and args.image_folder:
                full_image_path = os.path.join(args.image_folder, image_path)
            try:
                _ = load_image(full_image_path)  # 验证图像可加载
            except Exception as e:
                print(f"Error loading image {full_image_path}: {e}")
                continue

        # --- 构造 Qwen3-VL 消息并一步完成输入构造 ---
        messages = build_messages(question, full_image_path if image_path else None)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]

        # --- 对同一问题进行 num_samples 次采样，按 batch_size 分批 ---
        all_sample_indices = list(range(args.num_samples))
        batch_size = args.batch_size

        for batch_start in range(0, len(all_sample_indices), batch_size):
            batch_sample_indices = all_sample_indices[batch_start:batch_start + batch_size]
            current_batch_size = len(batch_sample_indices)

            # 将输入复制 batch_size 份
            batched_inputs = {}
            for key, val in inputs.items():
                if isinstance(val, torch.Tensor):
                    batched_inputs[key] = val.repeat(current_batch_size, *([1] * (val.dim() - 1)))
                else:
                    batched_inputs[key] = val

            try:
                with torch.inference_mode():
                    outputs = model.generate(
                        **batched_inputs,
                        do_sample=(args.temperature > 0),
                        temperature=args.temperature if args.temperature > 0 else None,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                output_ids = outputs.sequences  # [batch_size, seq_len]

                for batch_idx, sample_i in enumerate(batch_sample_indices):
                    # 提取该样本的生成 token 和文本
                    generated_ids = output_ids[batch_idx, input_len:]
                    text_output = processor.decode(generated_ids, skip_special_tokens=True).strip()
                    ans_id = shortuuid.uuid()

                    # 收集每个 step 的 top-k logits
                    topk_values_list = []
                    topk_indices_list = []
                    for step_score in outputs.scores:
                        # step_score: [batch_size, vocab_size]
                        sample_score = step_score[batch_idx]
                        # vals, _ = torch.topk(sample_score, k=args.top_k_logits)
                        # topk_values_list.append(vals.float().cpu())
                        vals, idxs = torch.topk(sample_score, k=args.top_k_logits)
                        topk_values_list.append(vals.float().cpu())
                        topk_indices_list.append(idxs.long().cpu())

                    if topk_values_list:
                        # all_logits_data[ans_id] = {
                        #     "topk_values": torch.stack(topk_values_list),   # [num_tokens, top_k]
                        #     "generated_tokens": generated_ids.cpu(),         # [num_tokens]
                        # }
                        all_logits_data[ans_id] = {
                            "topk_values": torch.stack(topk_values_list),
                            "topk_indices": torch.stack(topk_indices_list),
                            "generated_tokens": generated_ids.cpu(),
                        }

                    # 逐条写入 JSONL 文件
                    with open(answers_file, "a", encoding="utf-8") as ans_file:
                        ans_file.write(json.dumps({
                            "question_id": idx,
                            "image_path": image_path,
                            "prompt": question,
                            "text": text_output,
                            "answer_id": ans_id,
                            "metadata": {"sample_idx": sample_i},
                        }, ensure_ascii=False) + "\n")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error in batch generation for q_id {idx}: {e}")
                # CUDA device-side assert 触发后，整个 CUDA 上下文被污染，
                # 后续所有 CUDA 操作都会失败，fallback 也无法成功。
                # 尝试清理 CUDA 状态，如果失败则跳过该问题。
                torch.cuda.empty_cache()
                try:
                    torch.cuda.synchronize()
                except RuntimeError:
                    print(f"CUDA context corrupted, skipping q_id {idx}")
                    break  # 跳出当前问题的 batch 循环
                # 如果 CUDA 状态可恢复，再尝试逐条处理
                for sample_i in batch_sample_indices:
                    try:
                        process_single_sample(
                            line, sample_i, args, processor, model,
                            answers_file, all_logits_data,
                        )
                    except Exception as e2:
                        print(f"Error in fallback single processing: {e2}")

    # 所有问题处理完毕，统一保存 logits 文件
    if all_logits_data:
        print(f"Saving merged logits to {merged_logits_file}...")
        torch.save(all_logits_data, merged_logits_file)
        print(f"Saved {len(all_logits_data)} logits entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-8B multi-sample inference with logits collection"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--image-folder", type=str, default="",
                        help="Optional prefix folder for image paths")
    parser.add_argument("--question-file", type=str, required=True,
                        help="Path to the JSONL question file")
    parser.add_argument("--answers-file", type=str, required=True,
                        help="Path to the output JSONL answers file")
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Total number of chunks for multi-GPU parallel inference")
    parser.add_argument("--chunk-idx", type=int, default=0,
                        help="Index of the current chunk (0-based)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of samples to generate per question")
    parser.add_argument("--top-k-logits", type=int, default=50,
                        help="Number of top-k logits to save per generation step")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for inference per question")

    args = parser.parse_args()
    eval_model(args)
