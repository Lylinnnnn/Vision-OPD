"""
InternVL3.5-8B-HF inference script (Transformers backend).

Purpose:
- Generate multiple sampled responses per question.
- Save JSONL outputs and per-sample top-k logits for later RLSF-V data building.
- Support multi-GPU chunked inference via --num-chunks / --chunk-idx.

Output format matches qwen3_infer.py:
- JSONL: question_id, image_path, prompt, text, answer_id, metadata
- PT: {answer_id: {"topk_values": Tensor, "topk_indices": Tensor, "generated_tokens": Tensor}}
"""

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import shortuuid
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


# ==================== Utilities ====================

def setup_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_questions(question_file: str) -> List[dict]:
    """Load JSONL samples line by line."""
    questions: List[dict] = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: skipped invalid JSON line: {line[:80]}...")
    return questions


def get_chunk_manual(lst: List[Any], num_chunks: int, chunk_idx: int) -> List[Any]:
    """Split list into chunks and return one chunk by index."""
    if not lst:
        return []
    chunk_size = math.ceil(len(lst) / num_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(lst))
    return lst[start_idx:end_idx]


def get_id_from_sample(sample: dict) -> str:
    """Extract a unique ID from one sample."""
    if "question_id" in sample:
        return str(sample["question_id"])
    if "id" in sample:
        return str(sample["id"])
    if "image_id" in sample:
        return str(sample["image_id"])
    return shortuuid.uuid()


def get_question_from_sample(sample: dict) -> str:
    """Extract question text from one sample."""
    if "text" in sample:
        return str(sample["text"])
    if "query" in sample:
        return str(sample["query"])
    if "question" in sample:
        return str(sample["question"])
    return ""


def get_image_path_from_sample(sample: dict) -> Optional[str]:
    """Extract image path from one sample."""
    for key in ("image", "image_path", "image_file"):
        if key in sample and sample[key]:
            return str(sample[key])
    return None


def load_image(image_path: str) -> Image.Image:
    """Load and convert an image to RGB (for validation only)."""
    return Image.open(image_path).convert("RGB")


def build_messages(question: str, full_image_path: Optional[str]) -> List[dict]:
    """Build InternVL chat messages for one request."""
    content_parts: List[dict] = []
    if full_image_path:
        content_parts.append({"type": "image", "url": full_image_path})
    content_parts.append({"type": "text", "text": question})
    return [{"role": "user", "content": content_parts}]


def move_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor fields to target device without changing tensor dtypes."""
    out: Dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def prepare_inputs(processor, messages: List[dict], model_device: torch.device, batch_size: int) -> Dict[str, Any]:
    """Prepare tokenized inputs for a single or batched list of identical prompts."""
    if batch_size == 1:
        payload = messages
        inputs = processor.apply_chat_template(
            payload,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        payload = [messages for _ in range(batch_size)]
        inputs = processor.apply_chat_template(
            payload,
            padding=True,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    return move_to_device(inputs, model_device)


# ==================== Single-sample fallback ====================

def process_single_sample(
    line: dict,
    sample_i: int,
    args,
    processor,
    model,
    answers_file: str,
    all_logits_data: dict,
) -> None:
    """Fallback single-sample inference when batched generation fails."""
    idx = get_id_from_sample(line)
    question = get_question_from_sample(line)
    image_path = get_image_path_from_sample(line)

    full_image_path = image_path
    if image_path:
        if args.image_folder:
            full_image_path = os.path.join(args.image_folder, image_path)
        try:
            _ = load_image(full_image_path)
        except Exception as e:
            print(f"Error loading image {full_image_path}: {e}")
            return

    messages = build_messages(question, full_image_path if image_path else None)
    model_device = next(model.parameters()).device
    inputs = prepare_inputs(processor, messages, model_device, batch_size=1)

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

    input_len = int(inputs["input_ids"].shape[1])
    generated_ids = output.sequences[0, input_len:]
    text_output = processor.decode(generated_ids, skip_special_tokens=True).strip()
    ans_id = shortuuid.uuid()

    topk_values_list = []
    topk_indices_list = []
    for step_score in output.scores:
        sample_score = step_score.squeeze(0)
        vals, idxs = torch.topk(sample_score, k=args.top_k_logits)
        topk_values_list.append(vals.float().cpu())
        topk_indices_list.append(idxs.long().cpu())

    if topk_values_list:
        all_logits_data[ans_id] = {
            "topk_values": torch.stack(topk_values_list),
            "topk_indices": torch.stack(topk_indices_list),
            "generated_tokens": generated_ids.cpu(),
        }

    with open(answers_file, "a", encoding="utf-8") as ans_file:
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "image_path": image_path,
                    "prompt": question,
                    "text": text_output,
                    "answer_id": ans_id,
                    "metadata": {"sample_idx": sample_i},
                },
                ensure_ascii=False,
            )
            + "\n"
        )


# ==================== Main inference ====================

def eval_model(args) -> None:
    """Main flow: load model, run chunked multi-sample generation, save outputs."""
    setup_seeds(args.seed)

    print(f"Loading model: {args.model_name} ...")

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if not args.disable_flash_attn:
        model_kwargs["use_flash_attn"] = True

    try:
        model = AutoModelForImageTextToText.from_pretrained(args.model_name, **model_kwargs)
    except TypeError as e:
        if "use_flash_attn" in model_kwargs:
            print(f"Warning: use_flash_attn is not supported in this environment: {e}")
            model_kwargs.pop("use_flash_attn", None)
            model = AutoModelForImageTextToText.from_pretrained(args.model_name, **model_kwargs)
        else:
            raise

    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model_device = next(model.parameters()).device

    print(f"Loading questions from {args.question_file} ...")
    questions = load_questions(args.question_file)
    questions = get_chunk_manual(questions, args.num_chunks, args.chunk_idx)
    print(f"Processing chunk {args.chunk_idx}/{args.num_chunks}: {len(questions)} questions")

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    merged_logits_file = os.path.join(
        os.path.dirname(answers_file),
        f"all_logits_chunk{args.chunk_idx}_merged.pt",
    )

    open(answers_file, "w").close()
    all_logits_data: Dict[str, dict] = {}

    for line in tqdm(questions, desc="Processing questions"):
        idx = get_id_from_sample(line)
        question = get_question_from_sample(line)
        image_path = get_image_path_from_sample(line)

        full_image_path = image_path
        if image_path:
            if args.image_folder:
                full_image_path = os.path.join(args.image_folder, image_path)
            try:
                _ = load_image(full_image_path)
            except Exception as e:
                print(f"Error loading image {full_image_path}: {e}")
                continue

        messages = build_messages(question, full_image_path if image_path else None)

        all_sample_indices = list(range(args.num_samples))
        batch_size = args.batch_size

        for batch_start in range(0, len(all_sample_indices), batch_size):
            batch_sample_indices = all_sample_indices[batch_start : batch_start + batch_size]
            current_batch_size = len(batch_sample_indices)

            try:
                batched_inputs = prepare_inputs(
                    processor=processor,
                    messages=messages,
                    model_device=model_device,
                    batch_size=current_batch_size,
                )

                input_lens = (
                    batched_inputs["attention_mask"].sum(dim=1).tolist()
                    if "attention_mask" in batched_inputs
                    else [batched_inputs["input_ids"].shape[1]] * current_batch_size
                )

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

                output_ids = outputs.sequences

                for batch_idx, sample_i in enumerate(batch_sample_indices):
                    input_len = int(input_lens[batch_idx])
                    generated_ids = output_ids[batch_idx, input_len:]
                    text_output = processor.decode(generated_ids, skip_special_tokens=True).strip()
                    ans_id = shortuuid.uuid()

                    topk_values_list = []
                    topk_indices_list = []
                    for step_score in outputs.scores:
                        sample_score = step_score[batch_idx]
                        vals, idxs = torch.topk(sample_score, k=args.top_k_logits)
                        topk_values_list.append(vals.float().cpu())
                        topk_indices_list.append(idxs.long().cpu())

                    if topk_values_list:
                        all_logits_data[ans_id] = {
                            "topk_values": torch.stack(topk_values_list),
                            "topk_indices": torch.stack(topk_indices_list),
                            "generated_tokens": generated_ids.cpu(),
                        }

                    with open(answers_file, "a", encoding="utf-8") as ans_file:
                        ans_file.write(
                            json.dumps(
                                {
                                    "question_id": idx,
                                    "image_path": image_path,
                                    "prompt": question,
                                    "text": text_output,
                                    "answer_id": ans_id,
                                    "metadata": {"sample_idx": sample_i},
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Error in batch generation for q_id {idx}: {e}")

                torch.cuda.empty_cache()
                try:
                    torch.cuda.synchronize()
                except RuntimeError:
                    print(f"CUDA context corrupted, skipping q_id {idx}")
                    break

                for sample_i in batch_sample_indices:
                    try:
                        process_single_sample(
                            line=line,
                            sample_i=sample_i,
                            args=args,
                            processor=processor,
                            model=model,
                            answers_file=answers_file,
                            all_logits_data=all_logits_data,
                        )
                    except Exception as e2:
                        print(f"Error in fallback single processing: {e2}")

    if all_logits_data:
        print(f"Saving merged logits to {merged_logits_file} ...")
        torch.save(all_logits_data, merged_logits_file)
        print(f"Saved {len(all_logits_data)} logits entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="InternVL3.5-8B-HF multi-sample inference with logits collection"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="OpenGVLab/InternVL3_5-8B-HF",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default="",
        help="Optional prefix folder for image paths",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        required=True,
        help="Path to JSONL question file",
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        required=True,
        help="Path to output JSONL answers file",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help="Total number of chunks for multi-GPU parallel inference",
    )
    parser.add_argument(
        "--chunk-idx",
        type=int,
        default=0,
        help="Index of current chunk (0-based)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of sampled responses per question",
    )
    parser.add_argument(
        "--top-k-logits",
        type=int,
        default=50,
        help="Number of top-k logits to save per generation step",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per question",
    )
    parser.add_argument(
        "--disable-flash-attn",
        action="store_true",
        help="Disable use_flash_attn when loading model",
    )

    args = parser.parse_args()
    eval_model(args)
