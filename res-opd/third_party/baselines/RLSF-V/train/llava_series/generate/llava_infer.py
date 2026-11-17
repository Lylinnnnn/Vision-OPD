"""
Run LLaVA multi-sample inference and save generated answers plus top-k logits.

Supports chunked multi-GPU execution via --num-chunks and --chunk-idx.
"""

import argparse
import json
import math
import os
import sys
import torch
import shortuuid
from tqdm import tqdm

LLAVA_SERIES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.abspath(os.path.join(LLAVA_SERIES_DIR, ".."))
for path in (LLAVA_SERIES_DIR, TRAIN_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.utils import (
    get_id_from_sample,
    get_image_path_from_sample,
    get_question_from_sample,
    init_model,
    load_image,
    setup_seeds,
)


def load_questions(question_file):
    """Load question records from a JSONL file."""
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


def get_chunk_manual(lst, num_chunks, chunk_idx):
    """Return the slice assigned to one chunk."""
    if not lst:
        return []
    chunk_size = math.ceil(len(lst) / num_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(lst))
    return lst[start_idx:end_idx]


def process_single_sample(line, sample_i, args, tokenizer, model, image_processor,
                          answers_file, all_logits_data):
    """
    Fallback path for one sample.

    Batch generation can fail on an individual example; this keeps the rest of
    the chunk usable by retrying samples one at a time.
    """
    idx = get_id_from_sample(line)
    question = get_question_from_sample(line)
    image_path = get_image_path_from_sample(line)

    images_orig = None
    image_sizes = None

    if image_path:
        if hasattr(args, "image_folder") and args.image_folder:
            image_path = os.path.join(args.image_folder, image_path)
        try:
            image = load_image(image_path)
            image_sizes = [image.size]
            image_tensor_orig = process_images([image], image_processor, model.config)[0]
            images_orig = image_tensor_orig.unsqueeze(0).half().cuda()
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return

        if model.config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=images_orig,
            image_sizes=image_sizes,
            do_sample=(args.temperature > 0),
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    output_ids = output.sequences
    text_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    ans_id = shortuuid.uuid()

    num_generated_tokens = len(output.scores)
    generated_ids = output_ids[0, -num_generated_tokens:]

    topk_values_list = []
    for step_score in output.scores:
        step_score = step_score.squeeze(0)
        vals, _ = torch.topk(step_score, k=args.top_k_logits)
        topk_values_list.append(vals.float().cpu())

    if topk_values_list:
        all_logits_data[ans_id] = {
            "topk_values": torch.stack(topk_values_list),          # [num_tokens, top_k]
            "generated_tokens": generated_ids.cpu(),                # [num_tokens]
        }

    with open(answers_file, "a", encoding="utf-8") as ans_file:
        ans_file.write(json.dumps({
            "question_id": idx,
            "image_path": image_path,
            "prompt": question,
            "text": text_output,
            "answer_id": ans_id,
            "metadata": {"sample_idx": sample_i},
        }, ensure_ascii=False) + "\n")


def eval_model(args):
    """Run inference for one question chunk and write answers/logits."""
    setup_seeds(args.seed)

    print(f"Loading model: {args.model_name}...")
    tokenizer, model, image_processor = init_model(args)

    print(f"Loading questions from {args.question_file}...")
    questions = load_questions(args.question_file)
    questions = get_chunk_manual(questions, args.num_chunks, args.chunk_idx)
    print(f"Processing chunk {args.chunk_idx}/{args.num_chunks}: {len(questions)} questions.")

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Each chunk writes logits separately; the shell wrapper merges them.
    merged_logits_file = os.path.join(
        os.path.dirname(answers_file),
        f"all_logits_chunk{args.chunk_idx}_merged.pt",
    )

    open(answers_file, "w").close()

    all_logits_data = {}

    # Batch samples within one question to avoid cross-image alignment issues.
    for line in tqdm(questions, desc="Processing questions"):
        idx = get_id_from_sample(line)
        question = get_question_from_sample(line)
        image_path = get_image_path_from_sample(line)

        images_orig = None
        image_sizes = None

        if image_path:
            if hasattr(args, "image_folder") and args.image_folder:
                image_path = os.path.join(args.image_folder, image_path)
            try:
                image = load_image(image_path)
                image_sizes = [image.size]
                image_tensor_orig = process_images([image], image_processor, model.config)[0]
                images_orig = image_tensor_orig.unsqueeze(0).half().cuda()
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            if model.config.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
            else:
                question = DEFAULT_IMAGE_TOKEN + "\n" + question

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        all_sample_indices = list(range(args.num_samples))
        batch_size = args.batch_size

        for batch_start in range(0, len(all_sample_indices), batch_size):
            batch_sample_indices = all_sample_indices[batch_start:batch_start + batch_size]
            current_batch_size = len(batch_sample_indices)

            batched_input_ids = input_ids.repeat(current_batch_size, 1)
            batched_images = (
                images_orig.repeat(current_batch_size, 1, 1, 1)
                if images_orig is not None else None
            )
            batched_image_sizes = image_sizes * current_batch_size if image_sizes else None

            try:
                with torch.inference_mode():
                    outputs = model.generate(
                        batched_input_ids,
                        images=batched_images,
                        image_sizes=batched_image_sizes,
                        do_sample=(args.temperature > 0),
                        temperature=args.temperature if args.temperature > 0 else None,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                output_ids = outputs.sequences          # [batch_size, seq_len]
                num_generated_tokens = len(outputs.scores)

                for batch_idx, sample_i in enumerate(batch_sample_indices):
                    generated_ids = output_ids[batch_idx, -num_generated_tokens:]
                    text_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    ans_id = shortuuid.uuid()

                    topk_values_list = []
                    topk_indices_list = []
                    for step_score in outputs.scores:
                        # step_score: [batch_size, vocab_size]
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
                # Retry failed batch samples individually.
                for sample_i in batch_sample_indices:
                    try:
                        process_single_sample(
                            line, sample_i, args, tokenizer, model,
                            image_processor, answers_file, all_logits_data,
                        )
                    except Exception as e2:
                        print(f"Error in fallback single processing: {e2}")

    if all_logits_data:
        print(f"Saving merged logits to {merged_logits_file}...")
        torch.save(all_logits_data, merged_logits_file)
        print(f"Saved {len(all_logits_data)} logits entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA multi-sample inference with logits collection")
    parser.add_argument("--model-name", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of samples to generate per question")
    parser.add_argument("--top-k-logits", type=int, default=50,
                        help="Number of top-k logits to save per generation step")
    parser.add_argument("--lora-name", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference per question")

    args = parser.parse_args()

    # Normalize model arguments for init_model.
    if args.lora_name:
        args.model_path = args.lora_name
        args.model_base = args.model_name
    else:
        args.model_path = args.model_name
        args.model_base = None

    eval_model(args)

