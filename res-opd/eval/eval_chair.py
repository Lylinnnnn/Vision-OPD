"""
CHAIR Evaluation for Res-OPD trained models.

Generates captions for COCO test images using either:
  - A vLLM-served model via OpenAI-compatible API (concurrent, resumable)
  - Direct HuggingFace model loading (sequential, no vLLM dependency)

Features:
  - Concurrent API inference via ThreadPoolExecutor (default 8 workers)
  - Checkpoint/resume: skips already-generated samples, appends incrementally
  - Auto-detects student_px from checkpoint directory name (e.g. s224 → 224)

Then computes CHAIR metrics (CHAIRi, CHAIRs, ObjPrec, ObjRecall, ObjF1, RepRate).

Usage:
    # Via vLLM API (recommended — start vllm serve first):
    python eval/eval_chair.py \
        --model-name Res-OPD-2B \
        --api-base http://localhost:8000/v1/ \
        --test-json data/test.json \
        --output-dir eval_results/

    # Direct model loading:
    python eval/eval_chair.py \
        --model-path /path/to/merged/checkpoint \
        --test-json data/test.json \
        --output-dir eval_results/ \
        --student-px 224 --target-px 448
"""

import argparse
import io
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
from robust_chair_analysis import (
    parse_official_synonyms,
    build_double_word_dict,
    compute_per_sample,
    aggregate_from_arrays,
)

COCO_INSTANCES_PATH = "/home/liuyanlin.lyl/notebook/data/COCO/coco2017val/annotations/instances_val2017.json"
COCO_CAPTIONS_PATH = "/home/liuyanlin.lyl/notebook/data/COCO/coco2017val/annotations/captions_val2017.json"

PROMPT_TEXT = (
    "Please describe this image in detail. Include all visible objects, "
    "their spatial relationships, colors, sizes, and any text or fine details."
)


def infer_student_px_from_path(path_str):
    """Try to extract student resolution from a checkpoint path like '...-s224-...'."""
    match = re.search(r"-s(\d+)", os.path.basename(path_str))
    if match:
        return int(match.group(1))
    # Also check parent directories
    for part in path_str.split(os.sep):
        match = re.search(r"-s(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="CHAIR evaluation for Res-OPD models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-path", help="Path to merged HuggingFace model checkpoint")
    group.add_argument("--api-base", help="vLLM API base URL (e.g. http://localhost:8000/v1/)")

    parser.add_argument("--model-name", default="Res-OPD", help="Model name for API mode")
    parser.add_argument("--test-json", required=True, help="Path to test.json from prepare_data.py")
    parser.add_argument("--output-dir", default="./eval_results", help="Output directory")
    parser.add_argument("--student-px", type=int, default=-1,
                        help="Student resolution for eval (-1 = auto-detect from ckpt name, 0 = original image)")
    parser.add_argument("--target-px", type=int, default=448,
                        help="Target resolution for resizing (only used with --student-px > 0)")
    parser.add_argument("--degradation-mode", choices=["square", "original"], default="square",
                        help="square: student_px -> target_px square; original: ratio down/up at original size")
    parser.add_argument("--student-ratio", type=float, default=1.0,
                        help="Original-mode degradation ratio (1.0 = original, 0.75/0.5/0.25 = down/up sample)")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-samples", type=int, default=0, help="Max test samples (0 = all)")
    parser.add_argument("--parallel-workers", type=int, default=8,
                        help="Number of concurrent API workers (only for --api-base mode)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries per sample on API failure")
    parser.add_argument("--save-logprobs", action="store_true",
                        help="Request and save generated-token logprobs in API/direct mode")
    parser.add_argument("--top-logprobs", type=int, default=5,
                        help="Top-k logprobs to save when --save-logprobs is enabled")
    return parser.parse_args()


def make_degraded_image(image_path, student_px, target_px):
    """Degrade image: resize down then back up to simulate low resolution."""
    img = Image.open(image_path).convert("RGB")
    small = img.resize((student_px, student_px), Image.LANCZOS)
    degraded = small.resize((target_px, target_px), Image.LANCZOS)
    return degraded


def make_ratio_degraded_image(image_path, ratio):
    """Degrade original image by ratio, then restore original width/height."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    if ratio <= 0:
        return Image.new("RGB", (width, height), color=(128, 128, 128))
    if ratio >= 1.0:
        return img
    small_size = (
        max(1, int(round(width * ratio))),
        max(1, int(round(height * ratio))),
    )
    small = img.resize(small_size, Image.LANCZOS)
    return small.resize((width, height), Image.LANCZOS)


def load_eval_image(image_path, degradation_mode, student_px, target_px, student_ratio):
    if degradation_mode == "original":
        return make_ratio_degraded_image(image_path, student_ratio)
    if student_px > 0:
        return make_degraded_image(image_path, student_px, target_px)
    return Image.open(image_path).convert("RGB")


def load_existing_results(eval_results_path):
    """Load previously completed results for checkpoint/resume support."""
    completed = {}
    if not os.path.exists(eval_results_path):
        return completed
    with open(eval_results_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                image_id = record.get("image_id")
                caption = record.get("generated_caption", "")
                # Skip empty or error captions
                if image_id is not None and caption and not caption.startswith("[ERROR]"):
                    completed[image_id] = record
            except json.JSONDecodeError:
                continue
    return completed


def _safe_mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _as_numeric(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _get_attr(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _entropy_from_logprobs(logprobs):
    if not logprobs:
        return None
    max_logprob = max(logprobs)
    weights = [math.exp(lp - max_logprob) for lp in logprobs]
    total = sum(weights)
    if total <= 0:
        return None
    probs = [w / total for w in weights]
    return -sum(p * math.log(max(p, 1e-12)) for p in probs)


def _summarize_token_logprobs(token_logprobs, top_logprobs):
    top1_top2_margins = []
    topk_entropies = []
    for topk in top_logprobs:
        vals = [entry.get("logprob") for entry in topk if entry.get("logprob") is not None]
        if len(vals) >= 2:
            sorted_vals = sorted(vals, reverse=True)
            top1_top2_margins.append(sorted_vals[0] - sorted_vals[1])
            topk_entropies.append(_entropy_from_logprobs(sorted_vals))
    return {
        "mean_token_logprob": _safe_mean(token_logprobs),
        "mean_top1_top2_logprob_margin": _safe_mean(top1_top2_margins),
        "mean_topk_entropy": _safe_mean(topk_entropies),
    }


def serialize_api_logprobs(choice):
    """Serialize OpenAI/vLLM chat logprobs into JSON-friendly fields."""
    logprobs = _get_attr(choice, "logprobs")
    content = _get_attr(logprobs, "content")
    if not content:
        return {}

    tokens = []
    token_logprobs = []
    top_logprobs = []
    for item in content:
        token = _get_attr(item, "token")
        logprob = _as_numeric(_get_attr(item, "logprob"))
        tokens.append(token)
        token_logprobs.append(logprob)

        serialized_topk = []
        for top_item in _get_attr(item, "top_logprobs", []) or []:
            top_token = _get_attr(top_item, "token")
            top_logprob = _as_numeric(_get_attr(top_item, "logprob"))
            if top_logprob is not None:
                serialized_topk.append({
                    "token": top_token,
                    "logprob": top_logprob,
                })
        top_logprobs.append(serialized_topk)

    fields = {
        "generated_tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
    }
    fields.update(_summarize_token_logprobs(token_logprobs, top_logprobs))
    return fields


def generate_one_api(thread_local, api_base, model_name, image_path, prompt,
                     max_new_tokens, student_px, target_px, max_retries,
                     degradation_mode="square", student_ratio=1.0,
                     save_logprobs=False, top_logprobs=5):
    """Generate a single caption via API with retry logic. Thread-safe."""
    import base64

    client = getattr(thread_local, "client", None)
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key="EMPTY", timeout=300)
        thread_local.client = client

    # Prepare image data
    if degradation_mode == "original" or student_px > 0:
        image = load_eval_image(
            image_path,
            degradation_mode,
            student_px,
            target_px,
            student_ratio,
        )
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{image_data}"

    for attempt in range(1, max_retries + 1):
        try:
            request_kwargs = {
                "model": model_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            }
            if save_logprobs:
                request_kwargs["logprobs"] = True
                request_kwargs["top_logprobs"] = top_logprobs

            response = client.chat.completions.create(**request_kwargs)
            choice = response.choices[0]
            result = {
                "generated_caption": choice.message.content,
            }
            if save_logprobs:
                result.update(serialize_api_logprobs(choice))
            return result
        except Exception as exc:
            if attempt == max_retries:
                return {"generated_caption": f"[ERROR] {exc}"}
            time.sleep(1.0 * attempt)
    return {"generated_caption": "[ERROR] unknown"}


def generate_via_model(model, processor, image_path, prompt, max_new_tokens,
                       device, student_px=0, target_px=448,
                       degradation_mode="square", student_ratio=1.0,
                       save_logprobs=False, top_logprobs=5):
    """Generate caption via direct model inference (sequential only)."""
    import torch
    from qwen_vl_utils import process_vision_info

    img = load_eval_image(
        image_path,
        degradation_mode,
        student_px,
        target_px,
        student_ratio,
    )

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=save_logprobs,
            output_scores=save_logprobs,
        )
    input_len = inputs["input_ids"].shape[1]
    sequences = output_ids.sequences if save_logprobs else output_ids
    generated_ids = sequences[0, input_len:]
    generated_text = processor.decode(generated_ids,
                                      skip_special_tokens=True)

    result = {"generated_caption": generated_text}
    if save_logprobs:
        token_logprobs = []
        serialized_top_logprobs = []
        for step, score in enumerate(output_ids.scores):
            if step >= generated_ids.shape[0]:
                break
            log_probs = torch.log_softmax(score[0].float(), dim=-1)
            token_id = int(generated_ids[step].item())
            token_logprobs.append(float(log_probs[token_id].item()))
            k = min(max(top_logprobs, 0), log_probs.shape[-1])
            if k > 0:
                top_vals, top_ids = torch.topk(log_probs, k=k)
                serialized_top_logprobs.append([
                    {
                        "token_id": int(tok_id.item()),
                        "token": processor.decode([int(tok_id.item())]),
                        "logprob": float(tok_lp.item()),
                    }
                    for tok_id, tok_lp in zip(top_ids, top_vals)
                ])
            else:
                serialized_top_logprobs.append([])
        result.update({
            "generated_token_ids": [int(x) for x in generated_ids.tolist()],
            "token_logprobs": token_logprobs,
            "top_logprobs": serialized_top_logprobs,
        })
        result.update(_summarize_token_logprobs(
            token_logprobs, serialized_top_logprobs
        ))
    return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Resolve student degradation ---
    if args.degradation_mode == "original":
        print(
            f"degradation_mode=original: student_ratio={args.student_ratio} "
            "(down/up sample at original width/height)"
        )
    elif args.student_px < 0:
        # Auto-detect from model path or output dir
        detected = None
        if args.model_path:
            detected = infer_student_px_from_path(args.model_path)
        if detected is None:
            detected = infer_student_px_from_path(args.output_dir)
        if detected is not None:
            args.student_px = detected
            print(f"Auto-detected student_px={args.student_px} from path")
        else:
            args.student_px = 0
            print("Could not auto-detect student_px, using 0 (original image)")
    elif args.student_px == 0:
        print("student_px=0: using original image (no degradation)")
    else:
        print(f"student_px={args.student_px} → target_px={args.target_px}")

    # Load test data
    with open(args.test_json) as f:
        test_samples = json.load(f)

    if args.max_samples > 0:
        test_samples = test_samples[:args.max_samples]

    # --- Checkpoint / resume ---
    eval_results_path = os.path.join(args.output_dir, "eval_results.jsonl")
    completed = load_existing_results(eval_results_path)
    todo_samples = [s for s in test_samples if s["image_id"] not in completed]

    print(f"Evaluating {len(test_samples)} test samples "
          f"({len(completed)} already done, {len(todo_samples)} remaining)")

    if not todo_samples:
        print("All samples already completed. Computing metrics from checkpoint.")
    else:
        # Setup generation backend
        model, processor, device = None, None, None
        if args.model_path:
            import torch
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            print(f"Loading model from {args.model_path} ...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_path, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True)
            model.eval()
            processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
            print("Model loaded.")

        start_time = time.time()
        done_count = len(completed)

        if args.api_base:
            # --- Concurrent API mode ---
            thread_local = threading.local()
            write_lock = threading.Lock()

            def process_one(sample):
                image_path = sample["image_path"]
                if not os.path.exists(image_path):
                    return None
                caption = generate_one_api(
                    thread_local, args.api_base, args.model_name,
                    image_path, PROMPT_TEXT, args.max_new_tokens,
                    args.student_px, args.target_px, args.max_retries,
                    args.degradation_mode, args.student_ratio,
                    args.save_logprobs, args.top_logprobs,
                )
                result = {
                    "image_id": sample["image_id"],
                    "file_name": sample["file_name"],
                    "image_path": sample.get("image_path", image_path),
                    "gt_captions": sample.get("captions", []),
                    "gt_objects": sample.get("objects", []),
                    "eval_degradation_mode": args.degradation_mode,
                    "eval_student_px": args.student_px,
                    "eval_target_px": args.target_px,
                    "eval_student_ratio": args.student_ratio,
                }
                result.update(caption)
                return result

            with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor, \
                 open(eval_results_path, "a") as f_out:
                future_to_sample = {
                    executor.submit(process_one, s): s for s in todo_samples
                }
                for future in as_completed(future_to_sample):
                    result = future.result()
                    if result is None:
                        continue
                    with write_lock:
                        f_out.write(json.dumps(result) + "\n")
                        f_out.flush()
                        done_count += 1
                    if done_count % 20 == 0:
                        elapsed = time.time() - start_time
                        print(f"  Generated {done_count}/{len(test_samples)} "
                              f"({elapsed:.0f}s, {done_count/elapsed:.1f} samples/s)")
        else:
            # --- Sequential direct-model mode ---
            with open(eval_results_path, "a") as f_out:
                for sample in todo_samples:
                    image_path = sample["image_path"]
                    if not os.path.exists(image_path):
                        print(f"  Warning: {image_path} not found, skipping.")
                        continue
                    generation = generate_via_model(
                        model, processor, image_path, PROMPT_TEXT,
                        args.max_new_tokens, device,
                        args.student_px, args.target_px,
                        args.degradation_mode, args.student_ratio,
                        args.save_logprobs, args.top_logprobs,
                    )
                    result = {
                        "image_id": sample["image_id"],
                        "file_name": sample["file_name"],
                        "image_path": sample.get("image_path", image_path),
                        "gt_captions": sample.get("captions", []),
                        "gt_objects": sample.get("objects", []),
                        "eval_degradation_mode": args.degradation_mode,
                        "eval_student_px": args.student_px,
                        "eval_target_px": args.target_px,
                        "eval_student_ratio": args.student_ratio,
                    }
                    result.update(generation)
                    f_out.write(json.dumps(result) + "\n")
                    f_out.flush()
                    done_count += 1
                    if done_count % 20 == 0:
                        elapsed = time.time() - start_time
                        print(f"  Generated {done_count}/{len(test_samples)} "
                              f"({elapsed:.0f}s, {done_count/elapsed:.1f} samples/s)")

        elapsed = time.time() - start_time
        print(f"\nGeneration complete: {done_count} samples in {elapsed:.0f}s")

    # --- Load all results (including resumed ones) for metric computation ---
    all_results = load_existing_results(eval_results_path)
    results_list = list(all_results.values())
    print(f"Computing CHAIR metrics on {len(results_list)} samples ...")

    mscoco_objects, inverse_synonym_dict = parse_official_synonyms()
    double_word_dict = build_double_word_dict()

    eval_records = []
    for r in results_list:
        eval_records.append({
            "image_id": r["image_id"],
            "generated_text": r["generated_caption"],
            "gt_objects": set(r["gt_objects"]),
        })

    sample_dicts = compute_per_sample(
        eval_records, mscoco_objects, inverse_synonym_dict, double_word_dict)

    indices = list(sample_dicts.keys())
    metrics = aggregate_from_arrays(sample_dicts, indices)

    # Print results
    print("\n" + "=" * 70)
    print("  CHAIR Evaluation Results")
    print("=" * 70)
    print(f"  Samples:    {len(results_list)}")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:12s}: {val:.4f}")
    print("=" * 70)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "chair_metrics.json")
    metrics["num_samples"] = len(results_list)
    metrics["student_px"] = args.student_px
    metrics["target_px"] = args.target_px
    metrics["degradation_mode"] = args.degradation_mode
    metrics["student_ratio"] = args.student_ratio
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
