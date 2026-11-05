#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image


BASELINES_DIR = Path(__file__).resolve().parent
RES_OPD_ROOT = BASELINES_DIR.parent
EVAL_DIR = RES_OPD_ROOT / "eval"
OPERA_SOURCE = RES_OPD_ROOT / "third_party" / "baselines" / "OPERA"
sys.path.insert(0, str(BASELINES_DIR))
sys.path.insert(0, str(EVAL_DIR))

from eval_chair import extract_final_response_text, load_existing_results, remap_coco_image_paths
from opera_qwen3vl import locate_image_span, opera_generate
from robust_chair_analysis import (
    aggregate_from_arrays,
    build_double_word_dict,
    compute_per_sample,
    parse_official_synonyms,
)


DEFAULT_PROMPT = "Describe the image."


def parse_args():
    parser = argparse.ArgumentParser(description="OPERA evaluation on official CHAIR")
    parser.add_argument("--model-family", choices=["qwen3vl", "internvl3"], default="qwen3vl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--scale-factor", type=float, default=50.0)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--num-attn-candidates", type=int, default=5)
    parser.add_argument("--penalty-weights", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--allow-unvalidated-full-run", action="store_true")
    return parser.parse_args()


def opera_config(args):
    registry = json.loads((BASELINES_DIR / "registry.json").read_text())
    config = {
        "method": "opera",
        "source_commit": registry["baselines"]["opera"]["commit"],
        "model_name": Path(args.model_path).name,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "scale_factor": args.scale_factor,
        "threshold": args.threshold,
        "num_attn_candidates": args.num_attn_candidates,
        "penalty_weights": args.penalty_weights,
        "window_size": args.window_size,
        "torch_dtype": args.torch_dtype,
        "transformers_version": "5.5.0",
        "attention_implementation": "eager",
        "degradation_mode": "original",
        "student_ratio": 1.0,
    }
    if args.model_family == "internvl3":
        config["model_family"] = args.model_family
    return config


def validate_existing(completed, config):
    expected = {
        "opera_source_commit": config["source_commit"],
        "opera_num_beams": config["num_beams"],
        "opera_scale_factor": config["scale_factor"],
        "opera_threshold": config["threshold"],
        "opera_num_attn_candidates": config["num_attn_candidates"],
        "opera_penalty_weights": config["penalty_weights"],
        "opera_window_size": config["window_size"],
        "eval_prompt": config["prompt"],
    }
    if "model_family" in config:
        expected["model_family"] = config["model_family"]
    for image_id, record in completed.items():
        mismatches = [key for key, value in expected.items() if record.get(key) != value]
        if mismatches:
            raise RuntimeError(
                f"Refusing to mix incompatible OPERA results for image {image_id}: "
                f"mismatched fields={mismatches}"
            )


def build_inputs(processor, image, prompt, model_family="qwen3vl", device=None, dtype=None):
    if model_family == "internvl3":
        from internvl3_inputs import build_inputs as build_internvl_inputs

        return build_internvl_inputs(processor, image, prompt, device, dtype)
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    kwargs = {
        "text": [text],
        "images": image_inputs,
        "padding": True,
        "return_tensors": "pt",
    }
    if video_inputs:
        kwargs["videos"] = video_inputs
    return processor(**kwargs)


def generate_caption(model, processor, device, image, args):
    import torch

    inputs = build_inputs(
        processor, image, args.prompt, args.model_family,
        device if args.model_family == "internvl3" else None,
        model.dtype if args.model_family == "internvl3" else None,
    )
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    input_length = inputs["input_ids"].shape[1]
    image_token_id = int(model.config.image_token_id)
    key_position = locate_image_span(inputs["input_ids"], image_token_id)
    runtime = {}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            custom_generate=opera_generate,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=args.num_beams,
            num_return_sequences=1,
            output_attentions=True,
            return_dict_in_generate=False,
            use_cache=True,
            cache_implementation="dynamic",
            opera_key_position=key_position,
            opera_scale_factor=args.scale_factor,
            opera_threshold=args.threshold,
            opera_num_attn_candidates=args.num_attn_candidates,
            opera_penalty_weights=args.penalty_weights,
            opera_window_size=args.window_size,
            opera_runtime=runtime,
        )
    generated_ids = output_ids[0, input_length:]
    caption = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "generated_caption": caption,
        "generated_token_ids": [int(token_id) for token_id in generated_ids.tolist()],
        "generated_token_count": int(generated_ids.numel()),
        "thinking_status": "not_requested",
        "final_answer_available": True,
        "opera_key_position": key_position,
        "opera_rollback_count": runtime.get("rollback_count", 0),
        "opera_rollback_events": runtime.get("rollback_events", []),
        "opera_rejected_continuations": runtime.get("rejected_continuations", 0),
    }


def score_results(results, output_dir, config, shard_count, shard_index):
    objects, inverse = parse_official_synonyms()
    double_words = build_double_word_dict()
    records = [
        {
            "image_id": row["image_id"],
            "generated_text": extract_final_response_text(row["generated_caption"]),
            "gt_objects": set(row.get("gt_objects", [])),
            "gt_captions": row.get("gt_captions", []),
        }
        for row in results
        if row.get("generated_caption")
        and not str(row["generated_caption"]).startswith("[ERROR]")
    ]
    per_sample = compute_per_sample(records, objects, inverse, double_words)
    metrics = aggregate_from_arrays(per_sample, list(per_sample))
    metrics.update(
        {
            "num_samples": len(records),
            "total_results": len(results),
            "invalid_final_answer": 0,
            "unclosed_thinking": 0,
            "final_answer_valid_rate": len(records) / len(results) if results else 0.0,
            "strict_final_answer": False,
            "merged_from_shards": False,
            "shard_count": shard_count,
            "shard_index": shard_index,
            "eval_prompt": config["prompt"],
            "decoding_method": "opera",
            "opera_config": config,
        }
    )
    (output_dir / "chair_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )


def main():
    args = parse_args()
    if not OPERA_SOURCE.is_dir():
        raise RuntimeError("OPERA submodule is missing; run git submodule update --init --recursive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard-count/shard-index")
    if args.num_beams <= 1:
        raise ValueError("OPERA requires num-beams > 1")
    if args.num_attn_candidates < 1:
        raise ValueError("num-attn-candidates must be positive")
    if args.threshold < 0:
        raise ValueError("threshold must be nonnegative")
    if args.window_size < 1:
        raise ValueError("window-size must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if args.model_family == "internvl3" and (args.max_samples == 0 or args.max_samples > 5):
        if not args.allow_unvalidated_full_run:
            raise ValueError("InternVL3 OPERA needs a <=5-image GPU smoke before full evaluation")
    if args.model_family == "internvl3":
        from internvl3_inputs import require_checkpoint

        require_checkpoint(args.model_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = opera_config(args)
    (output_dir / "opera_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2)
    )
    samples = remap_coco_image_paths(json.loads(Path(args.test_json).read_text()))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    samples = [
        sample
        for index, sample in enumerate(samples)
        if index % args.shard_count == args.shard_index
    ]
    result_path = output_dir / "eval_results.jsonl"
    result_path.touch(exist_ok=True)
    completed = load_existing_results(str(result_path))
    validate_existing(completed, config)
    pending = [sample for sample in samples if sample["image_id"] not in completed]
    print(
        f"OPERA shard {args.shard_index}/{args.shard_count}: "
        f"samples={len(samples)} completed={len(completed)} pending={len(pending)}",
        flush=True,
    )

    if pending:
        import torch
        import transformers
        from transformers import AutoProcessor

        if args.model_family == "internvl3":
            from transformers import InternVLForConditionalGeneration as ModelClass
        else:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass

        if transformers.__version__ != "5.5.0":
            raise RuntimeError(
                f"OPERA adapter is pinned to transformers 5.5.0, got {transformers.__version__}"
            )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("OPERA evaluation requires a CUDA GPU")
        dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
        model = ModelClass.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        start = time.time()
        with result_path.open("a", encoding="utf-8") as output:
            for offset, sample in enumerate(pending, 1):
                image = Image.open(sample["image_path"]).convert("RGB")
                generation = generate_caption(model, processor, device, image, args)
                row = {
                    "image_id": sample["image_id"],
                    "file_name": sample["file_name"],
                    "image_path": sample["image_path"],
                    "gt_captions": sample.get("captions", []),
                    "gt_objects": sample.get("objects", []),
                    "eval_degradation_mode": "original",
                    "eval_student_ratio": 1.0,
                    "eval_prompt": args.prompt,
                    "decoding_method": "opera",
                    "opera_source_commit": config["source_commit"],
                    "opera_num_beams": args.num_beams,
                    "opera_scale_factor": args.scale_factor,
                    "opera_threshold": args.threshold,
                    "opera_num_attn_candidates": args.num_attn_candidates,
                    "opera_penalty_weights": args.penalty_weights,
                    "opera_window_size": args.window_size,
                }
                if args.model_family == "internvl3":
                    row["model_family"] = args.model_family
                row.update(generation)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    f"[{offset}/{len(pending)}] image_id={sample['image_id']} "
                    f"tokens={generation['generated_token_count']} "
                    f"rollbacks={generation['opera_rollback_count']} "
                    f"elapsed={time.time() - start:.1f}s",
                    flush=True,
                )

    selected_ids = {sample["image_id"] for sample in samples}
    all_results = load_existing_results(str(result_path))
    results = [record for image_id, record in all_results.items() if image_id in selected_ids]
    missing_ids = selected_ids - set(all_results)
    if missing_ids:
        raise RuntimeError(f"OPERA shard is incomplete; missing image ids: {sorted(missing_ids)[:20]}")
    if results:
        score_results(results, output_dir, config, args.shard_count, args.shard_index)
    print(f"OPERA CHAIR shard complete: {output_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
