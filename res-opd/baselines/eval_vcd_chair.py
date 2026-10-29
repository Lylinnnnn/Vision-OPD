#!/usr/bin/env python3

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image


BASELINES_DIR = Path(__file__).resolve().parent
RES_OPD_ROOT = BASELINES_DIR.parent
EVAL_DIR = RES_OPD_ROOT / "eval"
VCD_SOURCE = RES_OPD_ROOT / "third_party" / "baselines" / "VCD"
sys.path.insert(0, str(BASELINES_DIR))
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(VCD_SOURCE))

from eval_chair import (  # noqa: E402
    extract_final_response_text,
    load_existing_results,
    remap_coco_image_paths,
)
from robust_chair_analysis import (  # noqa: E402
    aggregate_from_arrays,
    build_double_word_dict,
    compute_per_sample,
    parse_official_synonyms,
)
from vcd_qwen3vl import (  # noqa: E402
    VCDLogitsProcessor,
    replace_second_image_with_diffusion_noise,
)


DEFAULT_PROMPT = "Describe the image."


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3VL VCD evaluation on official CHAIR")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--noise-step", type=int, default=500)
    parser.add_argument("--decode-mode", choices=["sample", "greedy"], default="sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    return parser.parse_args()


def stable_seed(base_seed, image_id, stream):
    payload = f"{base_seed}:{image_id}:{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def vcd_config(args):
    registry = json.loads((BASELINES_DIR / "registry.json").read_text())
    return {
        "method": "vcd",
        "source_commit": registry["baselines"]["vcd"]["commit"],
        "model_name": Path(args.model_path).name,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "alpha": args.alpha,
        "beta": args.beta,
        "noise_step": args.noise_step,
        "decode_mode": args.decode_mode,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "seed_scope": "deterministic_per_image_and_stream",
        "torch_dtype": args.torch_dtype,
        "degradation_mode": "original",
        "student_ratio": 1.0,
    }


def validate_existing(completed, config):
    keys = (
        "vcd_source_commit",
        "vcd_alpha",
        "vcd_beta",
        "vcd_noise_step",
        "vcd_decode_mode",
        "vcd_temperature",
        "vcd_top_p",
        "vcd_top_k",
        "eval_prompt",
    )
    expected = {
        "vcd_source_commit": config["source_commit"],
        "vcd_alpha": config["alpha"],
        "vcd_beta": config["beta"],
        "vcd_noise_step": config["noise_step"],
        "vcd_decode_mode": config["decode_mode"],
        "vcd_temperature": config["temperature"],
        "vcd_top_p": config["top_p"],
        "vcd_top_k": config["top_k"],
        "eval_prompt": config["prompt"],
    }
    for image_id, record in completed.items():
        mismatches = [key for key in keys if record.get(key) != expected[key]]
        if mismatches:
            raise RuntimeError(
                f"Refusing to mix incompatible VCD results for image {image_id}: "
                f"mismatched fields={mismatches}"
            )


def build_inputs(processor, image, prompt, noise_step, noise_seed):
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
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    images = list(image_inputs) * 2
    videos = list(video_inputs) * 2 if video_inputs else None
    processor_kwargs = {
        "text": [text, text],
        "images": images,
        "padding": True,
        "return_tensors": "pt",
    }
    if videos:
        processor_kwargs["videos"] = videos
    inputs = processor(**processor_kwargs)
    if inputs["input_ids"].shape[0] != 2 or not inputs["input_ids"][0].equal(
        inputs["input_ids"][1]
    ):
        raise RuntimeError("VCD branches produced different text inputs")
    return replace_second_image_with_diffusion_noise(inputs, noise_step, noise_seed)


def generate_caption(model, processor, device, image, args, image_id):
    import torch
    from transformers.generation.logits_process import LogitsProcessorList

    noise_seed = stable_seed(args.seed, image_id, "noise")
    sample_seed = stable_seed(args.seed, image_id, "sample")
    inputs = build_inputs(
        processor,
        image,
        args.prompt,
        args.noise_step,
        noise_seed,
    )
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    input_length = inputs["input_ids"].shape[1]
    logits_processor = VCDLogitsProcessor(
        alpha=args.alpha,
        beta=args.beta,
        decode_mode=args.decode_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=sample_seed,
    )
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([logits_processor]),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    if output_ids.shape[0] != 2 or not output_ids[0].equal(output_ids[1]):
        raise RuntimeError("VCD original/noisy branches diverged during generation")
    generated_ids = output_ids[0, input_length:]
    caption = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "generated_caption": caption,
        "generated_token_ids": [int(token_id) for token_id in generated_ids.tolist()],
        "generated_token_count": int(generated_ids.numel()),
        "thinking_status": "not_requested",
        "final_answer_available": True,
        "vcd_noise_seed": noise_seed,
        "vcd_sample_seed": sample_seed,
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
            "decoding_method": "vcd",
            "vcd_config": config,
        }
    )
    (output_dir / "chair_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )


def main():
    args = parse_args()
    if not VCD_SOURCE.is_dir():
        raise RuntimeError(
            "VCD submodule is missing; run git submodule update --init --recursive"
        )
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard-count/shard-index")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vcd_config(args)
    (output_dir / "vcd_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2)
    )

    samples = json.loads(Path(args.test_json).read_text())
    samples = remap_coco_image_paths(samples)
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
        f"VCD shard {args.shard_index}/{args.shard_count}: "
        f"samples={len(samples)} completed={len(completed)} pending={len(pending)}"
    )

    if pending:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("VCD Qwen3VL evaluation requires a CUDA GPU")
        dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )

        start = time.time()
        with result_path.open("a", encoding="utf-8") as output:
            for offset, sample in enumerate(pending, 1):
                image = Image.open(sample["image_path"]).convert("RGB")
                generation = generate_caption(
                    model,
                    processor,
                    device,
                    image,
                    args,
                    sample["image_id"],
                )
                row = {
                    "image_id": sample["image_id"],
                    "file_name": sample["file_name"],
                    "image_path": sample["image_path"],
                    "gt_captions": sample.get("captions", []),
                    "gt_objects": sample.get("objects", []),
                    "eval_degradation_mode": "original",
                    "eval_student_ratio": 1.0,
                    "eval_prompt": args.prompt,
                    "decoding_method": "vcd",
                    "vcd_source_commit": config["source_commit"],
                    "vcd_alpha": args.alpha,
                    "vcd_beta": args.beta,
                    "vcd_noise_step": args.noise_step,
                    "vcd_decode_mode": args.decode_mode,
                    "vcd_temperature": args.temperature,
                    "vcd_top_p": args.top_p,
                    "vcd_top_k": args.top_k,
                }
                row.update(generation)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                elapsed = time.time() - start
                print(
                    f"[{offset}/{len(pending)}] image_id={sample['image_id']} "
                    f"tokens={generation['generated_token_count']} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    selected_ids = {sample["image_id"] for sample in samples}
    all_results = load_existing_results(str(result_path))
    results = [
        record
        for image_id, record in all_results.items()
        if image_id in selected_ids
    ]
    missing_ids = selected_ids - set(all_results)
    if missing_ids:
        raise RuntimeError(
            f"VCD shard is incomplete; missing image ids: {sorted(missing_ids)[:20]}"
        )
    if results:
        score_results(results, output_dir, config, args.shard_count, args.shard_index)
    print(f"VCD CHAIR shard complete: {output_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
