#!/usr/bin/env python3
"""HALC/Qwen3VL adaptation with the shared CHAIR scorer."""

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image


BASELINES_DIR = Path(__file__).resolve().parent
RES_OPD_ROOT = BASELINES_DIR.parent
HALC_SOURCE = RES_OPD_ROOT / "third_party" / "baselines" / "HALC"
ADAPTER_REVISION = "focal_contrast_paper_eq4_blip_itm_v2"
sys.path.insert(0, str(BASELINES_DIR))
sys.path.insert(0, str(RES_OPD_ROOT / "eval"))
sys.path.insert(0, str(HALC_SOURCE / "decoder_zoo" / "GroundingDINO"))

from eval_chair import extract_final_response_text, load_existing_results, remap_coco_image_paths
from robust_chair_analysis import (
    aggregate_from_arrays,
    build_double_word_dict,
    compute_per_sample,
    parse_official_synonyms,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Experimental HALC/Qwen3VL CHAIR smoke")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-json", default=str(RES_OPD_ROOT / "data" / "test_1000.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--detector-config", required=True)
    parser.add_argument("--detector-weights", required=True)
    parser.add_argument("--blip-model-path", required=True)
    parser.add_argument("--prompt", default="Describe the image.")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--expand-ratio", type=float, default=0.6)
    parser.add_argument("--box-threshold", type=float, default=0.4)
    parser.add_argument("--contrast-weight", type=float, default=0.05)
    parser.add_argument("--skip-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-trigger", action="store_true")
    parser.add_argument("--require-change", action="store_true")
    parser.add_argument("--allow-unvalidated-full-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def preflight(args):
    import importlib
    import importlib.util
    import subprocess

    errors = []
    if not (HALC_SOURCE / "README.md").is_file():
        raise RuntimeError("HALC source missing; initialize the pinned Git submodule")
    registry = json.loads((BASELINES_DIR / "registry.json").read_text())
    expected = registry["baselines"]["halc"]["commit"]
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HALC_SOURCE, text=True
    ).strip()
    if actual != expected:
        raise RuntimeError(f"HALC source revision mismatch: expected {expected}, got {actual}")
    paths = {
        "Qwen3VL model": (args.model_path, True),
        "test JSON": (args.test_json, False),
        "GroundingDINO config": (args.detector_config, False),
        "GroundingDINO weights": (args.detector_weights, False),
        "BLIP model": (args.blip_model_path, True),
    }
    for label, (path, is_dir) in paths.items():
        item = Path(path)
        if not (item.is_dir() if is_dir else item.is_file()):
            errors.append(f"{label} missing or wrong type: {path}")
        elif not is_dir and item.stat().st_size == 0:
            errors.append(f"{label} is empty: {path}")
    if Path(args.test_json).is_file():
        try:
            if not json.loads(Path(args.test_json).read_text()):
                errors.append(f"test JSON has no samples: {args.test_json}")
        except (OSError, ValueError) as exc:
            errors.append(f"Cannot read test JSON: {exc}")
    for label, path in (("Qwen3VL model", args.model_path), ("BLIP model", args.blip_model_path)):
        root = Path(path)
        if root.is_dir():
            if not (root / "config.json").is_file():
                errors.append(f"{label} has no config.json: {root}")
            elif label == "BLIP model":
                try:
                    architectures = json.loads((root / "config.json").read_text()).get("architectures") or []
                    if "BlipForImageTextRetrieval" not in architectures:
                        errors.append(
                            f"BLIP model must be an image-text retrieval checkpoint, got {architectures}; "
                            "use Salesforce/blip-itm-base-coco"
                        )
                except (OSError, ValueError, TypeError) as exc:
                    errors.append(f"Cannot read BLIP model architecture: {exc}")
            weight_names = (
                "model.safetensors", "model.safetensors.index.json",
                "pytorch_model.bin", "pytorch_model.bin.index.json",
            )
            if not any((root / name).is_file() for name in weight_names):
                errors.append(f"{label} has no model weights or weight index: {root}")
            for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
                index_path = root / index_name
                if not index_path.is_file():
                    continue
                try:
                    shards = set(json.loads(index_path.read_text())["weight_map"].values())
                    missing_shards = sorted(name for name in shards if not (root / name).is_file())
                    if missing_shards:
                        errors.append(f"{label} missing {len(missing_shards)} weight shards: {missing_shards[:3]}")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f"Cannot read {label} weight index: {exc}")
    weights = Path(args.detector_weights)
    if weights.is_file() and weights.stat().st_size < 1_000_000:
        errors.append(f"GroundingDINO weights look truncated: {weights}")

    modules = (
        "torch", "torchvision", "numpy", "PIL", "cv2", "transformers",
        "qwen_vl_utils", "spacy", "timm", "addict", "yapf",
        "supervision", "groundingdino", "pycocotools", "yaml",
        "scipy", "matplotlib", "requests",
    )
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        errors.append("Missing Python packages: " + ", ".join(missing))
    for name in ("torch", "torchvision.ops", "cv2", "timm", "supervision",
                 "groundingdino._C", "groundingdino.util.inference"):
        if name.split(".")[0] in missing:
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f"Cannot import {name}: {type(exc).__name__}: {exc}")

    if "spacy" not in missing:
        try:
            importlib.import_module("spacy").load("en_core_web_sm")
        except Exception as exc:
            errors.append(f"Cannot load spaCy en_core_web_sm: {type(exc).__name__}: {exc}")
    transformers = None
    if "transformers" not in missing:
        try:
            transformers = importlib.import_module("transformers")
            for name in ("Qwen3VLForConditionalGeneration", "AutoProcessor", "BlipForImageTextRetrieval", "BlipProcessor"):
                getattr(transformers, name)
        except Exception as exc:
            errors.append(f"Transformers model imports failed: {type(exc).__name__}: {exc}")
        else:
            for label, path, processor in (
                ("Qwen3VL", args.model_path, transformers.AutoProcessor),
                ("BLIP", args.blip_model_path, transformers.BlipProcessor),
            ):
                if Path(path).is_dir():
                    try:
                        kwargs = {"local_files_only": True}
                        if label == "Qwen3VL":
                            kwargs["trust_remote_code"] = True
                        processor.from_pretrained(path, **kwargs)
                    except Exception as exc:
                        errors.append(f"Cannot load {label} processor: {type(exc).__name__}: {exc}")
    if "qwen_vl_utils" not in missing:
        try:
            getattr(importlib.import_module("qwen_vl_utils"), "process_vision_info")
        except Exception as exc:
            errors.append(f"Cannot import process_vision_info: {type(exc).__name__}: {exc}")
    if "torch" not in missing:
        try:
            torch = importlib.import_module("torch")
            if not torch.cuda.is_available():
                errors.append("PyTorch CUDA is unavailable")
        except Exception as exc:
            errors.append(f"PyTorch CUDA check failed: {type(exc).__name__}: {exc}")
    if transformers is not None and not errors:
        try:
            from halc_qwen3vl import install_groundingdino_bert_compatibility
            from groundingdino.models.GroundingDINO import groundingdino

            install_groundingdino_bert_compatibility()
            config = transformers.BertConfig(
                vocab_size=32, hidden_size=32, num_hidden_layers=1,
                num_attention_heads=4, intermediate_size=64,
            )
            bert = transformers.BertModel(config).eval()
            wrapped = groundingdino.BertModelWarper(bert).eval()
            if set(bert.state_dict()) != set(wrapped.state_dict()):
                raise RuntimeError("GroundingDINO BERT checkpoint keys changed")
            mask = torch.tensor([[[1, 1, 0], [1, 1, 0], [0, 0, 1]]], dtype=torch.bool)
            with torch.inference_mode():
                output = wrapped(
                    input_ids=torch.tensor([[1, 2, 3]]), attention_mask=mask
                )
            if output.last_hidden_state.shape != (1, 3, 32) or not torch.isfinite(output.last_hidden_state).all():
                raise RuntimeError("GroundingDINO BERT forward returned invalid values")
        except Exception as exc:
            errors.append(f"GroundingDINO BERT compatibility smoke failed: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("HALC_PREFLIGHT=FAIL\n- " + "\n- ".join(errors))
    print(f"HALC_PREFLIGHT=PASS source={actual} transformers={transformers.__version__}", flush=True)
    return actual


def main():
    args = parse_args()
    if args.max_samples < 0 or args.max_new_tokens < 1:
        raise ValueError("max-samples must be nonnegative and max-new-tokens positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard-count/shard-index")
    if (args.max_samples == 0 or args.max_samples > 5) and not args.allow_unvalidated_full_run:
        raise ValueError("HALC adapter is unvalidated; limit smoke to <=5 images")
    source_commit = preflight(args)
    if args.preflight_only:
        return

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from halc_qwen3vl import HALCConfig, HALCDecoder

    if not torch.cuda.is_available():
        raise RuntimeError("HALC/Qwen3VL smoke requires a GPU")
    config = HALCConfig(
        beam_size=args.beam_size,
        context_window=args.context_window,
        candidate_count=args.candidate_count,
        expand_ratio=args.expand_ratio,
        box_threshold=args.box_threshold,
        contrast_weight=args.contrast_weight,
        skip_rate=args.skip_rate,
        max_new_tokens=args.max_new_tokens,
    )
    config.validate()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_record = {
        "method": "halc_qwen3vl_adaptation",
        "adapter_revision": ADAPTER_REVISION,
        "source_commit": source_commit,
        "model_path": str(Path(args.model_path).resolve()),
        "detector_weights": str(Path(args.detector_weights).resolve()),
        "blip_model_path": str(Path(args.blip_model_path).resolve()),
        "prompt": args.prompt,
        "seed": args.seed,
        "parameters": vars(config),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
    }
    config_path = output_dir / "halc_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config_record:
        raise RuntimeError("Refusing to mix HALC runs with different configurations")
    config_path.write_text(json.dumps(config_record, ensure_ascii=False, indent=2))

    samples = remap_coco_image_paths(json.loads(Path(args.test_json).read_text()))
    if args.max_samples:
        samples = samples[:args.max_samples]
    samples = [sample for index, sample in enumerate(samples)
               if index % args.shard_count == args.shard_index]
    result_path = output_dir / "eval_results.jsonl"
    completed = load_existing_results(str(result_path)) if result_path.exists() else {}
    for image_id, row in completed.items():
        if row.get("halc_config") != config_record:
            raise RuntimeError(f"Refusing to mix incompatible HALC results for image {image_id}")
    pending = [sample for sample in samples if sample["image_id"] not in completed]
    print(f"HALC shard={args.shard_index}/{args.shard_count} "
          f"samples={len(samples)} pending={len(pending)}", flush=True)
    if pending:
        device = torch.device("cuda:0")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        decoder = HALCDecoder(
            model, processor, args.detector_config, args.detector_weights,
            args.blip_model_path, config, device,
        )
        started = time.monotonic()
        with result_path.open("a", encoding="utf-8") as output:
            for offset, sample in enumerate(pending, 1):
                with Image.open(sample["image_path"]) as source:
                    source.verify()
                generation = decoder.generate(
                    sample["image_path"], prompt=args.prompt,
                    seed=args.seed + int(sample["image_id"]),
                )
                row = {
                    "image_id": sample["image_id"],
                    "file_name": sample["file_name"],
                    "image_path": sample["image_path"],
                    "gt_captions": sample.get("captions", []),
                    "gt_objects": sample.get("objects", []),
                    "eval_prompt": args.prompt,
                    "eval_degradation_mode": "original",
                    "eval_student_ratio": 1.0,
                    "decoding_method": "halc_qwen3vl_adaptation",
                    "halc_source_commit": source_commit,
                    "halc_config": config_record,
                    **generation,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(f"[{offset}/{len(pending)}] image={sample['image_id']} "
                      f"tokens={generation['generated_token_count']} "
                      f"triggered={generation['halc_triggered_count']} "
                      f"changed={generation['halc_changed_token_count']} "
                      f"elapsed={time.monotonic() - started:.1f}s", flush=True)

    completed = load_existing_results(str(result_path))
    missing_ids = {sample["image_id"] for sample in samples} - set(completed)
    if missing_ids:
        raise RuntimeError(f"HALC shard incomplete; missing image ids: {sorted(missing_ids)[:20]}")
    rows = [completed[sample["image_id"]] for sample in samples]
    if args.require_trigger and not any(row["halc_triggered_count"] > 0 for row in rows):
        raise RuntimeError("HALC_SMOKE=FAIL: no focal correction was triggered")
    if args.require_change and not any(row["halc_changed_token_count"] > 0 for row in rows):
        raise RuntimeError("HALC_SMOKE=FAIL: no generated token was changed")
    objects, inverse = parse_official_synonyms()
    records = [{
        "image_id": row["image_id"],
        "generated_text": extract_final_response_text(row["generated_caption"]),
        "gt_objects": set(row.get("gt_objects", [])),
        "gt_captions": row.get("gt_captions", []),
    } for row in rows]
    per_sample = compute_per_sample(records, objects, inverse, build_double_word_dict())
    metrics = aggregate_from_arrays(per_sample, list(per_sample))
    metrics.update({
        "num_samples": len(records),
        "total_results": len(rows),
        "eval_prompt": args.prompt,
        "decoding_method": "halc_qwen3vl_adaptation",
        "halc_config": config_record,
        "halc_total_triggered": sum(row["halc_triggered_count"] for row in rows),
        "halc_total_changed_tokens": sum(row["halc_changed_token_count"] for row in rows),
        "metric_schema": "official_chair_caption_and_mention_v1",
    })
    (output_dir / "chair_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"HALC_EVAL=PASS rows={len(rows)} triggered={metrics['halc_total_triggered']} "
          f"changed={metrics['halc_total_changed_tokens']}", flush=True)


if __name__ == "__main__":
    main()
