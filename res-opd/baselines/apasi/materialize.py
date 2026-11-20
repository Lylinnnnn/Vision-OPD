#!/usr/bin/env python3
"""Validate APASI self-injection pairs and write LlamaFactory DPO/export configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TEMPLATES = {
    "qwen2b": ("qwen3.yaml", "qwen3_vl_nothink"),
    "qwen4b": ("qwen3.yaml", "qwen3_vl_nothink"),
    "qwen8b": ("qwen3.yaml", "qwen3_vl_nothink"),
    "internvl2b": ("intern3_5.yaml", "intern_vl"),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def collect_pairs(questions_path: Path, plans_path: Path, injected_dir: Path) -> list[dict]:
    question_rows = read_jsonl(questions_path)
    plan_rows = read_jsonl(plans_path)
    questions = {str(row["image_id"]): row for row in question_rows}
    plans = {str(row["image_id"]): row for row in plan_rows}
    if len(questions) == 0 or len(plans) == 0:
        raise ValueError("Empty questions or injection plans")
    if len(questions) != len(question_rows) or len(plans) != len(plan_rows):
        raise ValueError("Duplicate image IDs in questions or injection plans")
    if not set(plans) <= set(questions):
        raise ValueError("Injection plans contain images outside the training split")
    injected = [row for path in sorted(injected_dir.glob("shard_*.jsonl"))
                for row in read_jsonl(path)]
    ids = [str(row["image_id"]) for row in injected]
    if len(ids) != len(set(ids)) or set(ids) != set(plans):
        raise ValueError(f"Incomplete or duplicate self-injections: {len(ids)}/{len(plans)}")
    records = []
    for row in injected:
        image_id = str(row["image_id"])
        question = questions[image_id]
        plan = plans[image_id]
        if (row["image_path"] != question["image_path"] or
                row["prompt"] != question["question"] or
                row["chosen"] != plan["chosen"] or
                not row["chosen"].strip() or not row["rejected"].strip() or
                row["chosen"] == row["rejected"]):
            raise ValueError(f"Invalid APASI pair for image_id={image_id}")
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
        records.append({
            "instruction": "<image>" + row["prompt"], "input": "",
            "chosen": row["chosen"], "rejected": row["rejected"],
            "images": [row["image_path"]],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "export"))
    parser.add_argument("--model", choices=TEMPLATES, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lf-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    if args.epochs <= 0 or args.cutoff_len < 2048 or args.max_steps < 0:
        parser.error("--epochs must be positive, --cutoff-len >= 2048, and --max-steps nonnegative")
    import yaml

    template_file, template_name = TEMPLATES[args.model]
    adapter = args.work_dir / "adapter"
    merged = args.work_dir / "merged"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "train":
        records = collect_pairs(args.questions, args.work_dir / "plans.jsonl",
                                args.work_dir / "injected")
        dataset_dir = args.work_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "preferences.json").write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8")
        (dataset_dir / "dataset_info.json").write_text(json.dumps({
            "apasi_coco5k": {
                "file_name": "preferences.json",
                "columns": {"prompt": "instruction", "query": "input",
                            "chosen": "chosen", "rejected": "rejected", "images": "images"},
                "ranking": True,
            }
        }, indent=2), encoding="utf-8")
        config = yaml.safe_load((args.lf_root / template_file).read_text(encoding="utf-8"))
        config.update({
            "model_name_or_path": str(args.model_path), "template": template_name,
            "dataset": "apasi_coco5k", "dataset_dir": str(dataset_dir),
            "output_dir": str(adapter), "learning_rate": 4e-7,
            "pref_beta": 0.1, "lora_rank": 128, "lora_alpha": 256,
            "num_train_epochs": args.epochs, "cutoff_len": args.cutoff_len,
            "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
            "save_strategy": "epoch", "save_total_limit": 1,
            "save_only_model": True, "overwrite_output_dir": False,
        })
        if args.max_steps:
            config["max_steps"] = args.max_steps
        destination = args.work_dir / "train.yaml"
        print(f"APASI_PAIRS=PASS count={len(records)} questions={len(read_jsonl(args.questions))}")
    else:
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(adapter / "adapter_config.json")
        config = {
            "model_name_or_path": str(args.model_path), "adapter_name_or_path": str(adapter),
            "template": template_name, "trust_remote_code": True,
            "export_dir": str(merged), "export_size": 5,
            "export_device": "cpu", "export_legacy_format": False,
        }
        destination = args.work_dir / "export.yaml"
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"APASI_CONFIG=PASS path={destination}")


if __name__ == "__main__":
    main()
