#!/usr/bin/env python3
"""Validate RLSF-V preferences and write LlamaFactory configs for COCO 5k runs."""

import argparse
import json
from pathlib import Path


TEMPLATES = {
    "qwen2b": ("qwen3.yaml", "qwen3_vl_nothink"),
    "qwen4b": ("qwen3.yaml", "qwen3_vl_nothink"),
    "qwen8b": ("qwen3.yaml", "qwen3_vl_nothink"),
    "internvl2b": ("intern3_5.yaml", "intern_vl"),
}


def load_preferences(path, questions):
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, list) or not source:
        raise ValueError(f"No DPO preference pairs in {path}")
    expected = {
        (row["image_path"], row["question"])
        for row in (json.loads(line) for line in questions.read_text(encoding="utf-8").splitlines() if line)
    }
    records = []
    for item in source:
        image = item["img_path"]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        if (image, prompt) not in expected:
            raise ValueError(f"Preference pair is not from the COCO 5k prompt file: {image}")
        if not all(isinstance(value, str) and value.strip() for value in (chosen, rejected)):
            raise ValueError(f"Empty chosen/rejected response for {image}")
        if chosen == rejected:
            raise ValueError(f"Identical chosen/rejected responses for {image}")
        if not Path(image).is_file():
            raise FileNotFoundError(image)
        records.append({
            "instruction": "<image>" + prompt,
            "input": "",
            "chosen": chosen,
            "rejected": rejected,
            "images": [image],
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "export"))
    parser.add_argument("--model", choices=TEMPLATES, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    args = parser.parse_args()

    import yaml

    template_file, template_name = TEMPLATES[args.model]
    lf_root = args.upstream / "train/hf_series/LlamaFactory"
    adapter_dir = args.run_dir / "adapter"
    merged_dir = args.run_dir / "merged"
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "train":
        dpo_path = args.run_dir / "generate" / f"{args.model}_dpo_data.json"
        records = load_preferences(dpo_path, args.questions)
        dataset_dir = args.run_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "preferences.json").write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8"
        )
        (dataset_dir / "dataset_info.json").write_text(
            json.dumps({"rlsf_v_coco5k": {
                "file_name": "preferences.json",
                "columns": {
                    "prompt": "instruction", "query": "input",
                    "chosen": "chosen", "rejected": "rejected", "images": "images",
                },
                "ranking": True,
            }}, indent=2), encoding="utf-8"
        )
        config = yaml.safe_load((lf_root / template_file).read_text(encoding="utf-8"))
        config.update({
            "model_name_or_path": str(args.model_path),
            "template": template_name,
            "dataset": "rlsf_v_coco5k",
            "dataset_dir": str(dataset_dir),
            "output_dir": str(adapter_dir),
            "save_strategy": "epoch",
            "save_only_model": False,
            "overwrite_output_dir": False,
        })
        destination = args.run_dir / "train.yaml"
        print(f"Validated {len(records)} matched COCO 5k preference pairs")
    else:
        if not (adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(adapter_dir / "adapter_config.json")
        config = {
            "model_name_or_path": str(args.model_path),
            "adapter_name_or_path": str(adapter_dir),
            "template": template_name,
            "trust_remote_code": True,
            "export_dir": str(merged_dir),
            "export_size": 5,
            "export_device": "cpu",
            "export_legacy_format": False,
        }
        destination = args.run_dir / "export.yaml"
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
