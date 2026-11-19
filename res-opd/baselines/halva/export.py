#!/usr/bin/env python3
"""Merge a HALVA LoRA adapter into a Hugging Face VLM for existing vLLM eval."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen2b", "qwen4b", "qwen8b", "internvl2b"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (args.adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(args.adapter / "adapter_config.json")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    from peft import PeftModel
    from transformers import AutoProcessor
    from train import load_model

    base = load_model(args.model_path, args.model)
    adapted = PeftModel.from_pretrained(base, str(args.adapter))
    merged = adapted.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(str(args.output_dir), safe_serialization=True, max_shard_size="5GB")
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True, local_files_only=True)
    processor.save_pretrained(str(args.output_dir))
    manifest = json.loads((args.adapter / "halva_run.json").read_text(encoding="utf-8"))
    (args.output_dir / "halva_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"HALVA_EXPORT=PASS output={args.output_dir}")


if __name__ == "__main__":
    main()
