#!/usr/bin/env python3
"""Generate APASI preferred captions or blind, self-injected rejected captions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


MODELS = ("qwen2b", "qwen4b", "qwen8b", "internvl2b")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def completed_ids(path: Path, expected: set[str]) -> set[str]:
    if not path.exists():
        return set()
    rows = []
    with path.open("rb+") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                rows.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                if line.endswith(b"\n") or stream.read(1):
                    raise ValueError(f"Corrupt JSONL row in {path} at byte {position}")
                stream.truncate(position)
                print(f"APASI_RESUME=TRUNCATED incomplete final row in {path}", flush=True)
                break
    ids = [str(row["image_id"]) for row in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= expected:
        raise ValueError(f"Duplicate/unexpected IDs in {path}")
    return set(ids)


def model_class(family: str):
    if family == "internvl2b":
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    from transformers import Qwen3VLForConditionalGeneration
    return Qwen3VLForConditionalGeneration


def load_model(path: Path, family: str):
    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True,
                                               local_files_only=True)
    model = model_class(family).from_pretrained(
        str(path), torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, attn_implementation="sdpa",
    ).to("cuda:0").eval()
    return model, processor


def decode(model, tokenizer, inputs: dict, max_new_tokens: int) -> str:
    import torch

    inputs = {key: value.to(model.device) if hasattr(value, "to") else value
              for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False,
                                max_new_tokens=max_new_tokens, repetition_penalty=1.1)
    tokens = output[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


def preferred_answer(model, processor, row: dict, max_new_tokens: int) -> str:
    from PIL import Image

    with Image.open(row["image_path"]) as picture:
        image = picture.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": row["question"]},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    return decode(model, processor.tokenizer, inputs, max_new_tokens)


def blind_completion(model, processor, prompt: str, context: str,
                     max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True,
                                         tokenize=False)
    inputs = processor.tokenizer(chat + context, return_tensors="pt")
    answer = decode(model, processor.tokenizer, inputs, max_new_tokens)
    first = re.split(r"(?<!\d)\.(?!\d)", answer, maxsplit=1)[0].strip()
    return first.rstrip(".!? ")


def inject(model, processor, plan: dict, max_new_tokens: int) -> str:
    sentences = list(plan["sentences"])
    for index_text, guide in sorted(plan["guides"].items(), key=lambda item: int(item[0])):
        index = int(index_text)
        prefix = " ".join(sentences[:index] + [guide["text"]]) + " "
        completion = blind_completion(model, processor, plan["prompt"],
                                      prefix, max_new_tokens)
        if not completion:
            raise RuntimeError(f"Empty blind completion for image_id={plan['image_id']}")
        sentences[index] = f"{guide['text']} {completion}."
    rejected = " ".join(sentences)
    if rejected == plan["chosen"]:
        raise RuntimeError(f"Self-injection did not change image_id={plan['image_id']}")
    return rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preferred", "inject"))
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard index/count")
    config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    expected_type = "internvl" if args.model == "internvl2b" else "qwen3_vl"
    if config.get("model_type") != expected_type:
        raise ValueError(f"Expected {expected_type}, got {config.get('model_type')!r}")
    rows = read_jsonl(args.input)
    selected = rows[args.shard_index::args.shard_count]
    expected = {str(row["image_id"]) for row in selected}
    if len(expected) != len(selected):
        raise ValueError("Duplicate image IDs in input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(args.output, expected)
    pending = [row for row in selected if str(row["image_id"]) not in done]
    if not pending:
        print(f"APASI_{args.phase.upper()}=PASS shard={args.shard_index} existing={len(done)} pending=0")
        return
    model, processor = load_model(args.model_path, args.model)
    if args.phase == "preferred":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
        from eval_chair import extract_final_response_text
    with args.output.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(pending, 1):
            if args.phase == "preferred":
                caption = extract_final_response_text(
                    preferred_answer(model, processor, row, args.max_new_tokens)
                )
                if not caption:
                    raise RuntimeError(f"Empty caption for image_id={row['image_id']}")
                result = {"image_id": row["image_id"], "image_path": row["image_path"],
                          "prompt": row["question"], "caption": caption}
            else:
                result = {"image_id": row["image_id"], "image_path": row["image_path"],
                          "prompt": row["prompt"], "chosen": row["chosen"],
                          "rejected": inject(model, processor, row, args.max_new_tokens)}
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 25 == 0 or index == len(pending):
                print(f"APASI_{args.phase.upper()} shard={args.shard_index} "
                      f"rows={len(done) + index}/{len(selected)}", flush=True)
    print(f"APASI_{args.phase.upper()}=PASS shard={args.shard_index} rows={len(selected)}")


if __name__ == "__main__":
    main()
