#!/usr/bin/env python3
"""Convert the Res-OPD COCO train split to RLSF-V's question JSONL."""

import argparse
import json
from pathlib import Path


def prompt_text(value):
    messages = list(value)
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("Expected one user prompt per training row")
    content = messages[0]["content"]
    if not isinstance(content, str) or not content.startswith("<image>"):
        raise ValueError("Expected a text prompt beginning with <image>")
    return content.removeprefix("<image>").strip()


def convert_frame(frame, test_ids, data_root, check_images=True, max_samples=0):
    records = []
    seen = set()
    for _, row in frame.iterrows():
        images = list(row["images"])
        if len(images) != 1:
            raise ValueError("Expected exactly one image per training row")
        source_path = Path(images[0]["path"])
        image_id = int(row["extra_info"]["image_id"])
        if image_id in seen or image_id in test_ids:
            raise ValueError(f"Duplicate or test-set image_id: {image_id}")
        seen.add(image_id)
        if source_path.name != f"{image_id:012d}.jpg":
            raise ValueError(f"Unexpected image filename for {image_id}: {source_path.name}")
        image_path = data_root / "COCO" / "train2017" / source_path.name
        if check_images and not image_path.is_file():
            raise FileNotFoundError(image_path)
        records.append({
            "image_id": str(image_id),
            "image_path": str(image_path),
            "question": prompt_text(row["prompt"]),
        })
        if max_samples and len(records) >= max_samples:
            break
    if not records:
        raise ValueError("No training rows were converted")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()

    import pandas as pd

    test_ids = {int(row["image_id"]) for row in json.loads(args.test_json.read_text())}
    frame = pd.read_parquet(args.train_parquet)
    records = convert_frame(
        frame, test_ids, args.data_root,
        check_images=not args.skip_image_check,
        max_samples=args.max_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"RLSF-V questions: {len(records)} -> {args.output}")
    print(f"Prompt: {records[0]['question']}")


if __name__ == "__main__":
    main()
