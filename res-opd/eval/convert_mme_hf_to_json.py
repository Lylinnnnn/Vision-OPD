#!/usr/bin/env python3
"""Convert lmms-lab/MME parquet data to the JSON format used by eval_mme_classic."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re

from PIL import Image


def safe_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:120] or "sample"


def image_bytes_from_obj(image_obj) -> bytes:
    if isinstance(image_obj, (bytes, bytearray)):
        return bytes(image_obj)
    if isinstance(image_obj, dict):
        raw = image_obj.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
        path = image_obj.get("path")
        if path:
            with open(path, "rb") as f:
                return f.read()
    if isinstance(image_obj, Image.Image):
        buf = io.BytesIO()
        image_obj.convert("RGB").save(buf, format="JPEG")
        return buf.getvalue()
    if isinstance(image_obj, str) and image_obj:
        with open(image_obj, "rb") as f:
            return f.read()
    raise ValueError(f"Unsupported image object: {type(image_obj)}")


def load_hf_rows(source: str, split: str) -> list[dict]:
    try:
        from datasets import load_dataset

        return list(load_dataset(source, split=split))
    except Exception as datasets_exc:
        source_path = Path(source)
        parquet_files = sorted(source_path.rglob("*.parquet")) if source_path.exists() else []
        if not parquet_files:
            raise RuntimeError(
                f"Could not load lmms-lab/MME data from {source}. "
                f"datasets error: {datasets_exc}"
            ) from datasets_exc

        import pyarrow.parquet as pq

        rows = []
        for parquet_file in parquet_files:
            table = pq.read_table(str(parquet_file))
            rows.extend(table.to_pylist())
        return rows


def main():
    parser = argparse.ArgumentParser(description="Convert lmms-lab/MME parquet to Res-OPD MME JSON")
    parser.add_argument("--source", required=True, help="Local HF dataset directory or dataset id such as lmms-lab/MME")
    parser.add_argument("--split", default="test")
    parser.add_argument("--categories", default="existence,count,position,color")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    args = parser.parse_args()

    categories = {x.strip() for x in args.categories.split(",") if x.strip()}
    rows = load_hf_rows(args.source, args.split)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    for idx, row in enumerate(rows):
        category = str(row.get("category", "")).strip()
        if categories and category not in categories:
            continue
        qid = row.get("question_id", idx)
        image_payload = image_bytes_from_obj(row.get("image"))
        image_hash = hashlib.sha1(image_payload).hexdigest()
        img_path = args.image_dir / f"{category}_{safe_name(qid)}_{image_hash[:12]}.jpg"
        if not img_path.exists():
            with open(img_path, "wb") as f:
                f.write(image_payload)
        converted.append(
            {
                "image": str(img_path),
                "image_id": image_hash,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "category": category,
                "question_id": qid,
            }
        )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"Converted {len(converted)} MME samples to {args.output_json}")


if __name__ == "__main__":
    main()
