#!/usr/bin/env python3
"""Fail before model startup when Parquet image paths are unavailable."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def image_paths(value):
    if value is None:
        return
    if isinstance(value, (str, Path)):
        if value:
            yield str(value)
        return
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return
        for key in ("image", "path"):
            if value.get(key) is not None:
                yield from image_paths(value[key])
                return
    elif isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        for item in value:
            yield from image_paths(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_parquet", type=Path)
    args = parser.parse_args()

    import pandas as pd

    from res_opd_dataset import ResOPDDataset

    if not args.train_parquet.is_file():
        raise SystemExit(f"Training Parquet not found: {args.train_parquet}")
    data = pd.read_parquet(args.train_parquet)
    resolver = ResOPDDataset.__new__(ResOPDDataset)
    resolver._path_remap_logged = set()
    seen = set()
    missing = []
    for column in ("images", "hires_images"):
        if column not in data:
            continue
        for value in data[column]:
            for path in image_paths(value):
                if path in seen:
                    continue
                seen.add(path)
                try:
                    resolver._resolve_image_path(path)
                except FileNotFoundError:
                    missing.append(path)
    print(f"Training image preflight: rows={len(data)}, unique_paths={len(seen)}, missing={len(missing)}")
    if missing:
        for path in missing[:5]:
            print(f"  MISSING: {path}", file=sys.stderr)
        print(
            "Set RES_OPD_DATA_ROOT to a directory containing COCO/train2017 images. "
            "A data.tar.gz archive is not a directly readable image directory.",
            file=sys.stderr,
        )
        return 1
    if not seen:
        print("Training Parquet has no image paths to verify", file=sys.stderr)
        return 1
    print("TRAINING_IMAGE_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
