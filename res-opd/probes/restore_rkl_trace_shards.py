#!/usr/bin/env python3
"""Restore resumable shard files from a previously merged forced-scoring trace."""

import argparse
import json
from pathlib import Path


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-results", type=Path, required=True)
    parser.add_argument("--merged-trace", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--student-ratio", type=float, required=True)
    parser.add_argument("--teacher-ratio", type=float, required=True)
    args = parser.parse_args()
    if not args.merged_trace.is_file():
        return
    shard_paths = [args.shard_dir / f"shard_{idx}.jsonl" for idx in range(args.num_shards)]
    if any(path.exists() for path in shard_paths):
        return
    source = list(rows(args.eval_results))
    indices = {row["image_id"]: idx for idx, row in enumerate(source)}
    if len(indices) != len(source):
        raise ValueError("Duplicate image IDs in evaluation source")
    trace = list(rows(args.merged_trace))
    allocated = [[] for _ in shard_paths]
    seen = set()
    for row in trace:
        image_id = row.get("metadata", {}).get("image_id")
        if image_id not in indices or image_id in seen:
            raise ValueError(f"Unknown or duplicate image ID in merged trace: {image_id}")
        if (row.get("pair_mode") != "dual_view_same_model"
                or float(row.get("student_ratio")) != args.student_ratio
                or float(row.get("teacher_ratio")) != args.teacher_ratio
                or row.get("response_text") != source[indices[image_id]].get("generated_caption")):
            raise ValueError(f"Merged trace/source mismatch: image_id={image_id}")
        seen.add(image_id)
        allocated[indices[image_id] % args.num_shards].append(row)
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    for path, part in zip(shard_paths, allocated):
        with path.open("w") as stream:
            for row in part:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"RESTORED {len(trace)}/{len(source)} scored rows into {args.num_shards} shards")


if __name__ == "__main__":
    main()
