#!/usr/bin/env python3
"""Show merged and latest-shard progress for Qwen3VL OPERA/HALC evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path


SUFFIXES = {
    "opera": "chair_external_opera_official",
    "halc": "chair_external_halc_qwen3vl_adapted_official",
}
DATASET = "train5000_test1000_original_sr1p0"
POPE_SPLITS = ("pope_adv", "pope_pop", "pope_random")


def count_lines(path):
    if not path.is_file():
        return 0
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def latest_shards(directory, parent):
    root = directory / parent
    if not root.is_dir():
        return None, []
    runs = [path for path in root.iterdir()
            if path.is_dir() and list(path.glob("shard_*/out"))]
    if not runs:
        return None, []
    run = max(runs, key=lambda path: path.stat().st_mtime)
    return run, sorted(run.glob("shard_*/out"))


def counts(paths, relative):
    return [count_lines(path / relative) for path in paths]


def report(directory, method, size):
    label = f"{method.upper()} {size}"
    chair_final = count_lines(directory / "eval_results.jsonl")
    chair_run, chair_shards = latest_shards(directory, ".shards")
    chair_counts = counts(chair_shards, Path("eval_results.jsonl"))
    print(f"{label} CHAIR merged={chair_final}/1000 "
          f"latest={sum(chair_counts)}/{1000 if chair_shards else 0} "
          f"shards={chair_counts} run={chair_run.name if chair_run else '--'}")
    aux_run, aux_shards = latest_shards(directory, ".aux_shards")
    for split in POPE_SPLITS:
        relative = Path("pope") / split / "eval_results.jsonl"
        merged = count_lines(directory / relative)
        partial = counts(aux_shards, relative)
        print(f"{label} {split} merged={merged}/5223 latest={sum(partial)}/{5223 if aux_shards else 0} "
              f"run={aux_run.name if aux_run else '--'}")
    relative = Path("amber/raw_results.jsonl")
    merged = count_lines(directory / relative)
    partial = counts(aux_shards, relative)
    print(f"{label} AMBER merged={merged}/15220 latest={sum(partial)}/{15220 if aux_shards else 0} "
          f"run={aux_run.name if aux_run else '--'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "eval_results")
    args = parser.parse_args()
    for size in ("4B", "8B"):
        for method, suffix in SUFFIXES.items():
            directory = (args.eval_root / "external_baselines/full" /
                         f"Qwen3VL-{size}-Instruct" / DATASET / suffix)
            report(directory, method, size)


if __name__ == "__main__":
    main()
