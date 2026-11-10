#!/usr/bin/env python3
"""Show merged and latest-shard progress for Qwen3VL OPERA/HALC evaluations."""

from __future__ import annotations

import argparse
import json
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


def answer_counts(path):
    if not path.is_file():
        return 0, 0
    total = errors = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            total += 1
            try:
                answer = str(json.loads(line).get("model_answer") or "").strip()
            except (json.JSONDecodeError, AttributeError):
                answer = ""
            errors += not answer or answer.startswith(
                ("[ERROR]", "[API_ERROR]", "[FUTURE_ERROR]"))
    return total, errors


def report_aux(directory, label):
    aux_run, aux_shards = latest_shards(directory, ".aux_shards")
    for name, relative, expected in (
        *((split, Path("pope") / split / "eval_results.jsonl", 5223)
          for split in POPE_SPLITS),
        ("AMBER", Path("amber/raw_results.jsonl"), 15220),
    ):
        merged, merged_errors = answer_counts(directory / relative)
        partial = [answer_counts(path / relative) for path in aux_shards]
        latest = sum(total for total, _ in partial)
        latest_errors = sum(errors for _, errors in partial)
        print(f"{label} {name} merged={merged}/{expected} merged_errors={merged_errors} "
              f"latest={latest}/{expected if aux_shards else 0} latest_errors={latest_errors} "
              f"run={aux_run.name if aux_run else '--'}")


def report(directory, method, size):
    label = f"{method.upper()} {size}"
    chair_final = count_lines(directory / "eval_results.jsonl")
    chair_run, chair_shards = latest_shards(directory, ".shards")
    chair_counts = counts(chair_shards, Path("eval_results.jsonl"))
    print(f"{label} CHAIR merged={chair_final}/1000 "
          f"latest={sum(chair_counts)}/{1000 if chair_shards else 0} "
          f"shards={chair_counts} run={chair_run.name if chair_run else '--'}")
    report_aux(directory, label)
    if method == "halc":
        for presentation in sorted(directory.glob("presentation_mixed_8192_*")):
            if not presentation.is_dir():
                continue
            manifest = presentation / "presentation_manifest.json"
            old_count = json.loads(manifest.read_text()).get("old_count", 0) if manifest.is_file() else 0
            new_files = sorted(presentation.glob(".new_shards/worker_*/out/eval_results.jsonl"))
            new_count = sum(count_lines(path) for path in new_files)
            merged = count_lines(presentation / "eval_results.jsonl")
            mixed_label = f"{label} {presentation.name}"
            print(f"{mixed_label} CHAIR merged={merged}/1000 "
                  f"continuation={new_count}/{1000 - old_count} original={old_count}/1000")
            report_aux(presentation, mixed_label)


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
