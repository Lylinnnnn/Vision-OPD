#!/usr/bin/env python3
"""Audit inference and metric stages for training-free external baselines."""

import argparse
import json
from pathlib import Path

from summarize_main_tr05_inventory import (
    amber_result,
    chair_result,
    pope_result,
    vision_result,
)


MODELS = ("Qwen3VL-4B-Instruct", "Qwen3VL-8B-Instruct", "InternVL3-2B-hf")
METHODS = ("vcd", "opera", "halc")
DATASET = "train5000_test1000_original_sr1p0"


def config_model(path):
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return config.get("model_name") or Path(config.get("model_path", "")).name


def candidates(results_root, model, method):
    found = set()
    dataset_dir = results_root / "external_baselines/full" / model / DATASET
    if dataset_dir.is_dir():
        found.update(path for path in dataset_dir.iterdir()
                     if path.is_dir() and method in path.name.lower())
    full_root = results_root / "external_baselines/full"
    configs = list(full_root.rglob(f"{method}_config.json")) if full_root.is_dir() else []
    configs.extend(results_root.glob(f"chair_external_*/{method}_config.json"))
    for config in configs:
        if ".shards" in config.parts or "smoke" in config.parts:
            continue
        if config_model(config) == model:
            found.add(config.parent)
    return sorted(found)


def count_rows(path):
    if not path.is_file():
        return 0
    try:
        with path.open(encoding="utf-8") as handle:
            return sum(bool(line.strip()) for line in handle)
    except OSError:
        return 0


def stage(raw_count, expected, metric_valid, metric_exists=False):
    if raw_count == 0:
        return "指标在，原始推理缺失" if metric_exists else "未推理"
    if raw_count != expected:
        return f"推理不完整({raw_count}/{expected})"
    if metric_valid:
        return "完成"
    return "已推理，指标需核验/重算" if metric_exists else "已推理，未算指标"


def valid(reader, directory, *args):
    try:
        return reader(directory, *args) is not None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def statuses(directory):
    chair = stage(count_rows(directory / "eval_results.jsonl"), 1000,
                  valid(chair_result, directory),
                  (directory / "chair_metrics.json").is_file())
    pope_counts = [count_rows(directory / "pope" / split / "eval_results.jsonl")
                   for split in ("pope_adv", "pope_pop", "pope_random")]
    pope = stage(sum(pope_counts), 5223 * 3, valid(pope_result, directory),
                 (directory / "pope/pope_summary.json").is_file())
    amber = stage(count_rows(directory / "amber/raw_results.jsonl"), 15220,
                  valid(amber_result, directory),
                  (directory / "amber/amber_metrics.json").is_file())
    vision = []
    for bench, expected in (("mmstar", 1500), ("cv-bench", 2638)):
        answers = list((directory / bench / "model_answer").glob("*_answer.jsonl"))
        answers += list((directory / "vision_opd/model_answer" / bench).glob("*_answer.jsonl"))
        counts = [count_rows(path) for path in answers]
        judges = list((directory / bench / "judge").glob("*_answer.jsonl"))
        judges += list((directory / "vision_opd/judge" / bench).glob("*_answer.jsonl"))
        vision.append(stage(max(counts, default=0), expected,
                            valid(vision_result, directory, bench), bool(judges)))
    return (chair, pope, amber, *vision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("res-opd/eval_results"))
    args = parser.parse_args()
    print("VCD/OPERA/HALC are training-free; no training checkpoint is required.")
    print("CHAIR means official v2 on 1000 images with 'Describe the image.'.")
    print("Model | Method | CHAIR | POPE | AMBER | MMStar | CV-Bench | Evidence")
    for model in MODELS:
        for method in METHODS:
            directories = candidates(args.results_root, model, method)
            if not directories:
                print(f"{model} | {method.upper()} | 未推理 | 未推理 | 未推理 | 未推理 | 未推理 | --")
            for directory in directories:
                print(f"{model} | {method.upper()} | " + " | ".join(statuses(directory)) +
                      f" | {directory}")


if __name__ == "__main__":
    main()
