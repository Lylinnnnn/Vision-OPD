#!/usr/bin/env python3
"""Print HALVA CHAIR, POPE, and AMBER main-table metrics without recomputing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from eval_chair import extract_final_response_text  # noqa: E402


MODELS = {
    "qwen2b": ("Qwen3VL", "2B", "Qwen3VL-2B-Instruct"),
    "qwen4b": ("Qwen3VL", "4B", "Qwen3VL-4B-Instruct"),
    "qwen8b": ("Qwen3VL", "8B", "Qwen3VL-8B-Instruct"),
    "internvl2b": ("InternVL3", "2B", "InternVL3-2B-hf"),
}
POPE_SPLITS = ("pope_adv", "pope_pop", "pope_random")
ERROR_PREFIXES = ("[ERROR]", "[API_ERROR]", "[FUTURE_ERROR]")
HEADER = (
    "Family", "Size", "Method", "Step", "CHAIRi", "CHAIRs", "ObjF1",
    "Words/img", "Chars/img", "Correct mentions", "Hall mentions",
    "POPE F1", "A-CHAIR", "A-HAL",
)


def load_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_object(path: Path) -> dict:
    value = load_json(path)
    return value if isinstance(value, dict) else {}


def load_jsonl(path: Path) -> tuple[list[dict], int]:
    if not path.is_file():
        return [], 0
    rows = []
    invalid = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                invalid += 1
    return rows, invalid


def percentage(value: object) -> str:
    return f"{100 * value:.2f}" if isinstance(value, (int, float)) else "--"


def points(value: object) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "--"


def error_count(rows: list[dict]) -> int:
    return sum(str(row.get("model_answer") or "").startswith(ERROR_PREFIXES) for row in rows)


def summarize(directory: Path, family: str, size: str, run_name: str,
              chair_expected: int = 1000, pope_expected: int = 5223,
              amber_expected: int = 15220) -> tuple[list[str], str]:
    chair = load_object(directory / "chair_metrics.json")
    chair_rows, chair_invalid = load_jsonl(directory / "eval_results.jsonl")
    captions = [extract_final_response_text(row.get("generated_caption", ""))
                for row in chair_rows]
    chair_ok = (
        len(chair_rows) == chair_expected
        and chair_invalid == 0
        and len({row.get("image_id") for row in chair_rows}) == chair_expected
        and chair.get("num_samples") == chair_expected
        and str(chair.get("metric_schema", "")).startswith("official_chair")
    )

    pope = load_object(directory / "pope" / "pope_summary.json")
    split_rows = {split: load_jsonl(directory / "pope" / split / "eval_results.jsonl")
                  for split in POPE_SPLITS}
    pope_ok = all(
        len(rows) == pope_expected and invalid == 0
        and len({row.get("sample_uid") for row in rows}) == pope_expected
        and split in pope.get("benchmarks", {})
        for split, (rows, invalid) in split_rows.items()
    )

    amber = load_object(directory / "amber" / "amber_metrics.json")
    amber_rows, amber_invalid = load_jsonl(directory / "amber" / "raw_results.jsonl")
    responses = load_json(directory / "amber" / "amber_a_responses.json")
    amber_ok = (
        amber.get("official_returncode") == 0
        and len(amber_rows) == amber_expected and amber_invalid == 0
        and len({row.get("id") for row in amber_rows}) == amber_expected
        and isinstance(responses, list) and len(responses) == amber_expected
        and len({row.get("id") for row in responses if isinstance(row, dict)}) == amber_expected
    )
    amber_metrics = amber.get("metrics", {}) if amber_ok else {}
    if not isinstance(amber_metrics, dict):
        amber_metrics = {}

    words = sum(len(re.findall(r"[A-Za-z0-9]+", text)) for text in captions) / chair_expected if chair_ok else None
    chars = sum(len(text) for text in captions) / chair_expected if chair_ok else None
    row = [
        family, size, "HALVA", run_name.removeprefix("halva_coco5k_"),
        percentage(chair.get("CHAIRi")) if chair_ok else "--",
        percentage(chair.get("CHAIRs")) if chair_ok else "--",
        percentage(chair.get("ObjF1")) if chair_ok else "--",
        f"{words:.2f}" if words is not None else "--",
        f"{chars:.2f}" if chars is not None else "--",
        str(chair["total_correct_mentions"]) if chair_ok and "total_correct_mentions" in chair else "--",
        str(chair["total_hallucinated_mentions"]) if chair_ok and "total_hallucinated_mentions" in chair else "--",
        percentage(pope.get("macro_f1")) if pope_ok else "--",
        points(amber_metrics.get("generative_task_chair")),
        points(amber_metrics.get("generative_task_hal")),
    ]
    pope_errors = sum(error_count(rows) for rows, _ in split_rows.values())
    amber_errors = error_count(amber_rows)
    counts = ",".join(f"{split}={len(split_rows[split][0])}/{pope_expected}" for split in POPE_SPLITS)
    audit = (f"{directory.parent.parent.name}: CHAIR={len(chair_rows)}/{chair_expected} "
             f"POPE=[{counts}] AMBER={len(amber_rows)}/{amber_expected} "
             f"official_rc={amber.get('official_returncode', '--')} "
             f"POPE_errors={pope_errors} AMBER_errors={amber_errors} "
             f"invalid_jsonl={chair_invalid + amber_invalid + sum(n for _, n in split_rows.values())}")
    return row, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="all", choices=("all", *MODELS))
    parser.add_argument("--run-name", default="halva_coco5k_e1")
    parser.add_argument("--eval-root", type=Path,
                        default=ROOT / "eval_results/external_baselines/full")
    args = parser.parse_args()
    models = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    print(" | ".join(HEADER))
    print(" | ".join("---" for _ in HEADER))
    audits = []
    for family, size, name in models.values():
        directory = (args.eval_root / name / "train5000_test1000_original_sr1p0" /
                     f"{args.run_name}_official_v2")
        row, audit = summarize(directory, family, size, args.run_name)
        print(" | ".join(row))
        audits.append(audit)
    print("\nAudit (error-derived scores are provisional):")
    for audit in audits:
        print(audit)


if __name__ == "__main__":
    main()
