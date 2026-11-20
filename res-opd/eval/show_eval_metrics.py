#!/usr/bin/env python3
"""Display CHAIR, POPE, and AMBER metrics from arbitrary evaluation directories.

Pass any number of directories or eval_results.jsonl files. An optional
LABEL=PATH gives a concise name to each row. No results are modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from eval_chair import extract_final_response_text


POPE_SPLITS = ("pope_adv", "pope_pop", "pope_random")
ERROR_PREFIXES = ("[ERROR]", "[API_ERROR]", "[FUTURE_ERROR]")
HEADER = (
    "Result", "CHAIRi↓", "CHAIRs↓", "ObjF1↑", "Words/img", "Chars/img",
    "Correct mentions", "Hall mentions", "POPE F1↑", "A-CHAIR↓", "A-HAL↓",
)


def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def inspect_jsonl(path: Path, answer_key: str | None = None) -> tuple[int, int, int]:
    """Return row count, invalid lines, and error answers."""
    if not path.is_file():
        return 0, 0, 0
    count = invalid = errors = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if not isinstance(row, dict):
                invalid += 1
                continue
            count += 1
            if answer_key:
                answer = str(row.get(answer_key) or "")
                errors += answer.startswith(ERROR_PREFIXES)
    return count, invalid, errors


def inspect_chair(path: Path) -> tuple[int, int, int, int, int]:
    if not path.is_file():
        return 0, 0, 0, 0, 0
    count = invalid = errors = words = chars = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if not isinstance(row, dict):
                invalid += 1
                continue
            count += 1
            raw = str(row.get("generated_caption") or "")
            errors += raw.startswith(ERROR_PREFIXES)
            text = extract_final_response_text(raw)
            words += len(re.findall(r"[A-Za-z0-9]+", text))
            chars += len(text)
    return count, invalid, errors, words, chars


def percent(value: object) -> str:
    return f"{value * 100:.2f}" if isinstance(value, (int, float)) else "--"


def number(value: object) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "--"


def integer(value: object) -> str:
    return str(value) if isinstance(value, int) else "--"


def default_label(directory: Path) -> str:
    if directory.parent.name.startswith("train"):
        return f"{directory.parent.parent.name}/{directory.name}"
    return directory.name


def parse_result(value: str) -> tuple[str, Path]:
    if "=" in value and not value.startswith("/"):
        label, path = value.split("=", 1)
        if not label or not path:
            raise ValueError(f"Expected LABEL=PATH: {value}")
    else:
        label, path = "", value
    directory = Path(path).expanduser()
    if directory.is_file() and directory.name == "eval_results.jsonl":
        directory = directory.parent
    if not directory.is_dir():
        raise ValueError(f"Evaluation directory does not exist: {directory}")
    return label or default_label(directory), directory


def summarize(label: str, directory: Path) -> tuple[list[str], str]:
    chair = read_object(directory / "chair_metrics.json")
    chair_n, chair_invalid, chair_errors, words, chars = inspect_chair(directory / "eval_results.jsonl")
    chair_scored = bool(chair)

    pope = read_object(directory / "pope" / "pope_summary.json")
    split_counts = {}
    pope_errors = pope_invalid = 0
    for split in POPE_SPLITS:
        count, invalid, errors = inspect_jsonl(
            directory / "pope" / split / "eval_results.jsonl", "model_answer")
        split_counts[split] = count
        pope_invalid += invalid
        pope_errors += errors

    amber = read_object(directory / "amber" / "amber_metrics.json")
    amber_n, amber_invalid, amber_errors = inspect_jsonl(
        directory / "amber" / "raw_results.jsonl", "model_answer")
    amber_scores = amber.get("metrics") if amber.get("official_returncode") == 0 else None
    if not isinstance(amber_scores, dict):
        amber_scores = {}

    row = [
        label,
        percent(chair.get("CHAIRi")) if chair_scored else "--",
        percent(chair.get("CHAIRs")) if chair_scored else "--",
        percent(chair.get("ObjF1")) if chair_scored else "--",
        number(words / chair_n) if chair_n and not chair_invalid else "--",
        number(chars / chair_n) if chair_n and not chair_invalid else "--",
        integer(chair.get("total_correct_mentions")) if chair_scored else "--",
        integer(chair.get("total_hallucinated_mentions")) if chair_scored else "--",
        percent(pope.get("macro_f1")),
        number(amber_scores.get("generative_task_chair")),
        number(amber_scores.get("generative_task_hal")),
    ]
    split_text = ",".join(f"{split}={split_counts[split]}" for split in POPE_SPLITS)
    audit = (
        f"{label}: CHAIR rows={chair_n} scored={chair.get('num_samples', '--')} "
        f"schema={chair.get('metric_schema', '--')} errors={chair_errors}; "
        f"POPE rows=[{split_text}] errors={pope_errors}; "
        f"AMBER rows={amber_n} rc={amber.get('official_returncode', '--')} "
        f"errors={amber_errors}; invalid_jsonl={chair_invalid + pope_invalid + amber_invalid}; "
        f"dir={directory}"
    )
    return row, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", metavar="[LABEL=]PATH",
                        help="Evaluation directories or CHAIR eval_results.jsonl files")
    args = parser.parse_args()
    try:
        inputs = [parse_result(value) for value in args.results]
    except ValueError as exc:
        parser.error(str(exc))
    print(" | ".join(HEADER))
    print(" | ".join("---" for _ in HEADER))
    audits = []
    for label, directory in inputs:
        row, audit = summarize(label, directory)
        print(" | ".join(row))
        audits.append(audit)
    print("\nAudit (error-derived or partial scores require review):")
    for audit in audits:
        print(audit)


if __name__ == "__main__":
    main()
