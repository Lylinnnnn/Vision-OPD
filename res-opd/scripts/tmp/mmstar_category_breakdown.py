#!/usr/bin/env python3
"""MMStar rule-only category/l2-category breakdown from existing outputs.

This script ignores existing `judge` fields completely. It re-scores each
saved model answer with deterministic option extraction, so weak LLM-as-judge
outputs cannot affect the result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET_TAG = "train5000_test1000_original_sr1p0"

DEFAULT_EXPERIMENTS = [
    (
        "base",
        "Qwen3VL-2B-Instruct",
    ),
    (
        "tr0.75_rkl",
        "Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-full5k-e1_global_step_39",
    ),
    (
        "tr0.75_sw",
        "Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-sw-full5k-e1_global_step_39",
    ),
    (
        "tr0.75_sw075",
        "Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-sw075-full5k-e1_global_step_39",
    ),
]

TR10_EXPERIMENT = (
    "tr1.0_rkl",
    "Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr1.0-a1.0-frozen-rkl-full5k-e1_global_step_39",
)


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass
    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON line {line_no} in {path}: {exc}") from exc
        if isinstance(item, dict):
            records.append(item)
    return records


def extract_answer_text(model_answer_raw: str) -> str:
    if not isinstance(model_answer_raw, str):
        return ""
    if "<answer>" in model_answer_raw:
        start = model_answer_raw.find("<answer>")
        end = model_answer_raw.find("</answer>")
        if start != -1 and end != -1:
            return model_answer_raw[start + len("<answer>") : end].strip()
    lower = model_answer_raw.lower()
    marker = "answer:"
    if marker in lower:
        idx = lower.find(marker)
        return model_answer_raw[idx + len(marker) :].strip()
    return model_answer_raw.strip()


def extract_gt_option(answer: Any) -> str:
    if not isinstance(answer, str) or not answer:
        return ""
    text = answer.strip().upper()
    match = re.match(r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+)|\s)", text)
    return match.group(1) if match else ""


def extract_pred_option_legacy(text: str) -> tuple[str, str]:
    """Match eval/judge_qwenlm.py's deterministic MCQ rule as closely as possible."""
    if not text:
        return "", "empty"
    match = re.search(r"\(([A-Z])\)", text)
    if match:
        return match.group(1), "paren"
    match = re.search(r"([A-Z])[\.\)\s]", text)
    if match:
        return match.group(1), "letter_delim"
    match = re.search(r"([A-Z])", text)
    if match:
        return match.group(1), "first_upper"
    return "", "unresolved"


def extract_pred_option_strict(text: str) -> tuple[str, str]:
    if not text:
        return "", "empty"
    patterns = [
        (r"^[\s\(\[]*([A-F])(?:[\.\)\]\:\-]|$|\s)", "leading_letter"),
        (r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-F])\)?\b", "answer_phrase"),
        (r"\(([A-F])\)", "paren"),
        (r"\b([A-F])\b(?:\s*[\.\)]\s*)?$", "final_letter"),
    ]
    for pattern, source in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper(), source
    return "", "unresolved"


def rule_score_item(item: dict[str, Any], extract_mode: str) -> dict[str, Any]:
    raw_answer = item.get("model_answer")
    if not isinstance(raw_answer, str) or not raw_answer.strip():
        raw_answer = item.get("extracted_answer", "")
    answer_text = extract_answer_text(raw_answer)
    gt = extract_gt_option(item.get("response", ""))
    if extract_mode == "strict":
        pred, source = extract_pred_option_strict(answer_text)
    else:
        pred, source = extract_pred_option_legacy(answer_text)
    return {
        "gt": gt,
        "pred": pred,
        "source": source,
        "correct": bool(gt and pred and gt == pred),
        "unresolved": not bool(pred),
    }


def annotate_rule_scores(records: list[dict[str, Any]], extract_mode: str) -> list[dict[str, Any]]:
    annotated = []
    for item in records:
        scored = dict(item)
        rule = rule_score_item(scored, extract_mode)
        scored["_rule_gt"] = rule["gt"]
        scored["_rule_pred"] = rule["pred"]
        scored["_rule_source"] = rule["source"]
        scored["_rule_correct"] = rule["correct"]
        scored["_rule_unresolved"] = rule["unresolved"]
        annotated.append(scored)
    return annotated


def is_correct(item: dict[str, Any]) -> bool:
    return bool(item.get("_rule_correct"))


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.2f}%"


def signed_pct(value: float | None) -> str:
    if value is None:
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{100.0 * value:.2f}"


def safe_group(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if value is None or str(value).strip() == "":
        return "unknown"
    return str(value).strip()


def sample_key(item: dict[str, Any]) -> str:
    for key in ("sample_uid", "index", "question_id", "id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return f"{key}:{value}"
    images = item.get("images") or []
    image0 = images[0] if isinstance(images, list) and images else ""
    return "fallback:" + json.dumps(
        {"image": image0, "query": item.get("query", "")},
        ensure_ascii=False,
        sort_keys=True,
    )


def resolve_result_file(result_root: Path, dataset_tag: str, value: str) -> Path:
    candidate = Path(os.path.expanduser(value))
    if candidate.is_file():
        return candidate

    paths_to_try = []
    if candidate.is_dir():
        paths_to_try.append(candidate)
    else:
        paths_to_try.append(result_root / value / dataset_tag)
        paths_to_try.append(result_root / value)

    result_candidates: list[Path] = []
    for base in paths_to_try:
        result_candidates.extend(base.glob("mmstar/judge/*_answer.jsonl"))
        result_candidates.extend(base.glob("mmstar/model_answer/*_answer.jsonl"))
        result_candidates.extend(base.glob("vision_opd/judge/mmstar/*_answer.jsonl"))
        result_candidates.extend(base.glob("vision_opd/model_answer/mmstar/*_answer.jsonl"))

    if not result_candidates:
        raise FileNotFoundError(
            f"No MMStar result file found for {value}. Tried: "
            + ", ".join(str(x) for x in paths_to_try)
        )

    # Prefer judge files because they contain the same model_answer plus category
    # fields, but the actual judge label is ignored by this script.
    result_candidates.sort(
        key=lambda p: (
            1 if "/judge/" in str(p) else 0,
            p.stat().st_mtime,
            str(p),
        ),
        reverse=True,
    )
    return result_candidates[0]


def parse_experiment_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Experiment spec must be label=path_or_exp_name, got: {spec}")
    label, value = spec.split("=", 1)
    label = label.strip()
    value = value.strip()
    if not label or not value:
        raise ValueError(f"Invalid experiment spec: {spec}")
    return label, value


def summarize_group(
    records_by_label: dict[str, list[dict[str, Any]]],
    base_label: str,
    group_key: str,
) -> list[dict[str, Any]]:
    base_records = records_by_label.get(base_label, [])
    base_by_key = {sample_key(item): item for item in base_records}
    base_acc_by_group: dict[str, float] = {}

    rows = []
    for label, records in records_by_label.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in records:
            grouped[safe_group(item, group_key)].append(item)

        if label == base_label:
            for group, items in grouped.items():
                total = len(items)
                correct = sum(is_correct(item) for item in items)
                base_acc_by_group[group] = correct / total if total else 0.0

        for group, items in sorted(grouped.items()):
            total = len(items)
            correct = sum(is_correct(item) for item in items)
            acc = correct / total if total else 0.0

            paired_n = improved = regressed = same_correct = same_wrong = 0
            if label != base_label and base_by_key:
                for item in items:
                    key = sample_key(item)
                    base_item = base_by_key.get(key)
                    if base_item is None:
                        continue
                    paired_n += 1
                    base_ok = is_correct(base_item)
                    curr_ok = is_correct(item)
                    if not base_ok and curr_ok:
                        improved += 1
                    elif base_ok and not curr_ok:
                        regressed += 1
                    elif base_ok and curr_ok:
                        same_correct += 1
                    else:
                        same_wrong += 1

            base_acc = base_acc_by_group.get(group)
            rows.append(
                {
                    "group_key": group_key,
                    "group": group,
                    "model": label,
                    "n": total,
                    "correct": correct,
                    "accuracy": acc,
                    "delta_vs_base": None if base_acc is None else acc - base_acc,
                    "paired_n": paired_n,
                    "improved_vs_base": improved,
                    "regressed_vs_base": regressed,
                    "same_correct": same_correct,
                    "same_wrong": same_wrong,
                    "paired_delta_vs_base": ((improved - regressed) / paired_n) if paired_n else None,
                    "unresolved": sum(bool(item.get("_rule_unresolved")) for item in items),
                }
            )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "group_key",
        "group",
        "model",
        "n",
        "correct",
        "accuracy",
        "delta_vs_base",
        "paired_n",
        "improved_vs_base",
        "regressed_vs_base",
        "same_correct",
        "same_wrong",
        "paired_delta_vs_base",
        "unresolved",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_wide_rows(rows: list[dict[str, Any]], labels: list[str], base_label: str) -> list[dict[str, Any]]:
    by_group: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_group[row["group"]][row["model"]] = row

    wide = []
    for group, by_model in by_group.items():
        base_row = by_model.get(base_label)
        out: dict[str, Any] = {
            "group": group,
            "n": base_row["n"] if base_row else max((r["n"] for r in by_model.values()), default=0),
        }
        for label in labels:
            item = by_model.get(label)
            out[f"{label}_acc"] = item.get("accuracy") if item else None
            out[f"{label}_delta"] = item.get("delta_vs_base") if item else None
            out[f"{label}_improved"] = item.get("improved_vs_base") if item else None
            out[f"{label}_regressed"] = item.get("regressed_vs_base") if item else None
        wide.append(out)
    wide.sort(key=lambda x: (-x["n"], x["group"]))
    return wide


def write_markdown(path: Path, title: str, rows: list[dict[str, Any]], labels: list[str], base_label: str) -> None:
    lines = [f"# {title}", ""]
    headers = ["Group", "N"]
    for label in labels:
        headers.append(f"{label} Acc")
        if label != base_label:
            headers.append(f"{label} Δ")
            headers.append(f"{label} +/-")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        cells = [row["group"], str(row["n"])]
        for label in labels:
            cells.append(pct(row.get(f"{label}_acc")))
            if label != base_label:
                cells.append(signed_pct(row.get(f"{label}_delta")))
                improved = row.get(f"{label}_improved")
                regressed = row.get(f"{label}_regressed")
                if improved is None or regressed is None:
                    cells.append("-")
                else:
                    cells.append(f"{improved}/{regressed}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("`Δ` is accuracy-point change vs base on the same group. `+/-` is paired improved/regressed sample count vs base.")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(records_by_label: dict[str, list[dict[str, Any]]], source_files: dict[str, str]) -> dict[str, Any]:
    summary = {"models": {}, "source_files": source_files}
    for label, records in records_by_label.items():
        total = len(records)
        correct = sum(is_correct(item) for item in records)
        rule_sources = Counter(str(item.get("_rule_source", "unknown")) for item in records)
        unresolved = sum(bool(item.get("_rule_unresolved")) for item in records)
        summary["models"][label] = {
            "n": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "unresolved": unresolved,
            "unresolved_rate": unresolved / total if total else 0.0,
            "rule_sources": dict(sorted(rule_sources.items())),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="res-opd/eval_results/latest/full", type=Path)
    parser.add_argument("--dataset-tag", default=DEFAULT_DATASET_TAG)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--base-label", default="base")
    parser.add_argument(
        "--extract-mode",
        choices=("legacy", "strict"),
        default="legacy",
        help="legacy matches eval/judge_qwenlm.py's deterministic MCQ rule; strict avoids broad first-uppercase fallback.",
    )
    parser.add_argument("--include-tr10", action="store_true", help="Also include tr1.0_rkl if its MMStar result exists.")
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Experiment as label=path_or_exp_name. If omitted, uses current base/tr0.75 defaults.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any requested experiment is missing.")
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    output_dir = args.output_dir or (result_root / f"mmstar_rule_breakdown_{args.extract_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment:
        experiment_specs = [parse_experiment_spec(spec) for spec in args.experiment]
    else:
        experiment_specs = list(DEFAULT_EXPERIMENTS)
        if args.include_tr10:
            experiment_specs.insert(1, TR10_EXPERIMENT)

    records_by_label: dict[str, list[dict[str, Any]]] = {}
    source_files: dict[str, str] = {}
    missing: list[str] = []
    for label, value in experiment_specs:
        try:
            result_file = resolve_result_file(result_root, args.dataset_tag, value)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue
        records = annotate_rule_scores(load_records(result_file), args.extract_mode)
        if not records:
            missing.append(f"Empty MMStar output for {label}: {result_file}")
            continue
        records_by_label[label] = records
        source_files[label] = str(result_file)

    if missing:
        print("Missing MMStar inputs:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        if args.strict:
            return 1

    if args.base_label not in records_by_label:
        print(f"Error: base label {args.base_label!r} is missing.", file=sys.stderr)
        return 1
    if len(records_by_label) < 2:
        print("Error: need at least base plus one variant.", file=sys.stderr)
        return 1

    labels = [label for label, _ in experiment_specs if label in records_by_label]
    summary = build_summary(records_by_label, source_files)

    all_rows: list[dict[str, Any]] = []
    for group_key in ("category", "l2_category"):
        rows = summarize_group(records_by_label, args.base_label, group_key)
        all_rows.extend(rows)
        wide = make_wide_rows(rows, labels, args.base_label)
        write_markdown(
            output_dir / f"mmstar_rule_{group_key}_breakdown.md",
            f"MMStar Rule-Only {group_key} Breakdown ({args.extract_mode})",
            wide,
            labels,
            args.base_label,
        )
        write_csv(output_dir / f"mmstar_rule_{group_key}_breakdown_long.csv", rows)

    summary["extract_mode"] = args.extract_mode
    summary["output_dir"] = str(output_dir)
    summary["models_compared"] = labels
    summary_path = output_dir / "mmstar_rule_breakdown_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "mmstar_rule_breakdown_all_long.csv", all_rows)

    print(f"Saved summary: {summary_path}")
    print(f"Saved category markdown: {output_dir / 'mmstar_rule_category_breakdown.md'}")
    print(f"Saved l2 markdown: {output_dir / 'mmstar_rule_l2_category_breakdown.md'}")
    for label in labels:
        model = summary["models"][label]
        print(
            f"{label}: {pct(model['accuracy'])} ({model['correct']}/{model['n']}), "
            f"unresolved={model['unresolved']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
