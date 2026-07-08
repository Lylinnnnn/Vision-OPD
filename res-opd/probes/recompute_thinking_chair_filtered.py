#!/usr/bin/env python3
"""Recompute CHAIR for thinking eval results with closed-final filtering.

This probe never overwrites original eval outputs. It writes:
  - strict_closed_summary.json / .md
  - per-model filtered eval_results.jsonl
  - paired base-vs-model intersection metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROBE_DIR = Path(__file__).resolve().parent
RES_OPD_ROOT = PROBE_DIR.parent
EVAL_DIR = RES_OPD_ROOT / "eval"
sys.path.insert(0, str(EVAL_DIR))

from eval_chair import extract_final_response_text  # noqa: E402
from robust_chair_analysis import (  # noqa: E402
    aggregate_from_arrays,
    build_double_word_dict,
    compute_per_sample,
    parse_official_synonyms,
)


THINK_CLOSE = "</think>"
METRIC_KEYS = ("CHAIRi", "CHAIRs", "ObjPrec", "ObjRecall", "ObjF1", "RepRate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="LABEL=PATH entries. PATH may be eval_results.jsonl or a directory containing it.",
    )
    parser.add_argument("--base-label", default="baseline")
    parser.add_argument(
        "--output-dir",
        default=str(PROBE_DIR / "results" / "thinking_chair_filtered"),
    )
    parser.add_argument(
        "--allow-no-think",
        action="store_true",
        help="Treat rows without thinking tags as valid final captions. Keep false for thinking models.",
    )
    return parser.parse_args()


def resolve_eval_results(path_text: str) -> Path:
    path = Path(path_text)
    candidates = [path] if path.is_file() else [path / "eval_results.jsonl"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing eval_results.jsonl for {path_text}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_inputs(entries: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected LABEL=PATH, got: {entry}")
        label, path_text = entry.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in input: {entry}")
        out[label] = resolve_eval_results(path_text)
    return out


def row_validity(row: dict[str, Any], allow_no_think: bool) -> tuple[bool, str]:
    caption = str(row.get("generated_caption", "") or "")
    raw = str(row.get("raw_generated_caption", "") or "")
    if not caption or caption.startswith("[ERROR]"):
        return False, "empty_or_error_caption"
    if row.get("final_answer_available") is False or row.get("thinking_status") == "unclosed":
        return False, "unclosed_thinking"
    if THINK_CLOSE in raw or THINK_CLOSE in caption or row.get("thinking_status") == "closed":
        return True, "closed"
    if allow_no_think:
        return True, "no_think_allowed"
    return False, "missing_closed_think"


def score_rows(rows: list[dict[str, Any]], scorer_cache: dict[str, Any]) -> dict[str, Any]:
    if "objects" not in scorer_cache:
        mscoco_objects, inverse_synonym_dict = parse_official_synonyms()
        scorer_cache["objects"] = mscoco_objects
        scorer_cache["inverse"] = inverse_synonym_dict
        scorer_cache["double"] = build_double_word_dict()
    eval_records = [
        {
            "image_id": row["image_id"],
            "generated_text": extract_final_response_text(str(row.get("generated_caption", ""))),
            "gt_objects": set(row.get("gt_objects", [])),
        }
        for row in rows
    ]
    sample_dicts = compute_per_sample(
        eval_records,
        scorer_cache["objects"],
        scorer_cache["inverse"],
        scorer_cache["double"],
    )
    metrics = aggregate_from_arrays(sample_dicts, list(sample_dicts.keys()))
    metrics["num_samples"] = len(rows)
    return metrics


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    input_paths = parse_inputs(args.inputs)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scorer_cache: dict[str, Any] = {}
    per_model: dict[str, Any] = {}
    valid_rows_by_label: dict[str, dict[int, dict[str, Any]]] = {}

    for label, path in input_paths.items():
        rows = read_jsonl(path)
        valid_rows = []
        dropped = []
        reason_counts: dict[str, int] = {}
        for row in rows:
            ok, reason = row_validity(row, args.allow_no_think)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if ok:
                valid_rows.append(row)
            else:
                dropped.append(
                    {
                        "image_id": row.get("image_id"),
                        "reason": reason,
                        "generated_caption_prefix": str(row.get("generated_caption", ""))[:300],
                        "raw_generated_caption_prefix": str(row.get("raw_generated_caption", ""))[:300],
                    }
                )
        metrics = score_rows(valid_rows, scorer_cache) if valid_rows else {"num_samples": 0}
        model_dir = out_dir / label
        write_jsonl(model_dir / "filtered_eval_results.jsonl", valid_rows)
        write_jsonl(model_dir / "dropped_rows.jsonl", dropped)
        write_json(model_dir / "chair_metrics.json", metrics)
        valid_rows_by_label[label] = {int(row["image_id"]): row for row in valid_rows}
        per_model[label] = {
            "source_path": str(path),
            "raw_n": len(rows),
            "valid_n": len(valid_rows),
            "dropped_n": len(dropped),
            "valid_rate": len(valid_rows) / len(rows) if rows else 0.0,
            "reason_counts": reason_counts,
            "metrics": metrics,
        }

    paired: dict[str, Any] = {}
    if args.base_label in valid_rows_by_label:
        base_rows_by_id = valid_rows_by_label[args.base_label]
        for label, rows_by_id in valid_rows_by_label.items():
            if label == args.base_label:
                continue
            common_ids = sorted(set(base_rows_by_id) & set(rows_by_id))
            base_subset = [base_rows_by_id[i] for i in common_ids]
            other_subset = [rows_by_id[i] for i in common_ids]
            base_metrics = score_rows(base_subset, scorer_cache) if base_subset else {"num_samples": 0}
            other_metrics = score_rows(other_subset, scorer_cache) if other_subset else {"num_samples": 0}
            delta = {
                key: other_metrics.get(key, 0.0) - base_metrics.get(key, 0.0)
                for key in METRIC_KEYS
                if key in base_metrics and key in other_metrics
            }
            paired[label] = {
                "base_label": args.base_label,
                "paired_valid_n": len(common_ids),
                "base_metrics": base_metrics,
                "other_metrics": other_metrics,
                "delta_other_minus_base": delta,
            }

    summary = {
        "inputs": {label: str(path) for label, path in input_paths.items()},
        "base_label": args.base_label,
        "allow_no_think": args.allow_no_think,
        "per_model": per_model,
        "paired": paired,
    }
    write_json(out_dir / "strict_closed_summary.json", summary)

    lines = ["# Thinking CHAIR Strict-Closed Summary", ""]
    header = ["Model", "Raw N", "Valid N", "Valid Rate", *METRIC_KEYS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for label, item in per_model.items():
        metrics = item["metrics"]
        row = [
            label,
            str(item["raw_n"]),
            str(item["valid_n"]),
            fmt(item["valid_rate"]),
            *[fmt(metrics.get(key, "")) for key in METRIC_KEYS],
        ]
        lines.append("| " + " | ".join(row) + " |")

    if paired:
        lines.extend(["", "## Paired Intersection Delta", ""])
        header = ["Model", "Paired N", *[f"Delta {key}" for key in METRIC_KEYS]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for label, item in paired.items():
            delta = item["delta_other_minus_base"]
            row = [
                label,
                str(item["paired_valid_n"]),
                *[fmt(delta.get(key, "")) for key in METRIC_KEYS],
            ]
            lines.append("| " + " | ".join(row) + " |")

    (out_dir / "strict_closed_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_dir / 'strict_closed_summary.md'}")
    print(f"Wrote: {out_dir / 'strict_closed_summary.json'}")


if __name__ == "__main__":
    main()
