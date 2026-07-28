#!/usr/bin/env python3
"""Recompute official-definition CHAIR metrics from existing eval JSONL files.

This script performs no model loading or generation.  It recursively discovers
CHAIR ``eval_results.jsonl`` files, scores their existing captions, and writes a
new metrics file beside each input without overwriting legacy results by
default.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from eval_chair import extract_final_response_text, record_final_answer_available  # noqa: E402
from robust_chair_analysis import (  # noqa: E402
    aggregate_from_arrays,
    build_double_word_dict,
    compute_per_sample,
    parse_official_synonyms,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-recompute CHAIR from existing generated-caption JSONL files."
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Files or directories to scan recursively for eval_results.jsonl.",
    )
    parser.add_argument(
        "--metrics-name",
        default="chair_metrics_official.json",
        help="Per-run output filename (default preserves legacy chair_metrics.json).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("res-opd/eval_results/chair_official_recompute_summary.json"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("res-opd/eval_results/chair_official_recompute_summary.csv"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_inputs(roots):
    paths = set()
    for root in roots:
        if root.is_file():
            if root.name != "eval_results.jsonl":
                raise ValueError(f"Expected eval_results.jsonl, got: {root}")
            paths.add(root.resolve())
        elif root.is_dir():
            paths.update(path.resolve() for path in root.rglob("eval_results.jsonl"))
        else:
            raise FileNotFoundError(root)
    return sorted(paths)


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def is_chair_rows(rows):
    if not rows:
        return False
    sample = rows[0]
    return (
        "image_id" in sample
        and "generated_caption" in sample
        and ("gt_objects" in sample or "gt_captions" in sample)
    )


def score(path, mscoco_objects, inverse_synonym_dict, double_word_dict):
    rows = read_jsonl(path)
    if not is_chair_rows(rows):
        return None

    strict_final_answer = any(
        row.get("raw_generated_caption") is not None
        or row.get("thinking_status") is not None
        or row.get("final_answer_available") is not None
        for row in rows
    )
    records = []
    invalid = 0
    errors = 0
    duplicates = 0
    seen = set()
    for row in rows:
        image_id = row.get("image_id")
        caption = str(row.get("generated_caption", "") or "")
        if image_id is None:
            continue
        if image_id in seen:
            duplicates += 1
            continue
        seen.add(image_id)
        if not caption or caption.startswith("[ERROR]"):
            errors += 1
            continue
        if not record_final_answer_available(row, strict_final_answer):
            invalid += 1
            continue
        records.append(
            {
                "image_id": image_id,
                "generated_text": extract_final_response_text(caption),
                "gt_objects": row.get("gt_objects", []),
                "gt_captions": row.get("gt_captions", []),
            }
        )

    if not records:
        raise RuntimeError(f"No valid CHAIR captions in {path}")
    missing_reference_captions = [
        rec["image_id"] for rec in records if not rec["gt_captions"]
    ]
    if missing_reference_captions:
        preview = ", ".join(str(x) for x in missing_reference_captions[:10])
        raise RuntimeError(
            f"Official CHAIR requires reference captions, but {path} has "
            f"{len(missing_reference_captions)} rows without gt_captions "
            f"(first image ids: {preview})."
        )

    per_sample = compute_per_sample(
        records, mscoco_objects, inverse_synonym_dict, double_word_dict
    )
    metrics = aggregate_from_arrays(per_sample, list(per_sample))
    legacy_metrics_path = path.parent / "chair_metrics.json"
    legacy_metrics = {}
    if legacy_metrics_path.exists():
        try:
            legacy_metrics = json.loads(legacy_metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read legacy metrics {legacy_metrics_path}: {exc}") from exc
    previous_chair_i = legacy_metrics.get("CHAIRi")
    previous_chair_s = legacy_metrics.get("CHAIRs")
    metrics.update(
        {
            "num_samples": len(records),
            "total_results": len(rows),
            "invalid_final_answer": invalid,
            "error_or_empty_results": errors,
            "duplicate_image_rows_ignored": duplicates,
            "final_answer_valid_rate": len(records) / len(rows) if rows else 0.0,
            "source_eval_results": str(path),
            "ground_truth_protocol": "instance_categories_union_reference_caption_objects",
            "chair_i_protocol": "all_generated_object_mentions",
            "chair_s_protocol": "captions_with_any_hallucinated_object",
            "previous_metrics_path": str(legacy_metrics_path) if legacy_metrics else None,
            "previous_CHAIRi": previous_chair_i,
            "previous_CHAIRs": previous_chair_s,
            "delta_CHAIRi_vs_previous": (
                metrics["CHAIRi"] - previous_chair_i
                if isinstance(previous_chair_i, (int, float)) else None
            ),
            "delta_CHAIRs_vs_previous": (
                metrics["CHAIRs"] - previous_chair_s
                if isinstance(previous_chair_s, (int, float)) else None
            ),
        }
    )
    return metrics


def main():
    args = parse_args()
    inputs = discover_inputs(args.roots)
    if not inputs:
        raise FileNotFoundError("No eval_results.jsonl files found.")

    mscoco_objects, inverse_synonym_dict = parse_official_synonyms()
    double_word_dict = build_double_word_dict()
    summaries = []
    skipped = []
    failures = []
    for path in inputs:
        try:
            metrics = score(path, mscoco_objects, inverse_synonym_dict, double_word_dict)
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})
            print(f"ERROR {path}: {exc}", file=sys.stderr)
            continue
        if metrics is None:
            skipped.append(str(path))
            continue
        output = path.parent / args.metrics_name
        print(
            f"CHAIR {path}: n={metrics['num_samples']} "
            f"CHAIRi={metrics['CHAIRi']:.6f} CHAIRs={metrics['CHAIRs']:.6f}"
        )
        if not args.dry_run:
            output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
        summaries.append({"result_dir": str(path.parent), "metrics_path": str(output), **metrics})

    if not summaries and not failures:
        raise RuntimeError(
            f"Found {len(inputs)} JSONL files but none had the CHAIR schema; skipped={len(skipped)}"
        )

    payload = {
        "metric_schema": "official_chair_caption_and_mention_v1",
        "num_scored_runs": len(summaries),
        "num_non_chair_jsonl_skipped": len(skipped),
        "num_failed_chair_runs": len(failures),
        "skipped": skipped,
        "failures": failures,
        "runs": summaries,
    }
    if not args.dry_run:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "result_dir", "num_samples", "CHAIRi", "CHAIRs", "ObjPrec", "ObjRecall",
            "ObjF1", "total_object_mentions", "total_hallucinated_mentions",
            "hallucinated_captions", "total_mentioned", "total_hallucinated", "total_correct",
            "previous_CHAIRi", "previous_CHAIRs", "delta_CHAIRi_vs_previous",
            "delta_CHAIRs_vs_previous",
        ]
        with args.summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Wrote summary: {args.summary_json}")
        print(f"Wrote summary: {args.summary_csv}")
    print(f"Scored {len(summaries)} CHAIR runs; skipped {len(skipped)} non-CHAIR JSONL files.")
    if failures:
        print(f"Failed {len(failures)} CHAIR runs; inspect failures in {args.summary_json}.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
