#!/usr/bin/env python3
"""Recompute official-definition CHAIR metrics from existing eval JSONL files.

This script performs no model loading or generation.  It recursively discovers
CHAIR ``eval_results.jsonl`` files, scores their existing captions, and writes a
new metrics file beside each input without overwriting legacy results by
default.
"""

import argparse
import csv
import hashlib
import json
import re
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
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help=(
            "Resolve paper rows against legacy chair_metrics.json files before scoring. "
            "No captions are rescored unless every row resolves unambiguously."
        ),
    )
    parser.add_argument(
        "--resolved-selection-json",
        type=Path,
        default=Path("res-opd/eval_results/chair_paper_selection_resolved.json"),
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


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_name_for_metrics(metrics_path):
    result_dir = metrics_path.parent
    if result_dir.parent != result_dir:
        return result_dir.parent.name
    return result_dir.name


def discover_legacy_metric_candidates(roots):
    candidates = []
    seen = set()
    for root in roots:
        search_paths = []
        if root.is_dir():
            search_paths = root.rglob("chair_metrics.json")
        elif root.is_file() and root.name == "chair_metrics.json":
            search_paths = [root]
        elif not root.exists():
            raise FileNotFoundError(root)
        for metrics_path in search_paths:
            metrics_path = metrics_path.resolve()
            if metrics_path in seen:
                continue
            seen.add(metrics_path)
            eval_results = metrics_path.parent / "eval_results.jsonl"
            if not eval_results.is_file():
                continue
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Could not read {metrics_path}: {exc}") from exc
            candidates.append(
                {
                    "metrics_path": str(metrics_path),
                    "eval_results": str(eval_results.resolve()),
                    "result_dir": str(metrics_path.parent),
                    "experiment_name": experiment_name_for_metrics(metrics_path),
                    "metrics": metrics,
                }
            )
    return candidates


def metric_matches(actual, expected, tolerance):
    return (
        isinstance(actual, (int, float))
        and abs(float(actual) - float(expected)) <= tolerance
    )


def resolve_selection_manifest(roots, manifest_path, output_path, dry_run=False):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Selection manifest has no rows: {manifest_path}")
    candidates = discover_legacy_metric_candidates(roots)
    tolerance = float(manifest.get("metric_tolerance", 0.000051))
    resolved_rows = []
    unresolved_rows = []

    for spec in rows:
        label = spec["label"]
        expected = spec.get("expected", {})
        include = [str(x).lower() for x in spec.get("experiment_name_include", [])]
        exclude = [str(x).lower() for x in spec.get("experiment_name_exclude", [])]
        name_regex = spec.get("experiment_name_regex")
        path_include = [str(x).lower() for x in spec.get("result_path_include", [])]
        path_exclude = [str(x).lower() for x in spec.get("result_path_exclude", [])]
        path_regex = spec.get("result_path_regex")
        metrics_path_exact = spec.get("metrics_path_exact")
        if metrics_path_exact:
            metrics_path_exact = str(Path(metrics_path_exact).expanduser().resolve())
        matches = []
        for candidate in candidates:
            name = candidate["experiment_name"].lower()
            result_path = candidate["result_dir"].lower()
            if metrics_path_exact and candidate["metrics_path"] != metrics_path_exact:
                continue
            if any(token not in name for token in include):
                continue
            if any(token in name for token in exclude):
                continue
            if name_regex and not re.search(name_regex, name, flags=re.IGNORECASE):
                continue
            if any(token not in result_path for token in path_include):
                continue
            if any(token in result_path for token in path_exclude):
                continue
            if path_regex and not re.search(path_regex, result_path, flags=re.IGNORECASE):
                continue
            metrics = candidate["metrics"]
            if not all(
                metric_matches(metrics.get(key), value, tolerance)
                for key, value in expected.items()
            ):
                continue
            matches.append(candidate)

        # Duplicate directory layouts are harmless only when they contain the
        # exact same generated captions. Different hashes remain ambiguous.
        by_hash = {}
        for candidate in matches:
            content_hash = file_sha256(Path(candidate["eval_results"]))
            candidate = {**candidate, "eval_results_sha256": content_hash}
            by_hash.setdefault(content_hash, []).append(candidate)
        if len(by_hash) == 1:
            aliases = next(iter(by_hash.values()))
            selected = sorted(aliases, key=lambda item: (len(item["eval_results"]), item["eval_results"]))[0]
            resolved_rows.append(
                {
                    "label": label,
                    "paper_refs": spec.get("paper_refs", []),
                    "notes": spec.get("notes"),
                    "expected": expected,
                    "selected": selected,
                    "equivalent_aliases": aliases,
                }
            )
        else:
            unresolved_rows.append(
                {
                    "label": label,
                    "paper_refs": spec.get("paper_refs", []),
                    "notes": spec.get("notes"),
                    "expected": expected,
                    "reason": "no_match" if not matches else "multiple_distinct_caption_files",
                    "candidates": [item for values in by_hash.values() for item in values],
                }
            )

    payload = {
        "source_manifest": str(manifest_path),
        "metric_tolerance": tolerance,
        "legacy_metric_candidates_scanned": len(candidates),
        "resolved_count": len(resolved_rows),
        "unresolved_count": len(unresolved_rows),
        "resolved_rows": resolved_rows,
        "unresolved_rows": unresolved_rows,
    }
    # The selection audit is always written, including in dry-run mode; it is
    # the artifact the user reviews before any legacy paper result is replaced.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote selection audit: {output_path}")
    if unresolved_rows:
        labels = ", ".join(row["label"] for row in unresolved_rows)
        raise RuntimeError(
            f"Paper CHAIR selection is ambiguous/incomplete for {len(unresolved_rows)} rows: "
            f"{labels}. Inspect {output_path}; no captions were rescored."
        )
    selected_paths = {
        Path(row["selected"]["eval_results"]).resolve() for row in resolved_rows
    }
    return sorted(selected_paths), resolved_rows


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
    selection_rows = []
    if args.selection_manifest:
        inputs, selection_rows = resolve_selection_manifest(
            args.roots,
            args.selection_manifest,
            args.resolved_selection_json,
            args.dry_run,
        )
    else:
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
    if selection_rows:
        summary_by_source = {
            str(Path(run["source_eval_results"]).resolve()): run for run in summaries
        }
        paper_rows = []
        metric_keys = [
            "CHAIRi", "CHAIRs", "ObjPrec", "ObjRecall", "ObjF1", "RepRate",
            "total_captions", "hallucinated_captions", "total_object_mentions",
            "total_hallucinated_mentions", "total_correct_mentions", "total_mentioned",
            "total_hallucinated", "total_correct",
        ]
        for row in selection_rows:
            selected_path = str(Path(row["selected"]["eval_results"]).resolve())
            official_run = summary_by_source.get(selected_path)
            paper_rows.append(
                {
                    "label": row["label"],
                    "paper_refs": row.get("paper_refs", []),
                    "notes": row.get("notes"),
                    "expected_legacy": row["expected"],
                    "selected_eval_results": selected_path,
                    "official": (
                        {key: official_run.get(key) for key in metric_keys}
                        if official_run else None
                    ),
                }
            )
        payload["paper_rows"] = paper_rows
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
