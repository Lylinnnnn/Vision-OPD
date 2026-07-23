#!/usr/bin/env python3
"""Post-process model-delta traces into surgery-style mechanism tables.

This script does not run model forward passes.  It consumes
``model_delta_trace.jsonl`` files produced by ``probe_model_delta_logprob.py``
and builds the multi-axis analysis needed for the paper:

  object role x cross-view RKL x emitted-token uncertainty x teacher stance

The important naming convention is that ``offline_risk_tail`` is a bin computed
from the base trace.  For RiskMask checkpoints it approximates the trained mask;
for uniform Frozen RKL it is only a counterfactual stratification.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Any


PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_OPD_ROOT = os.path.dirname(PROBE_DIR)
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, os.path.join(RES_OPD_ROOT, "eval"))

from probe_model_delta_logprob import (  # noqa: E402
    TOKEN_METRIC_FIELDS,
    aggregate_token_span,
    assign_risk_scores,
    bool_finite,
    classify_mechanisms,
    enrich_mentions,
    flatten_tokens,
    fmt,
    group_summary,
    labeled_mentions,
    markdown_table,
    percentile,
    safe_div,
    safe_mean,
    summarize_rows,
)
from score_opd_eval_trace import as_int, load_jsonl, load_test_index, normalize_generation_record  # noqa: E402


DEFAULT_OUTPUT_ROOT = os.path.join(PROBE_DIR, "results", "model_delta_logprob_surgery")

TOKEN_MEAN_FIELDS = [
    "rkl_base_to_teacher",
    "jsd_base_teacher",
    "base_nll",
    "base_entropy",
    "teacher_nll",
    "teacher_entropy",
    "teacher_minus_base_nll",
    "rkl_after_to_base",
    "jsd_after_base",
    "delta_logp_after_base",
    "delta_entropy_after_base",
    "delta_top1_prob_after_base",
    "delta_top1_margin_after_base",
    "after_base_topk_jaccard",
    "after_mass_on_base_topk",
    "delta_after_base_mass_on_teacher_topk",
    "base_selected_rank",
    "after_selected_rank",
]

MENTION_MEAN_FIELDS = [
    "rkl_base_to_teacher_mean",
    "base_nll_mean",
    "base_entropy_mean",
    "teacher_nll_mean",
    "teacher_entropy_mean",
    "teacher_minus_base_nll_mean",
    "rkl_after_to_base_mean",
    "jsd_after_base_mean",
    "delta_logp_after_base_mean",
    "delta_entropy_after_base_mean",
    "after_base_topk_jaccard_mean",
    "riskmask_nll_selected_token_frac",
    "mechanism_sharpen_same_top1_frac",
    "mechanism_reshape_topk_frac",
    "mechanism_suppress_emitted_frac",
    "mechanism_boost_emitted_frac",
    "mechanism_stable_frac",
]

MECHANISMS = [
    "stable",
    "sharpen_same_top1",
    "reshape_topk",
    "suppress_emitted",
    "boost_emitted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build quadrant/fate/surgery summaries from model-delta traces."
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help=(
            "Run spec in NAME=PATH format. PATH can be a run directory containing "
            "model_delta_trace.jsonl or a trace JSONL file. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Run directory containing model_delta_trace.jsonl. Name is inferred from summary/config.",
    )
    parser.add_argument(
        "--trace-glob",
        action="append",
        default=[],
        help="Glob for trace JSONL files. Name is inferred from the parent directory.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--examples-jsonl", default=None)
    parser.add_argument("--top-p", type=float, default=0.30)
    parser.add_argument(
        "--high-frac",
        type=float,
        default=0.30,
        help="Fraction treated as high for RKL/NLL/entropy quadrant thresholds.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--test-json", default=os.path.join(RES_OPD_ROOT, "data", "test_1000.json"))
    parser.add_argument("--caption-field", default="auto")
    parser.add_argument("--after-eval-results", default=None)
    parser.add_argument("--logp-eps", type=float, default=0.05)
    parser.add_argument("--entropy-eps", type=float, default=0.02)
    parser.add_argument("--prob-eps", type=float, default=0.02)
    parser.add_argument("--topk-jaccard-threshold", type=float, default=0.50)
    parser.add_argument("--max-examples-per-group", type=int, default=20)
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def safe_path_part(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("_") or "unknown"


def maybe_load_json(path: str | None) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_run_name(path: str, summary: dict[str, Any]) -> str:
    config = summary.get("config") or {}
    after_name = config.get("after_name")
    if after_name:
        return safe_path_part(after_name)
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return safe_path_part(parent)


def resolve_path(path: str) -> tuple[str, str | None]:
    if os.path.isdir(path):
        trace = os.path.join(path, "model_delta_trace.jsonl")
        summary = os.path.join(path, "model_delta_summary.json")
        return trace, summary if os.path.exists(summary) else None
    return path, None


def collect_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[tuple[str | None, str]] = []
    for spec in args.run:
        if "=" in spec:
            name, path = spec.split("=", 1)
            specs.append((name.strip(), path.strip()))
        else:
            specs.append((None, spec.strip()))
    for path in args.run_dir:
        specs.append((None, path))
    for pattern in args.trace_glob:
        matches = sorted(glob.glob(pattern))
        if not matches:
            fail(f"--trace-glob matched no files: {pattern}")
        specs.extend((None, path) for path in matches)

    missing = []
    runs = []
    seen = set()
    for explicit_name, raw_path in specs:
        trace, summary_path = resolve_path(raw_path)
        trace_abs = os.path.abspath(trace)
        if trace_abs in seen:
            continue
        seen.add(trace_abs)
        if not os.path.exists(trace):
            missing.append(trace)
            continue
        summary = maybe_load_json(summary_path)
        name = safe_path_part(explicit_name) if explicit_name else infer_run_name(trace, summary)
        config = summary.get("config") or {}
        after_eval_results = args.after_eval_results or config.get("after_eval_results")
        test_json = config.get("test_json") or args.test_json
        runs.append(
            {
                "name": name,
                "trace_jsonl": trace_abs,
                "summary_json": os.path.abspath(summary_path) if summary_path else None,
                "summary": summary,
                "after_eval_results": after_eval_results,
                "test_json": test_json,
                "caption_field": config.get("caption_field") or args.caption_field,
            }
        )

    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        fail(f"Missing model_delta_trace.jsonl file(s):\n{lines}")
    if not runs:
        fail("No runs were provided.")
    return runs


def load_records(path: str) -> list[dict[str, Any]]:
    records = []
    for idx, record in enumerate(load_jsonl(path)):
        record = dict(record)
        record["_record_index"] = idx
        records.append(record)
    if not records:
        fail(f"Trace JSONL is empty: {path}")
    return records


def load_caption_index(path: str | None, test_json: str, caption_field: str) -> dict[int, str]:
    if not path:
        return {}
    test_index = load_test_index(test_json)
    out: dict[int, str] = {}
    for raw in load_jsonl(path):
        record = normalize_generation_record(raw, test_index, caption_field)
        image_id = as_int(record.get("image_id"))
        caption = record.get("generated_caption") or ""
        if image_id is None or not caption or caption.startswith("[ERROR]"):
            continue
        out[image_id] = caption
    return out


def finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if bool_finite(row.get(field))]


def add_teacher_delta(rows: list[dict[str, Any]], suffix: str = "") -> None:
    base_field = f"base_nll{suffix}"
    teacher_field = f"teacher_nll{suffix}"
    out_field = f"teacher_minus_base_nll{suffix}"
    for row in rows:
        if bool_finite(row.get(base_field)) and bool_finite(row.get(teacher_field)):
            row[out_field] = float(row[teacher_field]) - float(row[base_field])


def quantile_thresholds(rows: list[dict[str, Any]], high_frac: float, prefix: str = "") -> dict[str, Any]:
    q_hi = 1.0 - high_frac
    q_lo = high_frac
    fields = {
        "rkl": f"rkl_base_to_teacher{prefix}",
        "nll": f"base_nll{prefix}",
        "entropy": f"base_entropy{prefix}",
        "teacher_entropy": f"teacher_entropy{prefix}",
        "teacher_minus_base_nll": f"teacher_minus_base_nll{prefix}",
    }
    out: dict[str, Any] = {"high_frac": high_frac, "q_hi": q_hi, "q_lo": q_lo}
    for key, field in fields.items():
        values = finite_values(rows, field)
        if values:
            out[f"{key}_high"] = percentile(values, q_hi)
            out[f"{key}_low"] = percentile(values, q_lo)
    return out


def bin_ge(value: Any, threshold: Any, high_label: str, low_label: str) -> str:
    if not bool_finite(value) or not bool_finite(threshold):
        return "unknown"
    return high_label if float(value) >= float(threshold) else low_label


def teacher_stance(value: Any, low_threshold: Any, high_threshold: Any) -> str:
    if not bool_finite(value) or not bool_finite(low_threshold) or not bool_finite(high_threshold):
        return "unknown"
    value = float(value)
    if value >= float(high_threshold):
        return "teacher_rejects"
    if value <= float(low_threshold):
        return "teacher_supports"
    return "teacher_neutral"


def assign_bins(rows: list[dict[str, Any]], thresholds: dict[str, Any], prefix: str = "") -> None:
    for row in rows:
        row["_rkl_bin"] = bin_ge(row.get(f"rkl_base_to_teacher{prefix}"), thresholds.get("rkl_high"), "highRKL", "lowRKL")
        row["_nll_bin"] = bin_ge(row.get(f"base_nll{prefix}"), thresholds.get("nll_high"), "highNLL", "lowNLL")
        row["_entropy_bin"] = bin_ge(
            row.get(f"base_entropy{prefix}"), thresholds.get("entropy_high"), "highEnt", "lowEnt"
        )
        row["_teacher_unc_bin"] = bin_ge(
            row.get(f"teacher_entropy{prefix}"),
            thresholds.get("teacher_entropy_high"),
            "teacherUnc",
            "teacherCert",
        )
        row["_teacher_stance"] = teacher_stance(
            row.get(f"teacher_minus_base_nll{prefix}"),
            thresholds.get("teacher_minus_base_nll_low"),
            thresholds.get("teacher_minus_base_nll_high"),
        )
        if row.get("riskmask_nll_selected") is True or row.get("riskmask_nll_selected_token_frac", 0.0) > 0.0:
            row["_offline_risk_bin"] = "offline_risk_tail"
        else:
            row["_offline_risk_bin"] = "offline_safe_bulk"


def rate(rows: list[dict[str, Any]], predicate) -> float | None:
    if not rows:
        return None
    return safe_mean([1.0 if predicate(row) else 0.0 for row in rows])


def stats_for_rows(rows: list[dict[str, Any]], total_count: int, *, mention: bool = False) -> dict[str, Any]:
    metric_fields = MENTION_MEAN_FIELDS if mention else TOKEN_MEAN_FIELDS
    out = summarize_rows(rows, metric_fields)
    out["share"] = safe_div(len(rows), total_count) or 0.0
    if not mention:
        object_tokens = [row for row in rows if row.get("token_role") in {"correct_object", "hallucinated_object"}]
        out["object_token_count"] = len(object_tokens)
        out["hallucinated_object_token_rate"] = rate(
            object_tokens,
            lambda row: row.get("token_role") == "hallucinated_object",
        )
        out["offline_risk_tail_rate"] = rate(rows, lambda row: row.get("riskmask_nll_selected") is True)
        for mechanism in MECHANISMS:
            out[f"{mechanism}_rate"] = rate(rows, lambda row, m=mechanism: row.get("mechanism") == m)
        out["after_base_top1_match_rate"] = rate(rows, lambda row: row.get("after_base_top1_match") is True)
    else:
        labeled = [row for row in rows if row.get("object_type") in {"correct_object", "hallucinated_object"}]
        out["hallucination_rate"] = rate(labeled, lambda row: row.get("object_type") == "hallucinated_object")
        out["removed_rate"] = rate(labeled, lambda row: str(row.get("after_change_status", "")).startswith("removed_"))
        out["removed_hallucination_rate"] = rate(
            labeled,
            lambda row: row.get("after_change_status") == "removed_hallucinated",
        )
        out["removed_correct_rate"] = rate(labeled, lambda row: row.get("after_change_status") == "removed_correct")
    return out


def group_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "unknown")) for key in keys)].append(row)
    return grouped


def flatten_group_table(
    *,
    run_name: str,
    table_name: str,
    rows: list[dict[str, Any]],
    keys: list[str],
    total_count: int,
    mention: bool = False,
) -> list[dict[str, Any]]:
    out = []
    for key_values, group in sorted(group_rows(rows, keys).items()):
        stats = stats_for_rows(group, total_count, mention=mention)
        flat = {"run": run_name, "table": table_name}
        flat.update({key: value for key, value in zip(keys, key_values)})
        flat.update(stats)
        out.append(flat)
    return out


def top_rows(rows: list[dict[str, Any]], key, limit: int, reverse: bool = False) -> list[dict[str, Any]]:
    usable = [row for row in rows if key(row) is not None]
    return sorted(usable, key=key, reverse=reverse)[:limit]


def compact_example(row: dict[str, Any], run_name: str, group: str) -> dict[str, Any]:
    keys = [
        "image_id",
        "object_type",
        "after_change_status",
        "canonical_object",
        "mention_text",
        "context",
        "base_caption",
        "after_caption",
        "image_path",
        "_span_token_count",
        "_rkl_bin",
        "_nll_bin",
        "_entropy_bin",
        "_teacher_stance",
        "_teacher_unc_bin",
        "_offline_risk_bin",
        "riskmask_nll_selected_token_frac",
        "rkl_base_to_teacher_mean",
        "base_nll_mean",
        "base_entropy_mean",
        "teacher_minus_base_nll_mean",
        "rkl_after_to_base_mean",
        "jsd_after_base_mean",
        "delta_logp_after_base_mean",
        "delta_entropy_after_base_mean",
        "after_base_topk_jaccard_mean",
        "mechanism_suppress_emitted_frac",
        "mechanism_reshape_topk_frac",
        "mechanism_sharpen_same_top1_frac",
    ]
    out = {"run": run_name, "example_group": group}
    out.update({key: row.get(key) for key in keys if key in row})
    return out


def collect_examples(run_name: str, mentions: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    labeled = labeled_mentions(mentions)
    examples: list[dict[str, Any]] = []

    groups = {
        "removed_hallucination_highrisk_suppressed": top_rows(
            [
                row for row in labeled
                if row.get("after_change_status") == "removed_hallucinated"
                and row.get("_offline_risk_bin") == "offline_risk_tail"
            ],
            key=lambda row: float(row.get("delta_logp_after_base_mean", 0.0))
            if bool_finite(row.get("delta_logp_after_base_mean"))
            else None,
            limit=limit,
            reverse=False,
        ),
        "kept_correct_lowrisk_stable": top_rows(
            [
                row for row in labeled
                if row.get("after_change_status") == "kept_correct"
                and row.get("_offline_risk_bin") == "offline_safe_bulk"
            ],
            key=lambda row: abs(float(row.get("delta_logp_after_base_mean", 0.0)))
            if bool_finite(row.get("delta_logp_after_base_mean"))
            else None,
            limit=limit,
            reverse=False,
        ),
        "confident_hallucination_hard_case": top_rows(
            [
                row for row in labeled
                if row.get("object_type") == "hallucinated_object"
                and row.get("_nll_bin") == "lowNLL"
                and row.get("_rkl_bin") == "highRKL"
            ],
            key=lambda row: float(row.get("rkl_base_to_teacher_mean", 0.0))
            if bool_finite(row.get("rkl_base_to_teacher_mean"))
            else None,
            limit=limit,
            reverse=True,
        ),
        "uncertain_correct_preservation_or_damage": top_rows(
            [
                row for row in labeled
                if row.get("object_type") == "correct_object"
                and row.get("_nll_bin") == "highNLL"
                and row.get("_rkl_bin") == "highRKL"
            ],
            key=lambda row: float(row.get("jsd_after_base_mean", 0.0))
            if bool_finite(row.get("jsd_after_base_mean"))
            else None,
            limit=limit,
            reverse=True,
        ),
        "failure_removed_correct": top_rows(
            [row for row in labeled if row.get("after_change_status") == "removed_correct"],
            key=lambda row: float(row.get("riskmask_nll_selected_token_frac", 0.0))
            if bool_finite(row.get("riskmask_nll_selected_token_frac"))
            else None,
            limit=limit,
            reverse=True,
        ),
        "failure_kept_hallucinated_highrisk": top_rows(
            [
                row for row in labeled
                if row.get("after_change_status") == "kept_hallucinated"
                and row.get("_offline_risk_bin") == "offline_risk_tail"
            ],
            key=lambda row: float(row.get("riskmask_nll_selected_token_frac", 0.0))
            if bool_finite(row.get("riskmask_nll_selected_token_frac"))
            else None,
            limit=limit,
            reverse=True,
        ),
    }

    for group_name, rows in groups.items():
        for row in rows:
            examples.append(compact_example(row, run_name, group_name))
    return examples


def process_run(run: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(run["trace_jsonl"])
    tokens = flatten_tokens(records)
    if not tokens:
        fail(f"Trace has no token records: {run['trace_jsonl']}")

    assign_risk_scores(tokens, args.top_p, args.random_seed)
    classify_mechanisms(tokens, args)
    add_teacher_delta(tokens, "")

    enrich_args = SimpleNamespace(
        after_eval_results=run.get("after_eval_results"),
        test_json=run.get("test_json") or args.test_json,
        caption_field=run.get("caption_field") or args.caption_field,
    )
    mentions, mention_meta = enrich_mentions(records, enrich_args)
    add_teacher_delta(mentions, "_mean")
    labeled = labeled_mentions(mentions)
    after_captions = load_caption_index(
        run.get("after_eval_results"),
        run.get("test_json") or args.test_json,
        run.get("caption_field") or args.caption_field,
    )
    records_by_index = {record["_record_index"]: record for record in records}
    for mention in mentions:
        record = records_by_index.get(mention.get("_record_index"))
        metadata = (record or {}).get("metadata") or {}
        extra_info = metadata.get("extra_info") if isinstance(metadata.get("extra_info"), dict) else {}
        image_id = as_int(mention.get("image_id"))
        mention["base_caption"] = (record or {}).get("response_text")
        mention["after_caption"] = after_captions.get(image_id)
        mention["image_path"] = extra_info.get("image_path")

    token_thresholds = quantile_thresholds(tokens, args.high_frac, "")
    mention_thresholds = quantile_thresholds(labeled, args.high_frac, "_mean")
    assign_bins(tokens, token_thresholds, "")
    assign_bins(labeled, mention_thresholds, "_mean")

    token_tables = []
    token_tables.extend(
        flatten_group_table(
            run_name=run["name"],
            table_name="token_role_x_rkl_nll",
            rows=tokens,
            keys=["token_role", "_rkl_bin", "_nll_bin"],
            total_count=len(tokens),
        )
    )
    token_tables.extend(
        flatten_group_table(
            run_name=run["name"],
            table_name="token_role_x_rkl_entropy",
            rows=tokens,
            keys=["token_role", "_rkl_bin", "_entropy_bin"],
            total_count=len(tokens),
        )
    )
    token_tables.extend(
        flatten_group_table(
            run_name=run["name"],
            table_name="token_role_x_teacher_stance_nll",
            rows=tokens,
            keys=["token_role", "_teacher_stance", "_nll_bin"],
            total_count=len(tokens),
        )
    )
    token_tables.extend(
        flatten_group_table(
            run_name=run["name"],
            table_name="token_offline_risk_bin",
            rows=tokens,
            keys=["_offline_risk_bin"],
            total_count=len(tokens),
        )
    )

    mention_tables = []
    if labeled:
        mention_tables.extend(
            flatten_group_table(
                run_name=run["name"],
                table_name="mention_object_x_rkl_nll",
                rows=labeled,
                keys=["object_type", "_rkl_bin", "_nll_bin"],
                total_count=len(labeled),
                mention=True,
            )
        )
        mention_tables.extend(
            flatten_group_table(
                run_name=run["name"],
                table_name="mention_object_x_offline_risk_bin",
                rows=labeled,
                keys=["object_type", "_offline_risk_bin"],
                total_count=len(labeled),
                mention=True,
            )
        )
        if mention_meta.get("after_object_records"):
            mention_tables.extend(
                flatten_group_table(
                    run_name=run["name"],
                    table_name="mention_fate_x_rkl_nll",
                    rows=labeled,
                    keys=["object_type", "after_change_status", "_rkl_bin", "_nll_bin"],
                    total_count=len(labeled),
                    mention=True,
                )
            )
            mention_tables.extend(
                flatten_group_table(
                    run_name=run["name"],
                    table_name="mention_fate_x_teacher_stance",
                    rows=labeled,
                    keys=["object_type", "after_change_status", "_teacher_stance"],
                    total_count=len(labeled),
                    mention=True,
                )
            )
            mention_tables.extend(
                flatten_group_table(
                    run_name=run["name"],
                    table_name="mention_fate_x_offline_risk_bin",
                    rows=labeled,
                    keys=["object_type", "after_change_status", "_offline_risk_bin"],
                    total_count=len(labeled),
                    mention=True,
                )
            )

    focus_buckets = {
        "correct_lowrisk_certain": [
            row for row in labeled
            if row.get("object_type") == "correct_object"
            and row.get("_rkl_bin") == "lowRKL"
            and row.get("_nll_bin") == "lowNLL"
        ],
        "correct_highrisk_uncertain": [
            row for row in labeled
            if row.get("object_type") == "correct_object"
            and row.get("_rkl_bin") == "highRKL"
            and row.get("_nll_bin") == "highNLL"
        ],
        "hallucination_highrisk_uncertain": [
            row for row in labeled
            if row.get("object_type") == "hallucinated_object"
            and row.get("_rkl_bin") == "highRKL"
            and row.get("_nll_bin") == "highNLL"
        ],
        "hallucination_highrisk_confident": [
            row for row in labeled
            if row.get("object_type") == "hallucinated_object"
            and row.get("_rkl_bin") == "highRKL"
            and row.get("_nll_bin") == "lowNLL"
        ],
        "hallucination_lowrisk_uncertain": [
            row for row in labeled
            if row.get("object_type") == "hallucinated_object"
            and row.get("_rkl_bin") == "lowRKL"
            and row.get("_nll_bin") == "highNLL"
        ],
    }
    focus_summary = {
        name: stats_for_rows(rows, len(labeled), mention=True)
        for name, rows in focus_buckets.items()
    }

    return {
        "run": run,
        "num_records": len(records),
        "num_tokens": len(tokens),
        "num_mentions": len(mentions),
        "num_labeled_mentions": len(labeled),
        "token_thresholds": token_thresholds,
        "mention_thresholds": mention_thresholds,
        "token_role_summary": group_summary(tokens, "token_role", TOKEN_METRIC_FIELDS),
        "mention_object_summary": group_summary(labeled, "object_type", [f"{field}_mean" for field in TOKEN_METRIC_FIELDS]),
        "mention_fate_summary": group_summary(labeled, "after_change_status", [f"{field}_mean" for field in TOKEN_METRIC_FIELDS])
        if mention_meta.get("after_object_records")
        else {},
        "focus_summary": focus_summary,
        "token_tables": token_tables,
        "mention_tables": mention_tables,
        "examples": collect_examples(run["name"], labeled, args.max_examples_per_group),
        "mechanism_counts": dict(Counter(row.get("mechanism", "unknown") for row in tokens)),
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8g}"
    if value is None:
        return ""
    return value


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
    if not rows:
        return
    keys = []
    seen = set()
    preferred = [
        "run",
        "table",
        "token_role",
        "object_type",
        "after_change_status",
        "_rkl_bin",
        "_nll_bin",
        "_entropy_bin",
        "_teacher_stance",
        "_teacher_unc_bin",
        "_offline_risk_bin",
        "count",
        "share",
        "object_token_count",
        "hallucinated_object_token_rate",
        "hallucination_rate",
        "removed_rate",
        "offline_risk_tail_rate",
        "rkl_base_to_teacher",
        "rkl_base_to_teacher_mean",
        "base_nll",
        "base_nll_mean",
        "base_entropy",
        "base_entropy_mean",
        "teacher_minus_base_nll",
        "teacher_minus_base_nll_mean",
        "rkl_after_to_base",
        "rkl_after_to_base_mean",
        "jsd_after_base",
        "jsd_after_base_mean",
        "delta_logp_after_base",
        "delta_logp_after_base_mean",
        "delta_entropy_after_base",
        "delta_entropy_after_base_mean",
        "after_base_top1_match_rate",
        "after_base_topk_jaccard",
        "after_base_topk_jaccard_mean",
        "stable_rate",
        "sharpen_same_top1_rate",
        "reshape_topk_rate",
        "suppress_emitted_rate",
        "boost_emitted_rate",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in sorted(row):
            if key not in seen:
                keys.append(key)
                seen.add(key)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def write_examples(examples: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def key_stats_row(name: str, stats: dict[str, Any]) -> list[Any]:
    return [
        name,
        stats.get("count", 0),
        fmt(stats.get("share")),
        fmt(stats.get("hallucination_rate")),
        fmt(stats.get("removed_rate")),
        fmt(stats.get("rkl_base_to_teacher_mean")),
        fmt(stats.get("base_nll_mean")),
        fmt(stats.get("base_entropy_mean")),
        fmt(stats.get("teacher_minus_base_nll_mean")),
        fmt(stats.get("jsd_after_base_mean")),
        fmt(stats.get("delta_logp_after_base_mean")),
        fmt(stats.get("delta_entropy_after_base_mean")),
        fmt(stats.get("mechanism_sharpen_same_top1_frac")),
        fmt(stats.get("mechanism_reshape_topk_frac")),
        fmt(stats.get("mechanism_suppress_emitted_frac")),
    ]


def token_bin_row(name: str, stats: dict[str, Any]) -> list[Any]:
    return [
        name,
        stats.get("count", 0),
        fmt(stats.get("share")),
        fmt(stats.get("offline_risk_tail_rate")),
        fmt(stats.get("hallucinated_object_token_rate")),
        fmt(stats.get("rkl_base_to_teacher")),
        fmt(stats.get("base_nll")),
        fmt(stats.get("base_entropy")),
        fmt(stats.get("teacher_minus_base_nll")),
        fmt(stats.get("jsd_after_base")),
        fmt(stats.get("delta_logp_after_base")),
        fmt(stats.get("delta_entropy_after_base")),
        fmt(stats.get("after_base_top1_match_rate")),
        fmt(stats.get("after_base_topk_jaccard")),
        fmt(stats.get("sharpen_same_top1_rate")),
        fmt(stats.get("reshape_topk_rate")),
        fmt(stats.get("suppress_emitted_rate")),
    ]


def write_markdown(summary: dict[str, Any], path: str) -> None:
    lines = [
        "# Model-Delta Surgery Probe",
        "",
        "This is a post-processing analysis over `model_delta_trace.jsonl`. It is designed to explain *where* OPD training changes the base-caption distribution, and whether RiskMask behaves like a targeted edit rather than a generic 30% weakening of RKL.",
        "",
        "Important: `offline_risk_tail` means tokens that would be selected by the RiskMask score on the base trace. For a RiskMask checkpoint this approximates the trained mask; for Frozen RKL it is only a counterfactual bin, because Frozen RKL applies RKL to all valid response tokens.",
        "",
        "## Runs",
        "",
    ]
    run_rows = []
    for run_name, run in summary["runs"].items():
        run_rows.append([
            run_name,
            run.get("num_records"),
            run.get("num_tokens"),
            run.get("num_labeled_mentions"),
            run.get("trace_jsonl"),
        ])
    lines.append(markdown_table(["run", "records", "tokens", "labeled mentions", "trace"], run_rows))

    lines.extend([
        "",
        "## Focus Mention Buckets",
        "",
        "These buckets directly answer the paper questions: stable correct mentions, uncertain correct mentions, high-risk hallucinations, confident hallucinations, and low-RKL uncertain hallucinations.",
        "",
    ])
    focus_headers = [
        "run/bucket",
        "N",
        "share",
        "hallu rate",
        "removed rate",
        "gate RKL",
        "base NLL",
        "base Ent.",
        "teacher-base NLL",
        "JSD(after,base)",
        "Delta logp",
        "Delta Ent.",
        "Sharpen",
        "Reshape",
        "Suppress",
    ]
    focus_rows = []
    for run_name, run in summary["runs"].items():
        for bucket in (
            "correct_lowrisk_certain",
            "correct_highrisk_uncertain",
            "hallucination_highrisk_uncertain",
            "hallucination_highrisk_confident",
            "hallucination_lowrisk_uncertain",
        ):
            stats = run.get("focus_summary", {}).get(bucket, {})
            focus_rows.append(key_stats_row(f"{run_name}/{bucket}", stats))
    lines.append(markdown_table(focus_headers, focus_rows))

    lines.extend([
        "",
        "## Offline Risk Tail vs Safe Bulk",
        "",
        "Use this table to show whether training changes are localized. Do not call the Frozen RKL risk tail a training mask.",
        "",
    ])
    risk_rows = []
    for run_name, run in summary["runs"].items():
        for row in run.get("token_tables", []):
            if row.get("table") != "token_offline_risk_bin":
                continue
            risk_rows.append(token_bin_row(f"{run_name}/{row.get('_offline_risk_bin')}", row))
    lines.append(markdown_table([
        "run/bin",
        "N",
        "share",
        "risk rate",
        "hallu obj tok rate",
        "gate RKL",
        "base NLL",
        "base Ent.",
        "teacher-base NLL",
        "JSD(after,base)",
        "Delta logp",
        "Delta Ent.",
        "Top1 same",
        "TopK Jac.",
        "Sharpen",
        "Reshape",
        "Suppress",
    ], risk_rows))

    lines.extend([
        "",
        "## Figure Suggestions",
        "",
        "1. Heatmap: `token_role_x_rkl_nll`, with color = suppress rate or JSD(after,base), faceted by run. This directly shows where RKL/RiskMask performs surgery.",
        "2. Stacked bars: focus mention buckets, with bars = kept/removed and color = correct/hallucinated. This shows why hallucinated objects decrease while correct objects are preserved.",
        "3. Scatter/hexbin from `surgery_quadrants.csv`: x = gate RKL, y = base NLL, color = after_change_status, facet = run. This is the most intuitive plot for the RiskMask story.",
        "4. Example panel: use `surgery_examples.jsonl`, especially `removed_hallucination_highrisk_suppressed`, `kept_correct_lowrisk_stable`, and `failure_removed_correct`.",
        "",
        "## Writing Guidance",
        "",
        "- The central claim should be targeted editing: RiskMask acts on high-RKL/high-NLL regions while leaving low-risk correct regions stable.",
        "- For Frozen RKL, `offline_risk_tail` is a diagnostic bin only. Phrase it as \"tokens that would be selected by RiskMask\".",
        "- If high-risk correct mentions are mostly sharpened/kept, write that RiskMask protects visually valid but uncertain mentions by not blindly suppressing every uncertain token.",
        "- If confident hallucinations remain hard, write it as a limitation: NLL-based RiskMask is strongest when hallucination is both visually unstable and low-confidence; confidently hallucinated priors need stronger teacher-rejection or entropy-based variants.",
        "- Use examples for the figure. Numeric means alone are too abstract for this mechanism claim.",
        "",
    ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.top_p <= 1.0:
        fail("--top-p must satisfy 0 < top_p <= 1.")
    if not 0.0 < args.high_frac < 1.0:
        fail("--high-frac must satisfy 0 < high_frac < 1.")
    runs = collect_runs(args)
    os.makedirs(args.output_dir, exist_ok=True)
    summary_json = args.summary_json or os.path.join(args.output_dir, "surgery_summary.json")
    summary_md = args.summary_md or os.path.join(args.output_dir, "surgery_summary.md")
    csv_path = args.csv or os.path.join(args.output_dir, "surgery_quadrants.csv")
    examples_path = args.examples_jsonl or os.path.join(args.output_dir, "surgery_examples.jsonl")

    all_csv_rows: list[dict[str, Any]] = []
    all_examples: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "top_p": args.top_p,
        "high_frac": args.high_frac,
        "runs": {},
        "csv": csv_path,
        "examples_jsonl": examples_path,
    }
    for run in runs:
        processed = process_run(run, args)
        run_summary = {
            "trace_jsonl": run["trace_jsonl"],
            "summary_json": run.get("summary_json"),
            "after_eval_results": run.get("after_eval_results"),
            "num_records": processed["num_records"],
            "num_tokens": processed["num_tokens"],
            "num_mentions": processed["num_mentions"],
            "num_labeled_mentions": processed["num_labeled_mentions"],
            "token_thresholds": processed["token_thresholds"],
            "mention_thresholds": processed["mention_thresholds"],
            "token_role_summary": processed["token_role_summary"],
            "mention_object_summary": processed["mention_object_summary"],
            "mention_fate_summary": processed["mention_fate_summary"],
            "focus_summary": processed["focus_summary"],
            "mechanism_counts": processed["mechanism_counts"],
        }
        summary["runs"][run["name"]] = run_summary
        all_csv_rows.extend(processed["token_tables"])
        all_csv_rows.extend(processed["mention_tables"])
        all_examples.extend(processed["examples"])

    write_csv(all_csv_rows, csv_path)
    write_examples(all_examples, examples_path)
    os.makedirs(os.path.dirname(os.path.abspath(summary_json)), exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_markdown(summary, summary_md)
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved summary Markdown: {summary_md}")
    print(f"Saved figure CSV: {csv_path}")
    print(f"Saved examples JSONL: {examples_path}")


if __name__ == "__main__":
    main()
