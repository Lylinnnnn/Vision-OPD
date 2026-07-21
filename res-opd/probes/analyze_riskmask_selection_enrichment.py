#!/usr/bin/env python3
"""Analyze whether RiskMask-selected tokens are enriched for hallucinated objects.

This script is intentionally offline: it consumes JSONL traces produced by
``probe_same_image_rkl_signal.py`` and does not run model forward passes.  The
main diagnostic mirrors the training-time RiskMask score:

    rank(RKL_student_to_teacher) * rank(student NLL)

over valid response tokens, then selects the top-p tokens/mentions.  It also
reports RKL-only, NLL-only, entropy-only, RiskEntropy, and random top-p controls
so the paper can argue that the combined high-risk tail is not a generic
"keep fewer tokens" effect.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Any


PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(PROBE_DIR), "eval"))

from probe_same_image_rkl_signal import (  # noqa: E402
    build_canonical_synonym_map,
    bool_finite,
    find_object_mentions,
    parse_official_synonyms,
)
from score_opd_eval_trace import load_jsonl  # noqa: E402


SCORE_MODES = [
    "riskmask_nll",
    "riskmask_entropy",
    "rkl_only",
    "nll_only",
    "entropy_only",
    "random",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute top-p RiskMask selection enrichment over object mentions "
            "from same-image/dual-view KL trace JSONL files."
        )
    )
    parser.add_argument(
        "--trace-jsonl",
        action="append",
        default=[],
        help="Trace JSONL path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--trace-glob",
        action="append",
        default=[],
        help="Glob pattern for sharded trace JSONLs. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument("--mentions-jsonl", default=None)
    parser.add_argument("--top-p", type=float, default=0.30)
    parser.add_argument(
        "--rank-scope",
        choices=["global", "record"],
        default="global",
        help=(
            "global ranks over all trace tokens; record ranks within each caption. "
            "Training ranks within the current valid-token batch, so global is a "
            "stable offline approximation and record is a locality sanity check."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--mention-token-threshold",
        type=float,
        default=0.0,
        help=(
            "For token-overlap mention selection, require selected token fraction "
            "to be greater than this value. Default 0 means any selected token."
        ),
    )
    parser.add_argument("--top-examples", type=int, default=30)
    return parser.parse_args()


def fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def safe_div(num: float, den: float) -> float | None:
    return num / den if den else None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        clean = [str(value).replace("\n", " ").replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(clean) + " |")
    return "\n".join(lines)


def resolve_trace_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    for pattern in args.trace_glob:
        matches = sorted(glob.glob(pattern))
        if not matches:
            fail(f"--trace-glob matched no files: {pattern}")
        paths.extend(matches)
    paths.extend(args.trace_jsonl)

    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        fail(f"Missing trace JSONL path(s):\n{lines}")

    unique = []
    seen = set()
    for path in paths:
        abs_path = os.path.abspath(path)
        if abs_path not in seen:
            unique.append(abs_path)
            seen.add(abs_path)
    if not unique:
        fail("No trace JSONL inputs were provided.")
    return unique


def load_records(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        count = 0
        for record in load_jsonl(path):
            record = dict(record)
            record["_trace_path"] = path
            record["_record_index"] = len(records)
            records.append(record)
            count += 1
        if count == 0:
            fail(f"Trace JSONL is empty: {path}")
    if not records:
        fail("No records loaded from trace inputs.")
    return records


def token_metric(token: dict[str, Any], field: str) -> float | None:
    value = token.get(field)
    if bool_finite(value):
        return float(value)
    return None


def percentile_ranks(values: list[float]) -> list[float]:
    n = len(values)
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda idx: values[idx])
    ranks = [0.0] * n
    for rank, idx in enumerate(order):
        ranks[idx] = rank / float(n - 1)
    return ranks


def assign_ranks_global(tokens: list[dict[str, Any]], rng: random.Random) -> None:
    for field, out_field in (
        ("rkl_student_to_teacher", "_rank_rkl"),
        ("student_nll", "_rank_nll"),
        ("student_entropy", "_rank_entropy"),
    ):
        valid = [(idx, token_metric(token, field)) for idx, token in enumerate(tokens)]
        valid = [(idx, value) for idx, value in valid if value is not None]
        ranks = percentile_ranks([value for _, value in valid])
        for (idx, _), rank in zip(valid, ranks):
            tokens[idx][out_field] = rank
    for token in tokens:
        token["_rank_random"] = rng.random()


def assign_ranks_by_record(records: list[dict[str, Any]], rng: random.Random) -> None:
    for record in records:
        tokens = record.get("token_records") or []
        assign_ranks_global(tokens, rng)


def compute_scores(tokens: list[dict[str, Any]]) -> None:
    for token in tokens:
        rkl = token.get("_rank_rkl")
        nll = token.get("_rank_nll")
        entropy = token.get("_rank_entropy")
        random_rank = token.get("_rank_random")
        if bool_finite(rkl) and bool_finite(nll):
            token["_score_riskmask_nll"] = float(rkl) * float(nll)
        if bool_finite(rkl) and bool_finite(entropy):
            token["_score_riskmask_entropy"] = float(rkl) * float(entropy)
        if bool_finite(rkl):
            token["_score_rkl_only"] = float(rkl)
        if bool_finite(nll):
            token["_score_nll_only"] = float(nll)
        if bool_finite(entropy):
            token["_score_entropy_only"] = float(entropy)
        if bool_finite(random_rank):
            token["_score_random"] = float(random_rank)


def select_top_p(tokens: list[dict[str, Any]], mode: str, top_p: float) -> dict[str, Any]:
    score_field = f"_score_{mode}"
    valid = [token for token in tokens if bool_finite(token.get(score_field))]
    selected_field = f"_selected_{mode}"
    for token in tokens:
        token[selected_field] = False
    if not valid:
        return {
            "mode": mode,
            "valid_tokens": 0,
            "selected_tokens": 0,
            "selected_token_frac": 0.0,
            "threshold": None,
        }
    k = max(1, int(math.ceil(len(valid) * top_p)))
    valid_sorted = sorted(valid, key=lambda token: float(token[score_field]), reverse=True)
    threshold = float(valid_sorted[k - 1][score_field])
    selected = [token for token in valid if float(token[score_field]) >= threshold]
    for token in selected:
        token[selected_field] = True
    return {
        "mode": mode,
        "valid_tokens": len(valid),
        "selected_tokens": len(selected),
        "selected_token_frac": safe_div(len(selected), len(valid)) or 0.0,
        "threshold": threshold,
    }


def record_image_id(record: dict[str, Any]) -> Any:
    metadata = record.get("metadata") or {}
    return metadata.get("image_id")


def attach_mentions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _, inverse_synonym_dict = parse_official_synonyms()
    canonical_synonym_map = build_canonical_synonym_map(inverse_synonym_dict)
    mentions: list[dict[str, Any]] = []
    for record in records:
        record_mentions = record.get("object_mentions")
        if not record_mentions:
            record_mentions = find_object_mentions(record, canonical_synonym_map)
        for mention in record_mentions:
            row = dict(mention)
            row["_record_index"] = record["_record_index"]
            row["image_id"] = row.get("image_id", record_image_id(record))
            row["caption_source"] = row.get("caption_source") or record.get("caption_source")
            mentions.append(row)
    return mentions


def span_tokens(record: dict[str, Any], mention: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = record.get("token_records") or []
    start = int(mention.get("token_start", 0) or 0)
    end = int(mention.get("token_end", start) or start)
    start = max(0, min(start, len(tokens)))
    end = max(start, min(end, len(tokens)))
    return tokens[start:end]


def mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if bool_finite(v)]
    return sum(vals) / len(vals) if vals else None


def enrich_mentions(records: list[dict[str, Any]], mentions: list[dict[str, Any]]) -> None:
    by_index = {record["_record_index"]: record for record in records}
    for mention in mentions:
        record = by_index.get(mention["_record_index"])
        tokens = span_tokens(record, mention) if record else []
        mention["_span_token_count"] = len(tokens)
        for mode in SCORE_MODES:
            score_field = f"_score_{mode}"
            selected_field = f"_selected_{mode}"
            scores = [float(token[score_field]) for token in tokens if bool_finite(token.get(score_field))]
            selected_count = sum(1 for token in tokens if token.get(selected_field))
            mention[f"{mode}_score_mean"] = mean(scores)
            mention[f"{mode}_selected_token_count"] = selected_count
            mention[f"{mode}_selected_token_frac"] = safe_div(selected_count, len(tokens)) or 0.0


def summarize_selection(
    labeled_mentions: list[dict[str, Any]],
    *,
    mode: str,
    top_p: float,
    base_rate: float,
    total_hallucinated: int,
    total_correct: int,
    mention_token_threshold: float,
) -> dict[str, Any]:
    object_type_key = "object_type"
    score_key = f"{mode}_score_mean"
    eligible = [row for row in labeled_mentions if bool_finite(row.get(score_key))]
    selected_by_span: list[dict[str, Any]]
    if eligible:
        k = max(1, int(math.ceil(len(eligible) * top_p)))
        ordered = sorted(eligible, key=lambda row: float(row[score_key]), reverse=True)
        threshold = float(ordered[k - 1][score_key])
        selected_by_span = [row for row in eligible if float(row[score_key]) >= threshold]
    else:
        threshold = None
        selected_by_span = []

    token_frac_key = f"{mode}_selected_token_frac"
    selected_by_token = [
        row
        for row in labeled_mentions
        if bool_finite(row.get(token_frac_key)) and float(row[token_frac_key]) > mention_token_threshold
    ]

    def bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
        hallucinated = sum(1 for row in rows if row.get(object_type_key) == "hallucinated_object")
        correct = sum(1 for row in rows if row.get(object_type_key) == "correct_object")
        selected = hallucinated + correct
        precision = safe_div(hallucinated, selected) or 0.0
        return {
            "selected": selected,
            "selected_fraction": safe_div(selected, len(labeled_mentions)) or 0.0,
            "hallucinated": hallucinated,
            "correct": correct,
            "hallucination_precision": precision,
            "hallucination_recall": safe_div(hallucinated, total_hallucinated) or 0.0,
            "correct_selection_rate": safe_div(correct, total_correct) or 0.0,
            "precision_lift": safe_div(precision, base_rate) if base_rate else None,
        }

    out = {
        "mode": mode,
        "span_top_p": bucket(selected_by_span),
        "token_overlap": bucket(selected_by_token),
        "span_score_threshold": threshold,
        "eligible_mentions": len(eligible),
    }
    return out


def clean_example(row: dict[str, Any], mode: str) -> dict[str, Any]:
    keep = [
        "image_id",
        "caption_source",
        "object_type",
        "canonical_object",
        "mention_text",
        "context",
        "token_start",
        "token_end",
        "_span_token_count",
        f"{mode}_score_mean",
        f"{mode}_selected_token_frac",
        "rkl_student_to_teacher_mean",
        "student_nll_mean",
        "student_entropy_mean",
        "teacher_entropy_mean",
    ]
    return {key: row.get(key) for key in keep if key in row}


def write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(summary: dict[str, Any], path: str) -> None:
    lines = [
        "# RiskMask Selection Enrichment Probe",
        "",
        "## Inputs",
        "",
        f"- trace_jsonl: {', '.join(f'`{p}`' for p in summary['trace_jsonl'])}",
        f"- rank_scope: `{summary['rank_scope']}`",
        f"- top_p: {fmt(summary['top_p'])}",
        f"- records: {summary['num_records']}",
        f"- tokens: {summary['num_tokens']}",
        f"- labeled object mentions: {summary['num_labeled_object_mentions']}",
        f"- base hallucination rate: {fmt(summary['base_hallucination_rate'])}",
        "",
        "## Token Selection",
        "",
    ]
    token_rows = []
    for mode in SCORE_MODES:
        stats = summary["token_selection"].get(mode, {})
        token_rows.append([
            mode,
            stats.get("valid_tokens", 0),
            stats.get("selected_tokens", 0),
            fmt(stats.get("selected_token_frac")),
            fmt(stats.get("threshold")),
        ])
    lines.append(markdown_table(["mode", "valid", "selected", "selected frac", "threshold"], token_rows))

    lines.extend(["", "## Object Mention Enrichment", ""])
    rows = []
    for mode in SCORE_MODES:
        stats = summary["mention_selection"].get(mode, {})
        span = stats.get("span_top_p", {})
        token = stats.get("token_overlap", {})
        rows.append([
            mode,
            span.get("selected", 0),
            fmt(span.get("hallucination_precision")),
            fmt(span.get("hallucination_recall")),
            fmt(span.get("correct_selection_rate")),
            fmt(span.get("precision_lift")),
            token.get("selected", 0),
            fmt(token.get("hallucination_precision")),
            fmt(token.get("hallucination_recall")),
            fmt(token.get("precision_lift")),
        ])
    lines.append(markdown_table(
        [
            "mode",
            "span selected",
            "span precision",
            "span recall",
            "span correct sel.",
            "span lift",
            "token selected",
            "token precision",
            "token recall",
            "token lift",
        ],
        rows,
    ))

    lines.extend(["", "## Top RiskMask Hallucinated Mentions", ""])
    example_rows = []
    for row in summary.get("top_riskmask_hallucinated_examples", []):
        example_rows.append([
            row.get("image_id"),
            row.get("canonical_object"),
            fmt(row.get("riskmask_nll_score_mean")),
            fmt(row.get("riskmask_nll_selected_token_frac")),
            str(row.get("context", ""))[:180],
        ])
    lines.append(markdown_table(["image", "object", "score", "token frac", "context"], example_rows))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.top_p <= 1.0):
        fail("--top-p must satisfy 0 < top_p <= 1")
    if args.mention_token_threshold < 0.0 or args.mention_token_threshold >= 1.0:
        fail("--mention-token-threshold must satisfy 0 <= threshold < 1")

    trace_paths = resolve_trace_paths(args)
    records = load_records(trace_paths)
    all_tokens = []
    for record in records:
        for token_index, token in enumerate(record.get("token_records") or []):
            token["_record_index"] = record["_record_index"]
            token["_token_index"] = token_index
            all_tokens.append(token)
    if not all_tokens:
        fail("Trace records contain no token_records.")

    rng = random.Random(args.random_seed)
    if args.rank_scope == "global":
        assign_ranks_global(all_tokens, rng)
    else:
        assign_ranks_by_record(records, rng)
    compute_scores(all_tokens)

    token_selection = {
        mode: select_top_p(all_tokens, mode, args.top_p)
        for mode in SCORE_MODES
    }

    mentions = attach_mentions(records)
    enrich_mentions(records, mentions)
    labeled_mentions = [
        row for row in mentions
        if row.get("object_type") in {"correct_object", "hallucinated_object"}
    ]
    if not labeled_mentions:
        fail("No labeled correct/hallucinated object mentions found.")
    total_hallucinated = sum(1 for row in labeled_mentions if row.get("object_type") == "hallucinated_object")
    total_correct = sum(1 for row in labeled_mentions if row.get("object_type") == "correct_object")
    base_rate = safe_div(total_hallucinated, len(labeled_mentions)) or 0.0

    mention_selection = {
        mode: summarize_selection(
            labeled_mentions,
            mode=mode,
            top_p=args.top_p,
            base_rate=base_rate,
            total_hallucinated=total_hallucinated,
            total_correct=total_correct,
            mention_token_threshold=args.mention_token_threshold,
        )
        for mode in SCORE_MODES
    }

    risk_examples = sorted(
        [
            row for row in labeled_mentions
            if row.get("object_type") == "hallucinated_object"
            and bool_finite(row.get("riskmask_nll_score_mean"))
        ],
        key=lambda row: float(row.get("riskmask_nll_score_mean")),
        reverse=True,
    )[: args.top_examples]
    correct_pressure_examples = sorted(
        [
            row for row in labeled_mentions
            if row.get("object_type") == "correct_object"
            and bool_finite(row.get("riskmask_nll_score_mean"))
        ],
        key=lambda row: float(row.get("riskmask_nll_score_mean")),
        reverse=True,
    )[: args.top_examples]

    os.makedirs(args.output_dir, exist_ok=True)
    summary_json = args.summary_json or os.path.join(args.output_dir, "riskmask_selection_enrichment_summary.json")
    summary_md = args.summary_md or os.path.join(args.output_dir, "riskmask_selection_enrichment_summary.md")
    mentions_jsonl = args.mentions_jsonl or os.path.join(args.output_dir, "riskmask_selection_mentions.jsonl")

    summary = {
        "trace_jsonl": trace_paths,
        "rank_scope": args.rank_scope,
        "top_p": args.top_p,
        "random_seed": args.random_seed,
        "mention_token_threshold": args.mention_token_threshold,
        "num_records": len(records),
        "num_tokens": len(all_tokens),
        "num_object_mentions": len(mentions),
        "num_labeled_object_mentions": len(labeled_mentions),
        "num_hallucinated_object_mentions": total_hallucinated,
        "num_correct_object_mentions": total_correct,
        "base_hallucination_rate": base_rate,
        "token_selection": token_selection,
        "mention_selection": mention_selection,
        "top_riskmask_hallucinated_examples": [
            clean_example(row, "riskmask_nll") for row in risk_examples
        ],
        "top_riskmask_correct_pressure_examples": [
            clean_example(row, "riskmask_nll") for row in correct_pressure_examples
        ],
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_markdown(summary, summary_md)
    mention_rows = []
    for row in labeled_mentions:
        out = clean_example(row, "riskmask_nll")
        for mode in SCORE_MODES:
            out[f"{mode}_score_mean"] = row.get(f"{mode}_score_mean")
            out[f"{mode}_selected_token_frac"] = row.get(f"{mode}_selected_token_frac")
        mention_rows.append(out)
    write_jsonl(mention_rows, mentions_jsonl)

    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved summary Markdown: {summary_md}")
    print(f"Saved mention JSONL: {mentions_jsonl}")


if __name__ == "__main__":
    main()
