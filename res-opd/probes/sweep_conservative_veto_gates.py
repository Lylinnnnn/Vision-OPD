#!/usr/bin/env python3
"""
Offline gate sweep for conservative low-res teacher veto candidates.

This is a minimal pre-experiment: it does not generate captions and does not
score models. It reuses an existing pair_kl_trace.jsonl from
probe_same_image_rkl_signal.py, reconstructs object mentions, and asks whether
composed gates are purer than single-feature high-tail buckets.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter


PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROBE_DIR)

from probe_same_image_rkl_signal import (  # noqa: E402
    bool_finite,
    build_canonical_synonym_map,
    clean_mention,
    find_object_mentions,
    fmt,
    parse_official_synonyms,
    percentile,
    safe_mean,
)


DEFAULT_TRACE_JSONL = (
    "/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/"
    "res-opd/probes/results/same_image_rkl_signal/"
    "dual_view_base_sr10_full_vs_lowres075/pair_kl_trace.jsonl"
)
DEFAULT_OUTPUT_ROOT = os.path.join(
    PROBE_DIR,
    "results",
    "conservative_veto_gate_sweep",
    "dual_view_base_sr10_full_vs_lowres075",
)

OBJECT_TYPES = {"correct_object", "hallucinated_object"}

METRIC_LABELS = {
    "rkl_student_to_teacher_mean": "RKL",
    "jsd_mean": "JSD",
    "student_nll_mean": "student_NLL",
    "student_entropy_mean": "student_entropy",
    "teacher_entropy_mean": "teacher_entropy",
    "student_minus_teacher_selected_logprob_mean": "s_minus_t_logp",
    "topk_overlap_ratio_mean": "topk_overlap",
    "top1_match_frac": "top1_match",
}

SUMMARY_METRICS = [
    "rkl_student_to_teacher_mean",
    "jsd_mean",
    "student_nll_mean",
    "student_entropy_mean",
    "teacher_entropy_mean",
    "student_minus_teacher_selected_logprob_mean",
    "topk_overlap_ratio_mean",
    "top1_match_frac",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep object-level low-res disagreement / uncertainty / veto gates "
            "on an existing pair_kl_trace.jsonl."
        )
    )
    parser.add_argument("--trace-jsonl", default=DEFAULT_TRACE_JSONL)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to "
            "res-opd/probes/results/conservative_veto_gate_sweep/"
            "dual_view_base_sr10_full_vs_lowres075/max_examples_<N>."
        ),
    )
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument("--examples-jsonl", default=None)
    parser.add_argument("--max-examples-per-gate", type=int, default=20)
    parser.add_argument(
        "--use-existing-object-mentions",
        action="store_true",
        help="Use object_mentions already present in the trace if available.",
    )
    return parser.parse_args()


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_rate(num, den):
    return num / den if den else 0.0


def metric_value(row, field):
    value = row.get(field)
    return float(value) if bool_finite(value) else None


def finite_values(rows, field):
    return [float(row[field]) for row in rows if bool_finite(row.get(field))]


def quantile_threshold(rows, field, q):
    return percentile(finite_values(rows, field), q)


def threshold_label(field, q):
    return f"{METRIC_LABELS.get(field, field)}_p{int(round(q * 100)):02d}"


def ge_metric(row, field, threshold):
    value = metric_value(row, field)
    return value is not None and threshold is not None and value >= threshold


def le_metric(row, field, threshold):
    value = metric_value(row, field)
    return value is not None and threshold is not None and value <= threshold


def add_record_context(mention, record):
    metadata = record.get("metadata") or {}
    out = dict(mention)
    out.setdefault("image_id", metadata.get("image_id"))
    out["uid"] = metadata.get("uid")
    out["record_index"] = metadata.get("index")
    out["caption_source"] = record.get("caption_source")
    out["pair_mode"] = record.get("pair_mode")
    out["student_ratio"] = record.get("student_ratio")
    out["teacher_ratio"] = record.get("teacher_ratio")
    out["trace_record_num_tokens"] = record.get("num_tokens")
    return out


def build_object_mentions(records, use_existing=False):
    _, inverse_synonym_dict = parse_official_synonyms()
    canonical_synonym_map = build_canonical_synonym_map(inverse_synonym_dict)

    mentions = []
    for record in records:
        object_mentions = None
        if use_existing:
            raw_mentions = record.get("object_mentions")
            if isinstance(raw_mentions, list) and raw_mentions:
                object_mentions = raw_mentions
        if object_mentions is None:
            object_mentions = find_object_mentions(record, canonical_synonym_map)
        mentions.extend(add_record_context(mention, record) for mention in object_mentions)
    return mentions


def summarize_rows(rows):
    out = {"count": len(rows)}
    for field in SUMMARY_METRICS:
        values = finite_values(rows, field)
        if not values:
            continue
        out[field] = safe_mean(values)
        out[f"{field}_p50"] = percentile(values, 0.50)
        out[f"{field}_p75"] = percentile(values, 0.75)
        out[f"{field}_p90"] = percentile(values, 0.90)
    return out


def condition_single_metric(metric, q):
    return {
        "type": "single_metric",
        "metric": metric,
        "q": q,
    }


def condition_risk(disagree_q, uncertainty_q):
    return {
        "type": "risk",
        "disagree_q": disagree_q,
        "uncertainty_q": uncertainty_q,
    }


def condition_veto(disagree_q, uncertainty_q, gap_q, teacher_entropy_q=None, top1_max=None):
    return {
        "type": "directional_veto",
        "disagree_q": disagree_q,
        "uncertainty_q": uncertainty_q,
        "gap_q": gap_q,
        "teacher_entropy_q": teacher_entropy_q,
        "top1_match_max": top1_max,
    }


def default_gate_specs():
    return [
        {
            "name": "single_rkl_p90",
            "description": "Top 10% object mentions by full-vs-lowres RKL.",
            "condition": condition_single_metric("rkl_student_to_teacher_mean", 0.90),
        },
        {
            "name": "single_jsd_p90",
            "description": "Top 10% object mentions by full-vs-lowres JSD.",
            "condition": condition_single_metric("jsd_mean", 0.90),
        },
        {
            "name": "single_student_nll_p90",
            "description": "Top 10% object mentions by student NLL.",
            "condition": condition_single_metric("student_nll_mean", 0.90),
        },
        {
            "name": "single_student_entropy_p90",
            "description": "Top 10% object mentions by student entropy.",
            "condition": condition_single_metric("student_entropy_mean", 0.90),
        },
        {
            "name": "risk_mild_d85_u75",
            "description": "RKL/JSD high tail AND student NLL/entropy high-ish.",
            "condition": condition_risk(0.85, 0.75),
        },
        {
            "name": "risk_mid_d90_u80",
            "description": "RKL/JSD top 10% AND student NLL/entropy top 20%.",
            "condition": condition_risk(0.90, 0.80),
        },
        {
            "name": "risk_strict_d95_u85",
            "description": "Sharper candidate-risk gate.",
            "condition": condition_risk(0.95, 0.85),
        },
        {
            "name": "veto_mild_d85_u75_gap70_te75",
            "description": "Risk gate plus positive student-teacher logp gap and teacher entropy guard.",
            "condition": condition_veto(0.85, 0.75, 0.70, teacher_entropy_q=0.75),
        },
        {
            "name": "veto_mid_d90_u80_gap75_te75",
            "description": "Default conservative veto candidate.",
            "condition": condition_veto(0.90, 0.80, 0.75, teacher_entropy_q=0.75),
        },
        {
            "name": "veto_strict_d90_u85_gap80_te75",
            "description": "Stricter uncertainty and gap gate.",
            "condition": condition_veto(0.90, 0.85, 0.80, teacher_entropy_q=0.75),
        },
        {
            "name": "veto_top1_mismatch_d90_u80_gap75_te75",
            "description": "Default veto plus top-1 mismatch on the object span.",
            "condition": condition_veto(0.90, 0.80, 0.75, teacher_entropy_q=0.75, top1_max=0.5),
        },
    ]


def build_thresholds(labeled_mentions):
    threshold_specs = []
    for metric in (
        "rkl_student_to_teacher_mean",
        "jsd_mean",
        "student_nll_mean",
        "student_entropy_mean",
        "teacher_entropy_mean",
        "student_minus_teacher_selected_logprob_mean",
    ):
        for q in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
            threshold_specs.append((field_key(metric, q), metric, q))

    thresholds = {}
    for key, metric, q in threshold_specs:
        values = finite_values(labeled_mentions, metric)
        if metric == "student_minus_teacher_selected_logprob_mean":
            values = [value for value in values if value > 0.0]
        value = percentile(values, q)
        if metric == "student_minus_teacher_selected_logprob_mean" and value is None:
            value = 0.0
        thresholds[key] = value
    return thresholds


def field_key(field, q):
    return f"{field}__p{int(round(q * 100)):02d}"


def risk_condition(row, thresholds, disagree_q, uncertainty_q):
    rkl_thr = thresholds[field_key("rkl_student_to_teacher_mean", disagree_q)]
    jsd_thr = thresholds[field_key("jsd_mean", disagree_q)]
    nll_thr = thresholds[field_key("student_nll_mean", uncertainty_q)]
    entropy_thr = thresholds[field_key("student_entropy_mean", uncertainty_q)]
    high_disagreement = (
        ge_metric(row, "rkl_student_to_teacher_mean", rkl_thr)
        or ge_metric(row, "jsd_mean", jsd_thr)
    )
    high_uncertainty = (
        ge_metric(row, "student_nll_mean", nll_thr)
        or ge_metric(row, "student_entropy_mean", entropy_thr)
    )
    return high_disagreement and high_uncertainty


def eval_condition(row, thresholds, condition):
    condition_type = condition["type"]
    if condition_type == "single_metric":
        metric = condition["metric"]
        q = condition["q"]
        return ge_metric(row, metric, thresholds[field_key(metric, q)])

    if condition_type == "risk":
        return risk_condition(row, thresholds, condition["disagree_q"], condition["uncertainty_q"])

    if condition_type == "directional_veto":
        if not risk_condition(row, thresholds, condition["disagree_q"], condition["uncertainty_q"]):
            return False
        gap_q = condition["gap_q"]
        gap_thr = thresholds[field_key("student_minus_teacher_selected_logprob_mean", gap_q)]
        if not ge_metric(row, "student_minus_teacher_selected_logprob_mean", gap_thr):
            return False
        teacher_entropy_q = condition.get("teacher_entropy_q")
        if teacher_entropy_q is not None:
            teacher_entropy_thr = thresholds[field_key("teacher_entropy_mean", teacher_entropy_q)]
            if not le_metric(row, "teacher_entropy_mean", teacher_entropy_thr):
                return False
        top1_max = condition.get("top1_match_max")
        if top1_max is not None:
            top1 = metric_value(row, "top1_match_frac")
            if top1 is None or top1 > float(top1_max):
                return False
        return True

    raise ValueError(f"Unsupported condition type: {condition_type}")


def describe_condition(condition, thresholds):
    condition_type = condition["type"]
    if condition_type == "single_metric":
        metric = condition["metric"]
        q = condition["q"]
        key = field_key(metric, q)
        return {
            threshold_label(metric, q): thresholds.get(key),
            "rule": f"{metric} >= {key}",
        }

    if condition_type == "risk":
        d_q = condition["disagree_q"]
        u_q = condition["uncertainty_q"]
        return {
            threshold_label("rkl_student_to_teacher_mean", d_q): thresholds.get(
                field_key("rkl_student_to_teacher_mean", d_q)
            ),
            threshold_label("jsd_mean", d_q): thresholds.get(field_key("jsd_mean", d_q)),
            threshold_label("student_nll_mean", u_q): thresholds.get(
                field_key("student_nll_mean", u_q)
            ),
            threshold_label("student_entropy_mean", u_q): thresholds.get(
                field_key("student_entropy_mean", u_q)
            ),
            "rule": "(RKL high OR JSD high) AND (student_NLL high OR student_entropy high)",
        }

    if condition_type == "directional_veto":
        out = describe_condition(
            condition_risk(condition["disagree_q"], condition["uncertainty_q"]),
            thresholds,
        )
        gap_q = condition["gap_q"]
        out[threshold_label("student_minus_teacher_selected_logprob_mean", gap_q)] = thresholds.get(
            field_key("student_minus_teacher_selected_logprob_mean", gap_q)
        )
        teacher_entropy_q = condition.get("teacher_entropy_q")
        if teacher_entropy_q is not None:
            out[threshold_label("teacher_entropy_mean", teacher_entropy_q) + "_max"] = thresholds.get(
                field_key("teacher_entropy_mean", teacher_entropy_q)
            )
        if condition.get("top1_match_max") is not None:
            out["top1_match_max"] = condition.get("top1_match_max")
        out["rule"] = (
            out["rule"]
            + " AND student_minus_teacher_logp high"
            + (" AND teacher_entropy not high" if teacher_entropy_q is not None else "")
            + (" AND top1 mostly mismatched" if condition.get("top1_match_max") is not None else "")
        )
        return out

    return {}


def summarize_gate(name, description, selected, labeled, thresholds_used):
    total_hallucinated = len([row for row in labeled if row["object_type"] == "hallucinated_object"])
    total_correct = len([row for row in labeled if row["object_type"] == "correct_object"])
    total_labeled = total_hallucinated + total_correct
    base_rate = safe_rate(total_hallucinated, total_labeled)

    selected_hallucinated = len([row for row in selected if row["object_type"] == "hallucinated_object"])
    selected_correct = len([row for row in selected if row["object_type"] == "correct_object"])
    selected_total = selected_hallucinated + selected_correct
    precision = safe_rate(selected_hallucinated, selected_total)
    recall = safe_rate(selected_hallucinated, total_hallucinated)
    correct_fpr = safe_rate(selected_correct, total_correct)
    f1 = safe_rate(2 * precision * recall, precision + recall)

    return {
        "name": name,
        "description": description,
        "selected_total": selected_total,
        "selected_fraction": safe_rate(selected_total, total_labeled),
        "selected_hallucinated": selected_hallucinated,
        "selected_correct": selected_correct,
        "hallucination_precision": precision,
        "hallucination_recall": recall,
        "correct_false_positive_rate": correct_fpr,
        "hallucination_f1": f1,
        "precision_lift_vs_base": precision / base_rate if base_rate > 0 else None,
        "thresholds": thresholds_used,
        "selected_metric_summary": summarize_rows(selected),
    }


def risk_score(row):
    parts = []
    for field in (
        "rkl_student_to_teacher_mean",
        "jsd_mean",
        "student_nll_mean",
        "student_entropy_mean",
        "student_minus_teacher_selected_logprob_mean",
    ):
        value = metric_value(row, field)
        if value is not None:
            parts.append(value)
    return sum(parts) if parts else -math.inf


def compact_example(row, gate_name):
    out = clean_mention(row)
    out["gate_name"] = gate_name
    out["uid"] = row.get("uid")
    out["record_index"] = row.get("record_index")
    out["pair_mode"] = row.get("pair_mode")
    out["student_ratio"] = row.get("student_ratio")
    out["teacher_ratio"] = row.get("teacher_ratio")
    return out


def collect_examples(gate_name, selected, max_examples):
    if max_examples <= 0:
        return []
    ordered = sorted(selected, key=risk_score, reverse=True)
    hallucinated = [row for row in ordered if row["object_type"] == "hallucinated_object"][:max_examples]
    correct = [row for row in ordered if row["object_type"] == "correct_object"][:max_examples]
    examples = []
    examples.extend(compact_example(row, gate_name) for row in hallucinated)
    examples.extend(compact_example(row, gate_name) for row in correct)
    return examples


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        clean = [str(value).replace("\n", " ").replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(clean) + " |")
    return "\n".join(lines)


def write_markdown(summary, path):
    gates = summary.get("gates", [])
    sorted_by_precision = sorted(
        gates,
        key=lambda gate: (
            gate.get("hallucination_precision") or 0.0,
            gate.get("hallucination_recall") or 0.0,
            -(gate.get("correct_false_positive_rate") or 0.0),
        ),
        reverse=True,
    )
    sorted_by_f1 = sorted(
        gates,
        key=lambda gate: (
            gate.get("hallucination_f1") or 0.0,
            gate.get("hallucination_precision") or 0.0,
            -(gate.get("correct_false_positive_rate") or 0.0),
        ),
        reverse=True,
    )

    lines = [
        "# Conservative Veto Gate Sweep",
        "",
        "## Inputs",
        "",
        f"- trace_jsonl: `{summary.get('trace_jsonl')}`",
        f"- records: {summary.get('num_records')}",
        f"- object_mentions: {summary.get('num_object_mentions')}",
        f"- labeled_object_mentions: {summary.get('num_labeled_mentions')}",
        f"- correct_object: {summary.get('num_correct_mentions')}",
        f"- hallucinated_object: {summary.get('num_hallucinated_mentions')}",
        f"- base_hallucination_rate: {fmt(summary.get('base_hallucination_rate'))}",
        "",
        "## Object Metrics",
        "",
    ]

    object_summary = summary.get("object_type_summary", {})
    object_rows = []
    for object_type in ("correct_object", "hallucinated_object"):
        stats = object_summary.get(object_type, {})
        object_rows.append(
            [
                object_type,
                stats.get("count", 0),
                fmt(stats.get("rkl_student_to_teacher_mean")),
                fmt(stats.get("jsd_mean")),
                fmt(stats.get("student_nll_mean")),
                fmt(stats.get("student_entropy_mean")),
                fmt(stats.get("teacher_entropy_mean")),
                fmt(stats.get("student_minus_teacher_selected_logprob_mean")),
            ]
        )
    lines.append(
        markdown_table(
            [
                "group",
                "count",
                "RKL",
                "JSD",
                "student NLL",
                "student entropy",
                "teacher entropy",
                "s-t logp",
            ],
            object_rows,
        )
    )

    def gate_rows(items):
        rows = []
        for gate in items:
            rows.append(
                [
                    gate.get("name"),
                    gate.get("selected_total"),
                    gate.get("selected_hallucinated"),
                    gate.get("selected_correct"),
                    fmt(gate.get("selected_fraction")),
                    fmt(gate.get("hallucination_precision")),
                    fmt(gate.get("hallucination_recall")),
                    fmt(gate.get("correct_false_positive_rate")),
                    fmt(gate.get("hallucination_f1")),
                    fmt(gate.get("precision_lift_vs_base")),
                ]
            )
        return rows

    gate_headers = [
        "gate",
        "selected",
        "halluc",
        "correct",
        "sel frac",
        "precision",
        "recall",
        "correct FPR",
        "halluc F1",
        "lift",
    ]
    lines.extend(["", "## Gates Sorted By Precision", ""])
    lines.append(markdown_table(gate_headers, gate_rows(sorted_by_precision)))
    lines.extend(["", "## Gates Sorted By Hallucination F1", ""])
    lines.append(markdown_table(gate_headers, gate_rows(sorted_by_f1)))

    lines.extend(["", "## Thresholds", ""])
    threshold_rows = []
    for gate in gates:
        thresholds = gate.get("thresholds", {})
        threshold_rows.append(
            [
                gate.get("name"),
                thresholds.get("rule", ""),
                "; ".join(
                    f"{key}={fmt(value)}"
                    for key, value in thresholds.items()
                    if key != "rule"
                ),
            ]
        )
    lines.append(markdown_table(["gate", "rule", "thresholds"], threshold_rows))

    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- `precision` is hallucination precision among selected object mentions.",
            "- `recall` is selected hallucinated mentions divided by all hallucinated mentions.",
            "- `correct FPR` is selected correct mentions divided by all correct mentions; this is the recall-damage proxy.",
            "- Directional veto uses `student_minus_teacher_selected_logprob_mean > 0`, so the low-res teacher gives lower probability to the generated object token than the full-res student.",
            "",
        ]
    )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_sweep(args):
    if not os.path.exists(args.trace_jsonl):
        raise SystemExit(
            f"Trace JSONL not found: {args.trace_jsonl}\n"
            "Set TRACE_JSONL=... or pass --trace-jsonl to the launcher."
        )
    if os.path.getsize(args.trace_jsonl) == 0:
        raise SystemExit(f"Trace JSONL is empty: {args.trace_jsonl}")

    records = read_jsonl(args.trace_jsonl)
    if not records:
        raise SystemExit(f"No records loaded from {args.trace_jsonl}")

    object_mentions = build_object_mentions(records, args.use_existing_object_mentions)
    labeled_mentions = [row for row in object_mentions if row.get("object_type") in OBJECT_TYPES]
    if not labeled_mentions:
        raise SystemExit(
            "No labeled correct/hallucinated object mentions were found. "
            "Check that the trace has token_records and COCO gt_objects metadata."
        )

    object_counts = Counter(row.get("object_type") for row in labeled_mentions)
    total_correct = object_counts.get("correct_object", 0)
    total_hallucinated = object_counts.get("hallucinated_object", 0)
    base_rate = safe_rate(total_hallucinated, len(labeled_mentions))

    thresholds = build_thresholds(labeled_mentions)
    gates = []
    examples = []
    for spec in default_gate_specs():
        selected = [
            row for row in labeled_mentions
            if eval_condition(row, thresholds, spec["condition"])
        ]
        thresholds_used = describe_condition(spec["condition"], thresholds)
        gate_summary = summarize_gate(
            spec["name"],
            spec["description"],
            selected,
            labeled_mentions,
            thresholds_used,
        )
        gates.append(gate_summary)
        examples.extend(collect_examples(spec["name"], selected, args.max_examples_per_gate))

    summary = {
        "trace_jsonl": args.trace_jsonl,
        "num_records": len(records),
        "num_object_mentions": len(object_mentions),
        "num_labeled_mentions": len(labeled_mentions),
        "num_correct_mentions": total_correct,
        "num_hallucinated_mentions": total_hallucinated,
        "base_hallucination_rate": base_rate,
        "object_type_summary": {
            object_type: summarize_rows(
                [row for row in labeled_mentions if row.get("object_type") == object_type]
            )
            for object_type in ("correct_object", "hallucinated_object")
        },
        "thresholds": thresholds,
        "gates": gates,
        "object_type_counts": dict(object_counts),
    }
    return summary, examples


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(
        DEFAULT_OUTPUT_ROOT,
        f"max_examples_{args.max_examples_per_gate}",
    )
    summary_json = args.summary_json or os.path.join(output_dir, "gate_sweep_summary.json")
    summary_md = args.summary_md or os.path.join(output_dir, "gate_sweep_summary.md")
    examples_jsonl = args.examples_jsonl or os.path.join(output_dir, "gate_sweep_examples.jsonl")

    summary, examples = run_sweep(args)
    write_json(summary_json, summary)
    write_markdown(summary, summary_md)
    write_jsonl(examples_jsonl, examples)

    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote summary MD:   {summary_md}")
    print(f"Wrote examples:     {examples_jsonl}")


if __name__ == "__main__":
    main()
