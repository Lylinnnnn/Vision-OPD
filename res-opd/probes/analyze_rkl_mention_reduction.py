#!/usr/bin/env python3
"""
Analyze which baseline object mentions disappear after RKL training.

This is an offline diagnostic: it does not run model inference or teacher
forward. It compares two eval_results.jsonl files and joins baseline mentions
with a previously generated low-res object probe JSONL.
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict


PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_OPD_ROOT = os.path.dirname(PROBE_DIR)
EVAL_DIR = os.path.join(RES_OPD_ROOT, "eval")
sys.path.insert(0, EVAL_DIR)

from robust_chair_analysis import (  # noqa: E402
    build_double_word_dict,
    caption_to_words,
    parse_official_synonyms,
)


CAPTION_FIELDS = ("generated_caption", "output", "response_text", "caption")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline vs RKL captions and test whether removed object "
            "mentions come from both-reject/high-entropy/low-support regions."
        )
    )
    parser.add_argument("--baseline-results", required=True, help="Baseline eval_results.jsonl")
    parser.add_argument("--model-results", required=True, help="RKL/model eval_results.jsonl")
    parser.add_argument(
        "--baseline-probe-jsonl-glob",
        required=True,
        help=(
            "Glob for baseline lowres_teacher_object_probe*.jsonl files produced "
            "with --probe-views student,teacher."
        ),
    )
    parser.add_argument(
        "--test-json",
        default=None,
        help="Optional test JSON for GT fallback when eval_results lacks gt_objects.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <model-results dir>/mention_reduction_analysis.",
    )
    parser.add_argument("--support-margin", type=float, default=0.5)
    parser.add_argument("--confident-entropy-max", type=float, default=0.65)
    parser.add_argument("--top-state-pairs", type=int, default=12)
    return parser.parse_args()


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def bool_finite(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def safe_div(num, den):
    return num / den if den else None


def safe_mean(values):
    vals = [float(v) for v in values if bool_finite(v)]
    return sum(vals) / len(vals) if vals else None


def percentile(values, q):
    vals = sorted(float(v) for v in values if bool_finite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def infer_caption(record):
    for field in CAPTION_FIELDS:
        value = record.get(field)
        if value and not str(value).startswith("[ERROR]"):
            return str(value)
    return ""


def load_jsonl_records(path):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_id = as_int(record.get("image_id") or record.get("sample_id"))
            if image_id is None:
                continue
            caption = infer_caption(record)
            if not caption:
                continue
            records[image_id] = record
    return records


def load_test_index(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        samples = data.get("samples") or data.get("data") or data.get("images") or []
    else:
        samples = data
    out = {}
    for sample in samples:
        image_id = as_int(sample.get("image_id") or sample.get("id"))
        if image_id is not None:
            out[image_id] = sample
    return out


def get_gt_objects(record, test_index):
    for key in ("gt_objects", "objects", "mscoco_objects"):
        value = record.get(key)
        if value:
            return set(value)
    sample = test_index.get(as_int(record.get("image_id") or record.get("sample_id")), {})
    for key in ("gt_objects", "objects", "mscoco_objects"):
        value = sample.get(key)
        if value:
            return set(value)
    return set()


def extract_objects(caption, mscoco_objects, inverse_synonym_dict, double_word_dict):
    _, node_words = caption_to_words(caption, mscoco_objects, inverse_synonym_dict, double_word_dict)
    return set(node_words)


def classify_state(margin, entropy, support_margin, confident_entropy_max):
    if not bool_finite(margin) or not bool_finite(entropy):
        return "unknown"
    if float(entropy) > confident_entropy_max:
        return "uncertain"
    if float(margin) >= support_margin:
        return "support"
    if float(margin) <= -support_margin:
        return "reject"
    return "uncertain"


def state_from_probe(probe, prefix, support_margin, confident_entropy_max):
    state = probe.get(f"{prefix}_state")
    if state:
        return state
    return classify_state(
        probe.get(f"{prefix}_yes_minus_no_logprob"),
        probe.get(f"{prefix}_binary_entropy"),
        support_margin,
        confident_entropy_max,
    )


def reject_quadrant(student_state, teacher_state):
    student_reject = student_state == "reject"
    teacher_reject = teacher_state == "reject"
    if student_reject and teacher_reject:
        return "both_reject"
    if student_reject and not teacher_reject:
        return "student_only_reject"
    if teacher_reject and not student_reject:
        return "teacher_only_reject"
    return "neither_reject"


def aggregate_mention_metrics(mentions):
    buckets = defaultdict(list)
    for mention in mentions:
        metrics = mention.get("token_metrics") or {}
        for key, value in metrics.items():
            if bool_finite(value):
                buckets[key].append(float(value))
    return {key: safe_mean(values) for key, values in buckets.items()}


def load_probe_index(jsonl_glob, support_margin, confident_entropy_max):
    paths = sorted(glob.glob(jsonl_glob))
    if not paths:
        raise FileNotFoundError(f"No baseline probe JSONL files matched: {jsonl_glob}")

    index = {}
    record_count = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                image_id = as_int(record.get("image_id"))
                if image_id is None:
                    continue
                record_count += 1
                object_probes = {}
                for probe in record.get("object_probes", []):
                    obj = probe.get("object")
                    if not obj:
                        continue
                    student_state = state_from_probe(
                        probe, "student", support_margin, confident_entropy_max
                    )
                    teacher_state = state_from_probe(
                        probe, "teacher", support_margin, confident_entropy_max
                    )
                    probe = dict(probe)
                    probe["student_state"] = student_state
                    probe["teacher_state"] = teacher_state
                    probe["state_pair"] = f"student={student_state}|teacher={teacher_state}"
                    probe["reject_quadrant"] = reject_quadrant(student_state, teacher_state)
                    object_probes[obj] = probe

                mention_groups = defaultdict(list)
                for mention in record.get("object_mentions", []):
                    obj = mention.get("object")
                    if obj:
                        mention_groups[obj].append(mention)

                object_mentions = {
                    obj: {
                        "mention_count": len(mentions),
                        "token_metrics": aggregate_mention_metrics(mentions),
                    }
                    for obj, mentions in mention_groups.items()
                }
                index[image_id] = {
                    "object_probes": object_probes,
                    "object_mentions": object_mentions,
                    "caption": record.get("caption") or record.get("decoded_caption") or "",
                }
    return index, {"paths": paths, "record_count": record_count}


def base_object_feature(image_id, obj, label, probe_index):
    sample_probe = probe_index.get(image_id, {})
    object_probe = (sample_probe.get("object_probes") or {}).get(obj, {})
    mention_info = (sample_probe.get("object_mentions") or {}).get(obj, {})
    token_metrics = mention_info.get("token_metrics") or {}

    student_state = object_probe.get("student_state", "unknown")
    teacher_state = object_probe.get("teacher_state", "unknown")
    state_pair = object_probe.get("state_pair", f"student={student_state}|teacher={teacher_state}")
    quadrant = object_probe.get("reject_quadrant", reject_quadrant(student_state, teacher_state))

    student_entropy = token_metrics.get("student_entropy_mean")
    teacher_entropy = token_metrics.get("teacher_entropy_mean")
    max_entropy = None
    if bool_finite(student_entropy) and bool_finite(teacher_entropy):
        max_entropy = max(float(student_entropy), float(teacher_entropy))
    elif bool_finite(student_entropy):
        max_entropy = float(student_entropy)
    elif bool_finite(teacher_entropy):
        max_entropy = float(teacher_entropy)

    return {
        "image_id": image_id,
        "object": obj,
        "label": label,
        "student_state": student_state,
        "teacher_state": teacher_state,
        "state_pair": state_pair,
        "reject_quadrant": quadrant,
        "both_support": student_state == "support" and teacher_state == "support",
        "low_cross_resolution_support": not (student_state == "support" and teacher_state == "support"),
        "teacher_non_support": teacher_state != "support",
        "student_non_support": student_state != "support",
        "student_margin": object_probe.get("student_yes_minus_no_logprob"),
        "teacher_margin": object_probe.get("teacher_yes_minus_no_logprob"),
        "student_binary_entropy": object_probe.get("student_binary_entropy"),
        "teacher_binary_entropy": object_probe.get("teacher_binary_entropy"),
        "mention_count": mention_info.get("mention_count", 0),
        "student_entropy": student_entropy,
        "teacher_entropy": teacher_entropy,
        "max_entropy": max_entropy,
        "teacher_minus_student_logprob": token_metrics.get("teacher_minus_student_logprob_mean"),
        "teacher_minus_student_entropy": token_metrics.get("teacher_minus_student_entropy_mean"),
        "student_selected_logprob": token_metrics.get("student_selected_logprob_mean"),
        "teacher_selected_logprob": token_metrics.get("teacher_selected_logprob_mean"),
        "student_top1_mass": token_metrics.get("student_top1_mass_mean"),
        "teacher_top1_mass": token_metrics.get("teacher_top1_mass_mean"),
        "probe_found": bool(object_probe),
        "mention_metrics_found": bool(token_metrics),
    }


def summarize_bucket(rows):
    counter = Counter(row["status"] for row in rows)
    label_counter = Counter(row["label"] for row in rows)
    return {
        "count": len(rows),
        "removed": counter.get("removed", 0),
        "kept": counter.get("kept", 0),
        "removal_rate": safe_div(counter.get("removed", 0), len(rows)),
        "hallucinated": label_counter.get("hallucinated", 0),
        "correct": label_counter.get("correct", 0),
        "hallucination_rate": safe_div(label_counter.get("hallucinated", 0), len(rows)),
        "removed_hallucinated": sum(
            1 for row in rows if row["status"] == "removed" and row["label"] == "hallucinated"
        ),
        "removed_correct": sum(
            1 for row in rows if row["status"] == "removed" and row["label"] == "correct"
        ),
        "student_entropy_mean": safe_mean(row.get("student_entropy") for row in rows),
        "teacher_entropy_mean": safe_mean(row.get("teacher_entropy") for row in rows),
        "max_entropy_mean": safe_mean(row.get("max_entropy") for row in rows),
        "teacher_minus_student_logprob_mean": safe_mean(
            row.get("teacher_minus_student_logprob") for row in rows
        ),
    }


def group_summary(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    out = {}
    total_removed = sum(1 for row in rows if row["status"] == "removed")
    for value, bucket_rows in sorted(grouped.items()):
        current = summarize_bucket(bucket_rows)
        current["share_of_removed"] = safe_div(current["removed"], total_removed)
        out[value] = current
    return out


def add_threshold_flags(rows, thresholds):
    for row in rows:
        max_entropy = row.get("max_entropy")
        row["high_entropy_p75"] = (
            bool_finite(max_entropy)
            and bool_finite(thresholds.get("max_entropy_p75"))
            and float(max_entropy) >= float(thresholds["max_entropy_p75"])
        )
        row["high_entropy_p90"] = (
            bool_finite(max_entropy)
            and bool_finite(thresholds.get("max_entropy_p90"))
            and float(max_entropy) >= float(thresholds["max_entropy_p90"])
        )
        delta = row.get("teacher_minus_student_logprob")
        row["teacher_lower_logprob"] = bool_finite(delta) and float(delta) < 0
        row["strong_teacher_lower_logprob_p25"] = (
            bool_finite(delta)
            and bool_finite(thresholds.get("teacher_minus_student_logprob_p25"))
            and float(delta) <= float(thresholds["teacher_minus_student_logprob_p25"])
        )


def analyze(args):
    baseline_records = load_jsonl_records(args.baseline_results)
    model_records = load_jsonl_records(args.model_results)
    test_index = load_test_index(args.test_json)
    probe_index, probe_meta = load_probe_index(
        args.baseline_probe_jsonl_glob,
        args.support_margin,
        args.confident_entropy_max,
    )

    mscoco_objects, inverse_synonym_dict = parse_official_synonyms()
    double_word_dict = build_double_word_dict()
    common_ids = sorted(set(baseline_records) & set(model_records))

    baseline_rows = []
    added_rows = []
    sample_rows = []
    for image_id in common_ids:
        base_record = baseline_records[image_id]
        model_record = model_records[image_id]
        gt_objects = get_gt_objects(base_record, test_index) or get_gt_objects(model_record, test_index)
        base_caption = infer_caption(base_record)
        model_caption = infer_caption(model_record)
        base_objects = extract_objects(base_caption, mscoco_objects, inverse_synonym_dict, double_word_dict)
        model_objects = extract_objects(model_caption, mscoco_objects, inverse_synonym_dict, double_word_dict)

        removed = sorted(base_objects - model_objects)
        kept = sorted(base_objects & model_objects)
        added = sorted(model_objects - base_objects)

        for obj in sorted(base_objects):
            label = "correct" if obj in gt_objects else "hallucinated"
            row = base_object_feature(image_id, obj, label, probe_index)
            row["status"] = "removed" if obj in removed else "kept"
            baseline_rows.append(row)

        for obj in added:
            label = "correct" if obj in gt_objects else "hallucinated"
            added_rows.append({
                "image_id": image_id,
                "object": obj,
                "label": label,
                "status": "added",
            })

        sample_rows.append({
            "image_id": image_id,
            "baseline_mentioned": len(base_objects),
            "model_mentioned": len(model_objects),
            "removed": len(removed),
            "added": len(added),
            "kept": len(kept),
            "removed_objects": removed,
            "added_objects": added,
        })

    thresholds = {
        "student_entropy_p75": percentile((r.get("student_entropy") for r in baseline_rows), 0.75),
        "student_entropy_p90": percentile((r.get("student_entropy") for r in baseline_rows), 0.90),
        "teacher_entropy_p75": percentile((r.get("teacher_entropy") for r in baseline_rows), 0.75),
        "teacher_entropy_p90": percentile((r.get("teacher_entropy") for r in baseline_rows), 0.90),
        "max_entropy_p75": percentile((r.get("max_entropy") for r in baseline_rows), 0.75),
        "max_entropy_p90": percentile((r.get("max_entropy") for r in baseline_rows), 0.90),
        "teacher_minus_student_logprob_p25": percentile(
            (r.get("teacher_minus_student_logprob") for r in baseline_rows), 0.25
        ),
        "teacher_minus_student_logprob_p10": percentile(
            (r.get("teacher_minus_student_logprob") for r in baseline_rows), 0.10
        ),
    }
    add_threshold_flags(baseline_rows, thresholds)

    removed_rows = [row for row in baseline_rows if row["status"] == "removed"]
    kept_rows = [row for row in baseline_rows if row["status"] == "kept"]

    label_totals = group_summary(baseline_rows, "label")
    feature_keys = [
        "reject_quadrant",
        "state_pair",
        "student_state",
        "teacher_state",
        "low_cross_resolution_support",
        "high_entropy_p75",
        "high_entropy_p90",
        "teacher_lower_logprob",
        "strong_teacher_lower_logprob_p25",
    ]

    feature_summaries = {key: group_summary(baseline_rows, key) for key in feature_keys}
    added_label_counts = Counter(row["label"] for row in added_rows)

    summary = {
        "config": {
            "baseline_results": args.baseline_results,
            "model_results": args.model_results,
            "baseline_probe_jsonl_glob": args.baseline_probe_jsonl_glob,
            "test_json": args.test_json,
            "support_margin": args.support_margin,
            "confident_entropy_max": args.confident_entropy_max,
        },
        "probe_meta": probe_meta,
        "totals": {
            "common_samples": len(common_ids),
            "baseline_mentioned_objects": len(baseline_rows),
            "model_added_objects": len(added_rows),
            "removed_objects": len(removed_rows),
            "kept_objects": len(kept_rows),
            "samples_with_removal": sum(1 for row in sample_rows if row["removed"] > 0),
            "samples_with_addition": sum(1 for row in sample_rows if row["added"] > 0),
            "probe_found_rate": safe_div(sum(1 for row in baseline_rows if row["probe_found"]), len(baseline_rows)),
            "mention_metrics_found_rate": safe_div(
                sum(1 for row in baseline_rows if row["mention_metrics_found"]), len(baseline_rows)
            ),
        },
        "thresholds": thresholds,
        "baseline_objects_by_label": label_totals,
        "added_objects_by_label": {
            label: {
                "count": count,
                "rate": safe_div(count, len(added_rows)),
            }
            for label, count in sorted(added_label_counts.items())
        },
        "removed_objects_summary": summarize_bucket(removed_rows),
        "kept_objects_summary": summarize_bucket(kept_rows),
        "feature_summaries": feature_summaries,
    }

    return summary, baseline_rows, added_rows, sample_rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def compact_feature_rows(feature_summary, limit=None):
    items = list(feature_summary.items())
    items.sort(key=lambda kv: (kv[1].get("removed", 0), kv[1].get("count", 0)), reverse=True)
    if limit:
        items = items[:limit]
    rows = []
    for key, value in items:
        rows.append([
            key,
            value["count"],
            value["removed"],
            fmt(value["removal_rate"]),
            fmt(value["share_of_removed"]),
            value["hallucinated"],
            fmt(value["hallucination_rate"]),
            value["removed_hallucinated"],
            value["removed_correct"],
        ])
    return rows


def write_markdown(path, summary, top_state_pairs):
    totals = summary["totals"]
    removed = summary["removed_objects_summary"]
    kept = summary["kept_objects_summary"]
    label = summary["baseline_objects_by_label"]
    features = summary["feature_summaries"]

    lines = [
        "# RKL Mention Reduction Diagnostic",
        "",
        "This analysis asks whether object mentions removed by RKL mainly come from "
        "`both_reject`, high-entropy, or low cross-resolution support regions.",
        "",
        "## Inputs",
        "",
        f"- baseline_results: `{summary['config']['baseline_results']}`",
        f"- model_results: `{summary['config']['model_results']}`",
        f"- baseline_probe_jsonl_glob: `{summary['config']['baseline_probe_jsonl_glob']}`",
        f"- common samples: {totals['common_samples']}",
        f"- probe_found_rate: {fmt(totals['probe_found_rate'])}",
        f"- mention_metrics_found_rate: {fmt(totals['mention_metrics_found_rate'])}",
        "",
        "## Overall Diff",
        "",
        table(
            ["metric", "value"],
            [
                ["baseline mentioned objects", totals["baseline_mentioned_objects"]],
                ["RKL-added objects", totals["model_added_objects"]],
                ["removed baseline objects", totals["removed_objects"]],
                ["kept baseline objects", totals["kept_objects"]],
                ["samples with removal", totals["samples_with_removal"]],
                ["samples with addition", totals["samples_with_addition"]],
            ],
        ),
        "",
        "## Removed vs Kept",
        "",
        table(
            [
                "group",
                "count",
                "hallucinated",
                "halluc rate",
                "student entropy",
                "teacher entropy",
                "max entropy",
                "teacher-student logp",
            ],
            [
                [
                    "removed",
                    removed["count"],
                    removed["hallucinated"],
                    fmt(removed["hallucination_rate"]),
                    fmt(removed["student_entropy_mean"]),
                    fmt(removed["teacher_entropy_mean"]),
                    fmt(removed["max_entropy_mean"]),
                    fmt(removed["teacher_minus_student_logprob_mean"]),
                ],
                [
                    "kept",
                    kept["count"],
                    kept["hallucinated"],
                    fmt(kept["hallucination_rate"]),
                    fmt(kept["student_entropy_mean"]),
                    fmt(kept["teacher_entropy_mean"]),
                    fmt(kept["max_entropy_mean"]),
                    fmt(kept["teacher_minus_student_logprob_mean"]),
                ],
            ],
        ),
        "",
        "## Removal By Label",
        "",
        table(
            ["label", "baseline count", "removed", "removal rate", "share of removed"],
            [
                [
                    key,
                    value["count"],
                    value["removed"],
                    fmt(value["removal_rate"]),
                    fmt(value["share_of_removed"]),
                ]
                for key, value in sorted(label.items())
            ],
        ),
        "",
        "## Removal By Reject Quadrant",
        "",
        table(
            [
                "quadrant",
                "baseline count",
                "removed",
                "removal rate",
                "share of removed",
                "baseline halluc",
                "baseline halluc rate",
                "removed halluc",
                "removed correct",
            ],
            compact_feature_rows(features["reject_quadrant"]),
        ),
        "",
        "## Removal By Student/Teacher State Pair",
        "",
        table(
            [
                "state pair",
                "baseline count",
                "removed",
                "removal rate",
                "share of removed",
                "baseline halluc",
                "baseline halluc rate",
                "removed halluc",
                "removed correct",
            ],
            compact_feature_rows(features["state_pair"], top_state_pairs),
        ),
        "",
        "## Proxy Feature Checks",
        "",
        table(
            [
                "feature",
                "value",
                "baseline count",
                "removed",
                "removal rate",
                "share of removed",
                "baseline halluc rate",
            ],
            [
                [
                    "low_cross_resolution_support",
                    key,
                    value["count"],
                    value["removed"],
                    fmt(value["removal_rate"]),
                    fmt(value["share_of_removed"]),
                    fmt(value["hallucination_rate"]),
                ]
                for key, value in sorted(features["low_cross_resolution_support"].items())
            ]
            + [
                [
                    "high_entropy_p75",
                    key,
                    value["count"],
                    value["removed"],
                    fmt(value["removal_rate"]),
                    fmt(value["share_of_removed"]),
                    fmt(value["hallucination_rate"]),
                ]
                for key, value in sorted(features["high_entropy_p75"].items())
            ]
            + [
                [
                    "teacher_lower_logprob",
                    key,
                    value["count"],
                    value["removed"],
                    fmt(value["removal_rate"]),
                    fmt(value["share_of_removed"]),
                    fmt(value["hallucination_rate"]),
                ]
                for key, value in sorted(features["teacher_lower_logprob"].items())
            ],
        ),
        "",
        "## How To Read",
        "",
        "- Evidence for targeted hallucination suppression: removed objects should have a "
        "higher hallucination rate than kept objects, and removal rate should be higher "
        "in `both_reject`, `low_cross_resolution_support=True`, and `high_entropy_p75=True`.",
        "- Evidence for blind conservatism: removal rate is similar for correct and "
        "hallucinated objects, or most removed objects come from `student=support|teacher=support`.",
        "- Evidence against teacher-only veto: `teacher_only_reject` contributes little "
        "or has much lower hallucination rate than `both_reject`.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.model_results)),
        "mention_reduction_analysis",
    )
    os.makedirs(output_dir, exist_ok=True)

    summary, baseline_rows, added_rows, sample_rows = analyze(args)

    summary_path = os.path.join(output_dir, "mention_reduction_summary.json")
    removed_path = os.path.join(output_dir, "baseline_object_status.jsonl")
    added_path = os.path.join(output_dir, "model_added_objects.jsonl")
    sample_path = os.path.join(output_dir, "sample_mention_diff.jsonl")
    md_path = os.path.join(output_dir, "mention_reduction_summary.md")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_jsonl(removed_path, baseline_rows)
    write_jsonl(added_path, added_rows)
    write_jsonl(sample_path, sample_rows)
    write_markdown(md_path, summary, args.top_state_pairs)

    print(f"Saved summary JSON: {summary_path}")
    print(f"Saved markdown: {md_path}")
    print(f"Saved baseline object status JSONL: {removed_path}")
    print(f"Saved model-added objects JSONL: {added_path}")
    print(f"Saved sample diff JSONL: {sample_path}")


if __name__ == "__main__":
    main()
