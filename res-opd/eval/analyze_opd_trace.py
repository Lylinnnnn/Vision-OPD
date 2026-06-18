"""
Analyze OPD training-time student/teacher token traces.

This script consumes JSONL files emitted by
actor.self_distillation.trace_enabled=True in verl/workers/actor/dp_actor.py.
It is meant to answer whether the low-resolution teacher selectively pushes
down uncertain/hallucinated-looking tokens, without rerunning vLLM inference.

Examples:
    python res-opd/eval/analyze_opd_trace.py \
        --trace-dir res-opd/traces/Res-OPD-... \
        --output-json /tmp/opd_trace_summary.json

    python res-opd/eval/analyze_opd_trace.py \
        --trace-dir res-opd/traces/Res-OPD-... \
        --case-analysis res-opd/eval_results/v2/case_analysis/sr1.0-tr0.75_vs_baseline/all_cases_sorted.json \
        --output-json /tmp/opd_trace_summary.with_cases.json
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
from robust_chair_analysis import (  # noqa: E402
    build_double_word_dict,
    caption_to_words,
    parse_official_synonyms,
)


def safe_mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def iter_trace_files(trace_path):
    if os.path.isfile(trace_path):
        yield trace_path
        return
    patterns = [
        os.path.join(trace_path, "*.jsonl"),
        os.path.join(trace_path, "**", "*.jsonl"),
    ]
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            if path not in seen:
                seen.add(path)
                yield path


def load_trace_records(trace_path, max_records=0):
    records = []
    for path in iter_trace_files(trace_path):
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"WARNING: skip malformed JSON {path}:{line_no}: {exc}", file=sys.stderr)
                    continue
                record["_trace_file"] = path
                records.append(record)
                if max_records and len(records) >= max_records:
                    return records
    return records


def load_case_groups(case_analysis_path):
    if not case_analysis_path:
        return {}
    with open(case_analysis_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    groups = {}
    for case in cases:
        image_id = case.get("image_id")
        if image_id is None:
            continue
        diag = case.get("case_diagnostics", {})
        if diag.get("strict_selective_suppression"):
            group = "strict_selective_suppression"
        elif diag.get("harmful_gt_suppression"):
            group = "harmful_gt_suppression"
        elif diag.get("gt_hallucination_tradeoff"):
            group = "gt_hallucination_tradeoff"
        elif diag.get("coverage_improved_no_new_hallucination"):
            group = "coverage_improved_no_new_hallucination"
        else:
            group = case.get("distill_recommendation") or case.get("behavior_pattern") or "case_other"
        groups[int(image_id)] = {
            "group": group,
            "behavior_pattern": case.get("behavior_pattern"),
            "distill_recommendation": case.get("distill_recommendation"),
            "gt_objects": case.get("gt_objects", []),
            "model_mentioned_objects": case.get("model_mentioned_objects", []),
            "baseline_mentioned_objects": case.get("baseline_mentioned_objects", []),
        }
    return groups


def get_image_id(record):
    metadata = record.get("metadata") or {}
    image_id = metadata.get("image_id")
    if image_id is None:
        extra = metadata.get("extra_info")
        if isinstance(extra, dict):
            image_id = extra.get("image_id")
    try:
        return int(image_id)
    except (TypeError, ValueError):
        return None


def token_delta(token_record):
    student = token_record.get("student_selected_logprob")
    teacher = token_record.get("teacher_selected_logprob")
    if student is None or teacher is None:
        return None
    return teacher - student


def flatten_token_records(records, case_groups):
    tokens = []
    for record in records:
        image_id = get_image_id(record)
        case_info = case_groups.get(image_id, {})
        group = case_info.get("group", "unmatched_or_no_case")
        for token in record.get("token_records", []) or []:
            item = dict(token)
            item["_image_id"] = image_id
            item["_group"] = group
            item["_global_step"] = record.get("global_step")
            item["_trace_file"] = record.get("_trace_file")
            item["_teacher_minus_student_selected_logprob"] = token_delta(token)
            tokens.append(item)
    return tokens


def summarize_records(records, case_groups):
    by_group = defaultdict(list)
    for record in records:
        image_id = get_image_id(record)
        group = case_groups.get(image_id, {}).get("group", "unmatched_or_no_case")
        by_group[group].append(record)

    summary = {}
    for group, group_records in sorted(by_group.items()):
        record_summaries = [r.get("summary", {}) or {} for r in group_records]
        summary[group] = {
            "num_records": len(group_records),
            "num_unique_images": len({get_image_id(r) for r in group_records if get_image_id(r) is not None}),
            "student_selected_logprob_mean": safe_mean(
                s.get("student_selected_logprob_mean") for s in record_summaries
            ),
            "teacher_selected_logprob_mean": safe_mean(
                s.get("teacher_selected_logprob_mean") for s in record_summaries
            ),
            "teacher_minus_student_selected_logprob_mean": safe_mean(
                (
                    s.get("teacher_selected_logprob_mean") - s.get("student_selected_logprob_mean")
                    if s.get("teacher_selected_logprob_mean") is not None
                    and s.get("student_selected_logprob_mean") is not None
                    else None
                )
                for s in record_summaries
            ),
            "student_entropy_mean": safe_mean(s.get("student_entropy_mean") for s in record_summaries),
            "teacher_entropy_mean": safe_mean(s.get("teacher_entropy_mean") for s in record_summaries),
            "student_top1_top2_margin_mean": safe_mean(
                s.get("student_top1_top2_margin_mean") for s in record_summaries
            ),
            "teacher_top1_top2_margin_mean": safe_mean(
                s.get("teacher_top1_top2_margin_mean") for s in record_summaries
            ),
            "topk_overlap_ratio_mean": safe_mean(s.get("topk_overlap_ratio_mean") for s in record_summaries),
            "topk_jaccard_mean": safe_mean(s.get("topk_jaccard_mean") for s in record_summaries),
            "top1_match_frac": safe_mean(s.get("top1_match_frac") for s in record_summaries),
        }
    return summary


def summarize_tokens(tokens):
    by_group = defaultdict(list)
    for token in tokens:
        by_group[token["_group"]].append(token)

    summary = {}
    for group, group_tokens in sorted(by_group.items()):
        deltas = [t.get("_teacher_minus_student_selected_logprob") for t in group_tokens]
        top1_matches = [t.get("top1_match") for t in group_tokens if t.get("top1_match") is not None]
        teacher_lower = [d for d in deltas if d is not None and d < 0]
        token_text_counter = Counter(t.get("token_text", "") for t in group_tokens)
        summary[group] = {
            "num_tokens": len(group_tokens),
            "teacher_minus_student_selected_logprob_mean": safe_mean(deltas),
            "teacher_selected_logprob_lt_student_frac": safe_rate(len(teacher_lower), len([d for d in deltas if d is not None])),
            "student_entropy_mean": safe_mean(t.get("student_entropy") for t in group_tokens),
            "teacher_entropy_mean": safe_mean(t.get("teacher_entropy") for t in group_tokens),
            "student_top1_top2_margin_mean": safe_mean(t.get("student_top1_top2_margin") for t in group_tokens),
            "teacher_top1_top2_margin_mean": safe_mean(t.get("teacher_top1_top2_margin") for t in group_tokens),
            "topk_overlap_ratio_mean": safe_mean(t.get("topk_overlap_ratio") for t in group_tokens),
            "topk_jaccard_mean": safe_mean(t.get("topk_jaccard") for t in group_tokens),
            "top1_match_frac": safe_mean(1.0 if x else 0.0 for x in top1_matches),
            "selected_token_rank_in_teacher_topk_mean": safe_mean(
                t.get("selected_token_rank_in_teacher_topk") for t in group_tokens
            ),
            "top_tokens": dict(token_text_counter.most_common(30)),
        }
    return summary


def summarize_object_mentions(records, case_groups):
    mscoco_objects, inverse_synonym_dict = parse_official_synonyms()
    double_word_dict = build_double_word_dict()

    by_group = defaultdict(list)
    object_counter = defaultdict(Counter)
    for record in records:
        image_id = get_image_id(record)
        case_info = case_groups.get(image_id, {})
        group = case_info.get("group", "unmatched_or_no_case")
        response_text = record.get("response_text", "")
        _, mentioned = caption_to_words(
            response_text, mscoco_objects, inverse_synonym_dict, double_word_dict
        )
        mentioned = sorted(set(mentioned))
        by_group[group].append(mentioned)
        for obj in mentioned:
            object_counter[group][obj] += 1

    return {
        group: {
            "records_with_any_coco_object": sum(1 for objs in mentions if objs),
            "mean_unique_coco_objects": safe_mean(len(objs) for objs in mentions),
            "top_mentioned_objects": dict(object_counter[group].most_common(30)),
        }
        for group, mentions in sorted(by_group.items())
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze OPD training token traces")
    parser.add_argument("--trace-dir", required=True, help="Trace directory or single JSONL trace file")
    parser.add_argument("--case-analysis", help="Optional all_cases_sorted.json for outcome grouping")
    parser.add_argument("--output-json", required=True, help="Where to save summary JSON")
    parser.add_argument("--max-records", type=int, default=0, help="Debug cap; 0 = all")
    args = parser.parse_args()

    records = load_trace_records(args.trace_dir, max_records=args.max_records)
    if not records:
        raise SystemExit(f"No trace records found under {args.trace_dir}")
    case_groups = load_case_groups(args.case_analysis)
    tokens = flatten_token_records(records, case_groups)

    image_ids = [get_image_id(record) for record in records]
    matched_records = sum(1 for image_id in image_ids if image_id in case_groups)
    output = {
        "trace_dir": args.trace_dir,
        "case_analysis": args.case_analysis,
        "num_records": len(records),
        "num_tokens": len(tokens),
        "num_unique_images": len({image_id for image_id in image_ids if image_id is not None}),
        "case_matched_records": matched_records,
        "case_matched_record_rate": safe_rate(matched_records, len(records)),
        "global_steps": sorted({r.get("global_step") for r in records if r.get("global_step") is not None}),
        "record_group_summary": summarize_records(records, case_groups),
        "token_group_summary": summarize_tokens(tokens),
        "object_mention_summary": summarize_object_mentions(records, case_groups),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Loaded trace records: {len(records)}")
    print(f"Loaded tokens: {len(tokens)}")
    print(f"Matched records to case analysis: {matched_records}/{len(records)}")
    for group, stats in output["token_group_summary"].items():
        delta = stats["teacher_minus_student_selected_logprob_mean"]
        lower = stats["teacher_selected_logprob_lt_student_frac"]
        overlap = stats["topk_overlap_ratio_mean"]
        print(
            f"{group}: tokens={stats['num_tokens']} "
            f"teacher-student logp={delta if delta is not None else 'n/a'} "
            f"teacher<student={lower:.3f} overlap={overlap if overlap is not None else 'n/a'}"
        )
    print(f"Saved summary to: {args.output_json}")


if __name__ == "__main__":
    main()
