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
        --experiment-name Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a0.5-ema-e1 \
        --fetch-from-oss \
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
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RES_OPD_ROOT = os.path.dirname(EVAL_DIR)
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


def get_oss_name(experiment_name):
    suffix = experiment_name
    prefix = "Res-OPD-Qwen3VL-2B-Instruct-"
    if suffix.startswith(prefix):
        suffix = suffix[len(prefix):]

    epoch_tag = ""
    parts = suffix.rsplit("-", 1)
    if len(parts) == 2 and parts[1].startswith("e") and parts[1][1:].isdigit():
        suffix, epoch_tag = parts[0], f"-{parts[1]}"

    return f"ResOPD_{suffix.replace('-', '_')}{epoch_tag}"


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


def trace_dir_has_records(trace_path):
    if not trace_path or not os.path.exists(trace_path):
        return False
    return any(True for _ in iter_trace_files(trace_path))


def is_trace_record(record):
    return (
        isinstance(record, dict)
        and "global_step" in record
        and "token_records" in record
        and "summary" in record
    )


def load_trace_records(trace_path, max_records=0):
    records = []
    skipped_non_trace = 0
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
                if not is_trace_record(record):
                    skipped_non_trace += 1
                    continue
                record["_trace_file"] = path
                records.append(record)
                if max_records and len(records) >= max_records:
                    return records, skipped_non_trace
    return records, skipped_non_trace


def fetch_traces_from_oss(oss_base, oss_name, trace_dir):
    oss_trace_path = f"{oss_base.rstrip('/')}/{oss_name}/training_artifacts/traces"
    if shutil.which("ossutil") is None:
        raise SystemExit(
            "ossutil not found in PATH. Install/configure ossutil, or fetch traces manually "
            f"from {oss_trace_path}/"
        )

    os.makedirs(trace_dir, exist_ok=True)
    cmd = [
        "ossutil",
        "cp",
        "-r",
        f"{oss_trace_path.rstrip('/')}/",
        f"{trace_dir.rstrip('/')}/",
        "-f",
    ]
    print(f"Fetching traces from OSS: {oss_trace_path}/ -> {trace_dir}/")
    subprocess.run(cmd, check=True)
    return oss_trace_path


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


def summarize_trace_coverage(records):
    by_step = defaultdict(list)
    by_rank = Counter()
    by_file = Counter()
    for record in records:
        step = record.get("global_step")
        by_step[step].append(record)
        if record.get("rank") is not None:
            by_rank[str(record.get("rank"))] += 1
        trace_file = record.get("_trace_file")
        if trace_file:
            by_file[trace_file] += 1

    def step_sort_key(step):
        if step is None:
            return (-1, "")
        try:
            return (0, int(step))
        except (TypeError, ValueError):
            return (1, str(step))

    step_summary = {}
    for step, step_records in sorted(by_step.items(), key=lambda item: step_sort_key(item[0])):
        summaries = [record.get("summary", {}) or {} for record in step_records]
        step_summary[str(step)] = {
            "num_records": len(step_records),
            "num_tokens": sum((record.get("summary", {}) or {}).get("num_tokens", 0) for record in step_records),
            "num_unique_images": len({get_image_id(record) for record in step_records if get_image_id(record) is not None}),
            "student_selected_logprob_mean": safe_mean(
                summary.get("student_selected_logprob_mean") for summary in summaries
            ),
            "teacher_selected_logprob_mean": safe_mean(
                summary.get("teacher_selected_logprob_mean") for summary in summaries
            ),
            "teacher_minus_student_selected_logprob_mean": safe_mean(
                (
                    summary.get("teacher_selected_logprob_mean") - summary.get("student_selected_logprob_mean")
                    if summary.get("teacher_selected_logprob_mean") is not None
                    and summary.get("student_selected_logprob_mean") is not None
                    else None
                )
                for summary in summaries
            ),
            "student_entropy_mean": safe_mean(summary.get("student_entropy_mean") for summary in summaries),
            "teacher_entropy_mean": safe_mean(summary.get("teacher_entropy_mean") for summary in summaries),
            "top1_match_frac": safe_mean(summary.get("top1_match_frac") for summary in summaries),
            "topk_overlap_ratio_mean": safe_mean(summary.get("topk_overlap_ratio_mean") for summary in summaries),
        }

    numeric_steps = []
    non_numeric_steps = []
    for step in by_step:
        if step is None:
            continue
        try:
            numeric_steps.append(int(step))
        except (TypeError, ValueError):
            non_numeric_steps.append(str(step))

    return {
        "global_steps": sorted(numeric_steps) + sorted(non_numeric_steps),
        "num_global_steps": len(numeric_steps) + len(non_numeric_steps),
        "min_global_step": min(numeric_steps, default=None),
        "max_global_step": max(numeric_steps, default=None),
        "records_by_rank": dict(
            sorted(
                by_rank.items(),
                key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0]),
            )
        ),
        "records_by_file": dict(sorted(by_file.items())),
        "step_summary": step_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze OPD training token traces")
    parser.add_argument(
        "--trace-dir",
        help=(
            "Trace directory or single JSONL trace file. If omitted with --experiment-name, "
            "defaults to res-opd/traces/<experiment-name>."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        help="Experiment name used to derive local trace dir and OSS path.",
    )
    parser.add_argument(
        "--oss-name",
        help="Explicit OSS experiment name. If omitted, inferred from --experiment-name.",
    )
    parser.add_argument(
        "--oss-base",
        default=os.environ.get("OSS_BASE", "oss://industry-algo/yanlin/ckpt/OPD/v4"),
        help="OSS root containing <oss-name>/training_artifacts/traces.",
    )
    parser.add_argument(
        "--fetch-from-oss",
        action="store_true",
        help="Fetch traces from OSS before analysis.",
    )
    parser.add_argument(
        "--no-auto-fetch",
        action="store_true",
        help="Do not auto-fetch from OSS when local trace files are missing.",
    )
    parser.add_argument("--case-analysis", help="Optional all_cases_sorted.json for outcome grouping")
    parser.add_argument("--output-json", required=True, help="Where to save summary JSON")
    parser.add_argument("--max-records", type=int, default=0, help="Debug cap; 0 = all")
    args = parser.parse_args()

    trace_dir = args.trace_dir
    if not trace_dir and args.experiment_name:
        trace_dir = os.path.join(RES_OPD_ROOT, "traces", args.experiment_name)
    if not trace_dir:
        parser.error("--trace-dir is required unless --experiment-name is provided")

    oss_name = args.oss_name
    if not oss_name and args.experiment_name:
        oss_name = get_oss_name(args.experiment_name)

    fetched_from_oss = False
    oss_trace_path = None
    should_auto_fetch = (
        args.experiment_name
        and oss_name
        and not args.no_auto_fetch
        and not trace_dir_has_records(trace_dir)
    )
    if args.fetch_from_oss or should_auto_fetch:
        if not oss_name:
            parser.error("--fetch-from-oss requires --experiment-name or --oss-name")
        oss_trace_path = fetch_traces_from_oss(args.oss_base, oss_name, trace_dir)
        fetched_from_oss = True

    records, skipped_non_trace_records = load_trace_records(trace_dir, max_records=args.max_records)
    if not records:
        hint = ""
        if args.experiment_name and oss_name:
            hint = (
                f"\nTried local trace dir: {trace_dir}\n"
                f"Expected OSS trace path: {args.oss_base.rstrip('/')}/{oss_name}/training_artifacts/traces/"
            )
        raise SystemExit(f"No trace records found under {trace_dir}.{hint}")
    case_groups = load_case_groups(args.case_analysis)
    tokens = flatten_token_records(records, case_groups)

    image_ids = [get_image_id(record) for record in records]
    matched_records = sum(1 for image_id in image_ids if image_id in case_groups)
    trace_coverage = summarize_trace_coverage(records)
    output = {
        "trace_dir": trace_dir,
        "experiment_name": args.experiment_name,
        "oss_name": oss_name,
        "oss_base": args.oss_base,
        "oss_trace_path": oss_trace_path
        or (f"{args.oss_base.rstrip('/')}/{oss_name}/training_artifacts/traces" if oss_name else None),
        "fetched_from_oss": fetched_from_oss,
        "case_analysis": args.case_analysis,
        "num_records": len(records),
        "skipped_non_trace_records": skipped_non_trace_records,
        "num_tokens": len(tokens),
        "num_unique_images": len({image_id for image_id in image_ids if image_id is not None}),
        "case_matched_records": matched_records,
        "case_matched_record_rate": safe_rate(matched_records, len(records)),
        "trace_coverage": trace_coverage,
        "global_steps": trace_coverage["global_steps"],
        "record_group_summary": summarize_records(records, case_groups),
        "token_group_summary": summarize_tokens(tokens),
        "object_mention_summary": summarize_object_mentions(records, case_groups),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Loaded trace records: {len(records)}")
    print(f"Loaded tokens: {len(tokens)}")
    print(
        "Trace steps: "
        f"{trace_coverage['min_global_step']}..{trace_coverage['max_global_step']} "
        f"({trace_coverage['num_global_steps']} unique steps)"
    )
    if fetched_from_oss:
        print(f"Fetched traces from OSS: {output['oss_trace_path']}/")
    if skipped_non_trace_records:
        print(f"Skipped non-trace JSON records: {skipped_non_trace_records}")
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
