#!/usr/bin/env python3
"""
Probe whether low-resolution views provide a selective hallucination veto signal
on fixed base-model captions.

This script does not run vLLM generation. It reads an existing base
eval_results.jsonl, forced-scores the same captions with a high-resolution
student view and several low-resolution critic views, then reuses
analyze_opd_trace.py to summarize correct vs hallucinated object signals.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from types import SimpleNamespace

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)

import analyze_opd_trace as analyzer  # noqa: E402
import score_opd_eval_trace as scorer  # noqa: E402


DEFAULT_MODEL_PATH = "/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-2B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a base-model OPD critic probe over fixed eval_results.jsonl "
            "captions for multiple low-resolution teacher ratios."
        )
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Base HF model path used for forced scoring.",
    )
    parser.add_argument("--eval-results", required=True, help="Base model eval_results.jsonl")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <eval-results dir>/base_trace_probe/",
    )
    parser.add_argument(
        "--teacher-ratios",
        default="0.25,0.5,0.75",
        help="Comma-separated original-size low-res critic ratios.",
    )
    parser.add_argument("--student-ratio", type=float, default=1.0)
    parser.add_argument("--degradation-mode", choices=["original"], default="original")
    parser.add_argument("--student-px", type=int, default=0, help="Legacy metadata only; original-ratio scoring ignores this value")
    parser.add_argument("--teacher-px", type=int, default=0, help="Legacy metadata only; original-ratio scoring ignores this value")
    parser.add_argument("--target-px", type=int, default=448, help="Legacy metadata only; original-ratio scoring ignores this value")
    parser.add_argument("--case-analysis", help="Optional all_cases_sorted.json for case grouping/GT fallback")
    parser.add_argument("--test-json", help="Optional test.json for image path/GT fallback")
    parser.add_argument("--image-root", default=scorer.DEFAULT_COCO_VAL_ROOT)
    parser.add_argument("--prompt", default=scorer.PROMPT_TEXT)
    parser.add_argument("--caption-field", default="auto")
    parser.add_argument("--trace-scope", default="base_probe")
    parser.add_argument("--checkpoint-step", type=int, default=0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=0, help="Debug cap by eval sample; 0 = all")
    parser.add_argument("--max-model-len", type=int, default=9728)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-ratio traces")
    parser.add_argument("--skip-scoring", action="store_true", help="Only analyze existing traces")
    parser.add_argument("--skip-analysis", action="store_true", help="Only write traces, do not summarize")
    parser.add_argument("--score-baseline-caption", action="store_true")
    parser.add_argument("--no-entropy", action="store_true", help="Disable full-vocabulary entropy scoring")
    parser.add_argument(
        "--entropy-bin-edges",
        default="0,0.5,1,1.5,inf",
        help="Passed to analyze_opd_trace.py.",
    )
    parser.add_argument(
        "--gate-entropy-thresholds",
        default="0.5,0.75,1,1.25,1.5,2",
        help="Passed to analyze_opd_trace.py.",
    )
    parser.add_argument(
        "--gate-logp-margins",
        default="0,0.01,0.02,0.05,0.1,0.2",
        help="Passed to analyze_opd_trace.py.",
    )
    parser.add_argument("--quadrant-support-delta-min", type=float, default=None)
    parser.add_argument("--quadrant-reject-delta-max", type=float, default=None)
    parser.add_argument("--quadrant-teacher-entropy-max", type=float, default=None)
    parser.add_argument("--quadrant-teacher-margin-min", type=float, default=None)
    parser.add_argument("--quadrant-support-rank-max", type=float, default=None)
    parser.add_argument("--quadrant-reject-rank-min", type=float, default=None)
    parser.add_argument("--quadrant-support-top1-min", type=float, default=None)
    parser.add_argument("--quadrant-reject-top1-max", type=float, default=None)
    return parser.parse_args()


def parse_ratios(raw):
    ratios = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        ratio = float(text)
        if ratio <= 0:
            raise ValueError(f"teacher ratio must be > 0, got {ratio}")
        ratios.append(ratio)
    if not ratios:
        raise ValueError("--teacher-ratios cannot be empty")
    return ratios


def format_ratio(ratio):
    text = f"{ratio:.6g}"
    if "." not in text and "e" not in text:
        text += ".0"
    return text


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def ratio_state(output_dir, ratio, overwrite):
    label = f"tr{format_ratio(ratio)}"
    ratio_dir = os.path.join(output_dir, label)
    trace_path = os.path.join(ratio_dir, "opd_eval_trace.jsonl")
    summary_json = os.path.join(ratio_dir, "opd_trace_summary.json")
    summary_md = os.path.join(ratio_dir, "opd_trace_summary.md")
    os.makedirs(ratio_dir, exist_ok=True)
    if overwrite and os.path.exists(trace_path):
        os.remove(trace_path)
    return {
        "ratio": ratio,
        "label": label,
        "dir": ratio_dir,
        "trace_path": trace_path,
        "summary_json": summary_json,
        "summary_md": summary_md,
        "done_keys": scorer.existing_trace_keys(trace_path),
        "written": 0,
        "skipped": 0,
        "failures": 0,
        "source_counter": Counter(),
    }


def make_record_args(args, teacher_ratio):
    return SimpleNamespace(
        trace_scope=args.trace_scope,
        model_path=args.model_path,
        degradation_mode=args.degradation_mode,
        student_px=args.student_px,
        teacher_px=args.teacher_px,
        target_px=args.target_px,
        student_ratio=args.student_ratio,
        teacher_ratio=teacher_ratio,
    )


def checkpoint_step_for_record(args, record, default_step):
    checkpoint_step = args.checkpoint_step if args.checkpoint_step is not None else default_step
    record_step = record.get("generation_step")
    if checkpoint_step == 0 and record_step is not None:
        checkpoint_step = record_step
    return checkpoint_step


def run_scoring(args, states):
    test_index = scorer.load_test_index(args.test_json)
    eval_records = [
        scorer.normalize_generation_record(record, test_index, args.caption_field)
        for record in scorer.load_jsonl(args.eval_results)
    ]
    case_index = scorer.load_case_index(args.case_analysis)
    checkpoint_step_default = (
        args.checkpoint_step
        if args.checkpoint_step is not None
        else scorer.parse_checkpoint_step(args.model_path) or scorer.parse_checkpoint_step(args.eval_results) or 0
    )

    model, processor, device = scorer.load_model_and_processor(args.model_path, args.torch_dtype)
    handles = {
        ratio: open(state["trace_path"], "a", encoding="utf-8")
        for ratio, state in states.items()
    }
    started = time.time()
    jobs_seen = 0
    try:
        for record, caption_source, caption in scorer.iter_caption_jobs(
            eval_records,
            case_index,
            args.max_samples,
            args.score_baseline_caption,
        ):
            jobs_seen += 1
            image_id = record.get("image_id")
            image_id_int = scorer.as_int(image_id)
            checkpoint_step = checkpoint_step_for_record(args, record, checkpoint_step_default)
            key = (scorer.as_int(checkpoint_step), image_id_int, caption_source)
            needed = [
                ratio
                for ratio, state in states.items()
                if key not in state["done_keys"]
            ]
            for ratio, state in states.items():
                if ratio not in needed:
                    state["skipped"] += 1
            if not needed:
                continue

            image_path = scorer.resolve_image_path(record, test_index, args.image_root)
            if not image_path:
                print(f"WARNING: missing image path for image_id={image_id}", file=sys.stderr)
                for ratio in needed:
                    states[ratio]["failures"] += 1
                continue

            case = case_index.get(image_id_int, {}) if image_id_int is not None else {}
            try:
                student_image = scorer.load_view_image(
                    image_path,
                    args.degradation_mode,
                    args.student_px,
                    args.target_px,
                    args.student_ratio,
                    blank_when_px_zero=False,
                )
                student = scorer.score_view(
                    model,
                    processor,
                    device,
                    student_image,
                    args.prompt,
                    caption,
                    args.topk,
                    not args.no_entropy,
                    args.max_model_len,
                )
            except Exception as exc:
                print(
                    f"WARNING: failed student scoring image_id={image_id} source={caption_source}: {exc}",
                    file=sys.stderr,
                )
                for ratio in needed:
                    states[ratio]["failures"] += 1
                continue

            for ratio in needed:
                state = states[ratio]
                try:
                    teacher_image = scorer.load_view_image(
                        image_path,
                        args.degradation_mode,
                        args.teacher_px,
                        args.target_px,
                        ratio,
                        blank_when_px_zero=True,
                    )
                    teacher = scorer.score_view(
                        model,
                        processor,
                        device,
                        teacher_image,
                        args.prompt,
                        caption,
                        args.topk,
                        not args.no_entropy,
                        args.max_model_len,
                    )
                    combined = scorer.combine_student_teacher(student, teacher)
                    out = scorer.build_trace_record(
                        make_record_args(args, ratio),
                        record,
                        caption_source,
                        caption,
                        combined,
                        case,
                        image_path,
                        checkpoint_step,
                    )
                    handles[ratio].write(json.dumps(out, ensure_ascii=False) + "\n")
                    handles[ratio].flush()
                    state["done_keys"].add(key)
                    state["written"] += 1
                    state["source_counter"][caption_source] += 1
                except Exception as exc:
                    print(
                        f"WARNING: failed teacher scoring image_id={image_id} "
                        f"source={caption_source} ratio={ratio}: {exc}",
                        file=sys.stderr,
                    )
                    state["failures"] += 1

            total_written = sum(state["written"] for state in states.values())
            if total_written and total_written % 30 == 0:
                elapsed = time.time() - started
                print(f"Scored {total_written} ratio-traces from {jobs_seen} captions ({elapsed:.0f}s)")
    finally:
        for handle in handles.values():
            handle.close()

    print("Base trace probe scoring complete:")
    for state in states.values():
        print(
            f"  {state['label']}: written={state['written']} skipped={state['skipped']} "
            f"failures={state['failures']} sources={dict(state['source_counter'])}"
        )


def run_analysis(args, state):
    quadrant_config = analyzer.build_quadrant_config(
        support_delta_min=args.quadrant_support_delta_min,
        reject_delta_max=args.quadrant_reject_delta_max,
        teacher_entropy_max=args.quadrant_teacher_entropy_max,
        teacher_margin_min=args.quadrant_teacher_margin_min,
        support_rank_max=args.quadrant_support_rank_max,
        reject_rank_min=args.quadrant_reject_rank_min,
        support_top1_min=args.quadrant_support_top1_min,
        reject_top1_max=args.quadrant_reject_top1_max,
    )
    summary = analyzer.build_trace_analysis_summary(
        state["trace_path"],
        case_analysis=args.case_analysis,
        max_records=0,
        entropy_bin_edges=args.entropy_bin_edges,
        gate_entropy_thresholds=args.gate_entropy_thresholds,
        gate_logp_margins=args.gate_logp_margins,
        quadrant_config=quadrant_config,
    )
    with open(state["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    analyzer.write_markdown_summary(summary, state["summary_md"])
    return summary


def get_type_stats(summary, object_type):
    return (
        summary.get("object_trace_summary", {})
        .get("mention_type_summary", {})
        .get(object_type, {})
    )


def extract_row(state, summary):
    object_trace = summary.get("object_trace_summary", {})
    correct = get_type_stats(summary, "correct_object")
    hallucinated = get_type_stats(summary, "hallucinated_object")
    signal = object_trace.get("correct_vs_hallucinated_signal", {})
    contrast = signal.get("hallucinated_minus_correct", {})
    dist = (
        object_trace.get("logp_delta_distribution", {})
        .get("teacher_minus_student_selected_logprob_mean", {})
    )
    dist_contrast = dist.get("hallucinated_minus_correct", {})
    best_gate = (object_trace.get("gate_sweep", {}).get("top_by_f1") or [{}])[0]
    quadrants = object_trace.get("lowres_quadrant_summary", {})
    quadrant_interpretation = quadrants.get("interpretation", {})
    quadrant_buckets = quadrants.get("bucket_summary", {})
    support_bucket = quadrant_buckets.get("lowres_confident_support", {})
    reject_bucket = quadrant_buckets.get("lowres_confident_reject", {})
    uncertain_bucket = quadrant_buckets.get("lowres_uncertain", {})
    agreement = object_trace.get("agreement_bucket_summary", {})
    agreement_buckets = agreement.get("bucket_summary", {})
    agreement_support = agreement_buckets.get("support", {})
    agreement_strong = agreement_buckets.get("strong_disagree", {})
    return {
        "teacher_ratio": state["ratio"],
        "label": state["label"],
        "trace_path": state["trace_path"],
        "summary_json": state["summary_json"],
        "summary_md": state["summary_md"],
        "num_records": summary.get("num_records"),
        "num_unique_images": summary.get("num_unique_images"),
        "correct_mentions": correct.get("num_mentions", 0),
        "hallucinated_mentions": hallucinated.get("num_mentions", 0),
        "correct_logp_delta_mean": correct.get("teacher_minus_student_selected_logprob_mean"),
        "hallucinated_logp_delta_mean": hallucinated.get("teacher_minus_student_selected_logprob_mean"),
        "logp_gap_hallucinated_minus_correct": contrast.get(
            "teacher_minus_student_selected_logprob_mean_diff"
        ),
        "frac_delta_lt_neg0p1_gap": dist_contrast.get("frac_lt_neg0p1_diff"),
        "frac_delta_lt_neg0p05_gap": dist_contrast.get("frac_lt_neg0p05_diff"),
        "correct_teacher_lt_student_frac": correct.get("teacher_selected_logprob_lt_student_frac"),
        "hallucinated_teacher_lt_student_frac": hallucinated.get("teacher_selected_logprob_lt_student_frac"),
        "teacher_lt_student_gap": contrast.get("teacher_selected_logprob_lt_student_frac_diff"),
        "correct_student_entropy": correct.get("student_entropy_mean"),
        "hallucinated_student_entropy": hallucinated.get("student_entropy_mean"),
        "best_gate_entropy_gt": best_gate.get("student_entropy_gt"),
        "best_gate_logp_lt": best_gate.get("teacher_minus_student_logp_lt"),
        "best_gate_precision": best_gate.get("hallucination_precision"),
        "best_gate_recall": best_gate.get("hallucination_recall"),
        "best_gate_correct_fpr": best_gate.get("correct_false_positive_rate"),
        "best_gate_f1": best_gate.get("f1"),
        "best_gate_precision_lift": best_gate.get("precision_lift_vs_base"),
        "lowres_support_labeled": support_bucket.get("num_labeled_mentions", 0),
        "lowres_support_hallucination_rate": support_bucket.get("hallucination_rate"),
        "lowres_support_correct_capture_rate": support_bucket.get("correct_capture_rate"),
        "lowres_reject_labeled": reject_bucket.get("num_labeled_mentions", 0),
        "lowres_reject_hallucination_rate": reject_bucket.get("hallucination_rate"),
        "lowres_reject_precision_lift": reject_bucket.get("precision_lift_vs_base"),
        "lowres_reject_hallucinated_recall": reject_bucket.get("hallucinated_recall"),
        "lowres_reject_correct_fpr": reject_bucket.get("correct_false_positive_rate"),
        "lowres_uncertain_labeled": uncertain_bucket.get("num_labeled_mentions", 0),
        "lowres_uncertain_hallucination_rate": uncertain_bucket.get("hallucination_rate"),
        "lowres_quadrant_base_hallucination_rate": quadrants.get("base_hallucination_rate"),
        "lowres_quadrant_support_correct_enrichment": quadrant_interpretation.get(
            "support_bucket_correct_enrichment"
        ),
        "agreement_q70_delta_threshold": agreement.get("config", {}).get("q_low_delta_threshold"),
        "agreement_q90_delta_threshold": agreement.get("config", {}).get("q_high_delta_threshold"),
        "agreement_support_labeled": agreement_support.get("num_labeled_mentions", 0),
        "agreement_support_hallucination_rate": agreement_support.get("hallucination_rate"),
        "agreement_support_correct_capture_rate": agreement_support.get("correct_capture_rate"),
        "agreement_strong_labeled": agreement_strong.get("num_labeled_mentions", 0),
        "agreement_strong_hallucination_rate": agreement_strong.get("hallucination_rate"),
        "agreement_strong_precision_lift": agreement_strong.get("precision_lift_vs_base"),
        "agreement_strong_hallucinated_recall": agreement_strong.get("hallucinated_recall"),
        "agreement_strong_correct_fpr": agreement_strong.get("correct_false_positive_rate"),
    }


def write_aggregate(output_dir, rows, args):
    aggregate = {
        "model_path": args.model_path,
        "eval_results": args.eval_results,
        "case_analysis": args.case_analysis,
        "student_ratio": args.student_ratio,
        "degradation_mode": args.degradation_mode,
        "teacher_ratios": [row["teacher_ratio"] for row in rows],
        "rows": rows,
    }
    json_path = os.path.join(output_dir, "base_trace_probe_summary.json")
    md_path = os.path.join(output_dir, "base_trace_probe_summary.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    headers = [
        "teacher ratio",
        "correct",
        "halluc",
        "logp gap",
        "tail gap < -0.05",
        "teacher<student gap",
        "correct entropy",
        "halluc entropy",
        "best gate F1",
        "gate precision",
        "gate recall",
        "gate correct FPR",
    ]
    lines = [
        "# Base Low-Resolution Critic Probe",
        "",
        f"- model_path: `{args.model_path}`",
        f"- eval_results: `{args.eval_results}`",
        f"- case_analysis: `{args.case_analysis}`",
        f"- student_ratio: `{args.student_ratio}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            format_ratio(row["teacher_ratio"]),
            fmt(row["correct_mentions"]),
            fmt(row["hallucinated_mentions"]),
            fmt(row["logp_gap_hallucinated_minus_correct"]),
            fmt(row["frac_delta_lt_neg0p05_gap"]),
            fmt(row["teacher_lt_student_gap"]),
            fmt(row["correct_student_entropy"]),
            fmt(row["hallucinated_student_entropy"]),
            fmt(row["best_gate_f1"]),
            fmt(row["best_gate_precision"]),
            fmt(row["best_gate_recall"]),
            fmt(row["best_gate_correct_fpr"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Positive evidence for a useful low-res critic means `logp gap` is negative, "
            "`tail gap < -0.05` is positive, and the best gate has precision lift with "
            "a tolerable correct-object false-positive rate.",
            "",
            "## Low-Resolution Agreement Buckets",
            "",
            "| teacher ratio | q70 delta | q90 delta | support n | support halluc rate | "
            "support correct capture | strong n | strong halluc rate | strong lift | "
            "strong recall | strong correct FPR |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        values = [
            format_ratio(row["teacher_ratio"]),
            fmt(row["agreement_q70_delta_threshold"]),
            fmt(row["agreement_q90_delta_threshold"]),
            fmt(row["agreement_support_labeled"]),
            fmt(row["agreement_support_hallucination_rate"]),
            fmt(row["agreement_support_correct_capture_rate"]),
            fmt(row["agreement_strong_labeled"]),
            fmt(row["agreement_strong_hallucination_rate"]),
            fmt(row["agreement_strong_precision_lift"]),
            fmt(row["agreement_strong_hallucinated_recall"]),
            fmt(row["agreement_strong_correct_fpr"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "For the near-sighted critic hypothesis, `support` should have a lower "
            "hallucination rate than the base rate, while `strong_disagree` should "
            "show hallucination enrichment without a large correct-object FPR.",
            "",
            "## Low-Resolution Support/Reject Quadrants",
            "",
            "| teacher ratio | support n | support halluc rate | support correct capture | "
            "reject n | reject halluc rate | reject lift | reject recall | reject correct FPR | "
            "uncertain n | uncertain halluc rate |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        values = [
            format_ratio(row["teacher_ratio"]),
            fmt(row["lowres_support_labeled"]),
            fmt(row["lowres_support_hallucination_rate"]),
            fmt(row["lowres_support_correct_capture_rate"]),
            fmt(row["lowres_reject_labeled"]),
            fmt(row["lowres_reject_hallucination_rate"]),
            fmt(row["lowres_reject_precision_lift"]),
            fmt(row["lowres_reject_hallucinated_recall"]),
            fmt(row["lowres_reject_correct_fpr"]),
            fmt(row["lowres_uncertain_labeled"]),
            fmt(row["lowres_uncertain_hallucination_rate"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "For the near-sighted critic hypothesis, a good ratio should have a low "
            "`support halluc rate` and a high `reject lift`, while keeping "
            "`reject correct FPR` small.",
            "",
        ]
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return json_path, md_path


def main():
    args = parse_args()
    ratios = parse_ratios(args.teacher_ratios)
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(args.eval_results)), "base_trace_probe")
    os.makedirs(args.output_dir, exist_ok=True)

    states = {
        ratio: ratio_state(args.output_dir, ratio, args.overwrite)
        for ratio in ratios
    }

    print("============================================================")
    print(" Base Low-Resolution Critic Probe")
    print("============================================================")
    print(f"Model path:       {args.model_path}")
    print(f"Eval results:     {args.eval_results}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Teacher ratios:   {[format_ratio(ratio) for ratio in ratios]}")
    print(f"Student ratio:    {args.student_ratio}")
    print(f"Max samples:      {args.max_samples}")
    print(f"Entropy:          {not args.no_entropy}")
    print("============================================================")

    if not args.skip_scoring:
        run_scoring(args, states)

    rows = []
    if not args.skip_analysis:
        for state in states.values():
            print(f"Analyzing {state['label']} trace: {state['trace_path']}")
            summary = run_analysis(args, state)
            rows.append(extract_row(state, summary))
        json_path, md_path = write_aggregate(args.output_dir, rows, args)
        print(f"Saved aggregate JSON to: {json_path}")
        print(f"Saved aggregate Markdown to: {md_path}")
        for row in rows:
            print(
                f"{row['label']}: logp_gap={fmt(row['logp_gap_hallucinated_minus_correct'])} "
                f"tail_gap<-0.05={fmt(row['frac_delta_lt_neg0p05_gap'])} "
                f"best_gate_f1={fmt(row['best_gate_f1'])} "
                f"correct_fpr={fmt(row['best_gate_correct_fpr'])}"
            )


if __name__ == "__main__":
    main()
