"""
Badcase / Goodcase Analyzer for Res-OPD CHAIR evaluation (Enhanced v2).

Compares a model's eval_results.jsonl against a baseline's eval_results.jsonl,
computes per-sample ObjF1 for both, and outputs the top-N worst (badcases) and
top-N best (goodcases) samples ranked by F1 delta (model - baseline).

ENHANCED FEATURES (v2):
  - Object-level fine-grained comparison: which specific GT objects were
    hit / missed / hallucinated by model vs baseline
  - Behavior pattern classification: automatically categorizes each case as
    silent_on_gt, over_suppressed, hallucination_reduced, hallucination_increased,
    gt_coverage_improved, object_substitution, or mixed
  - GT coverage analysis: how many GT objects model covers vs baseline
  - Summary includes behavior pattern distribution and coverage statistics

Usage:
    python eval/analyze_cases.py \
        --model-results  eval_results/v2/Res-OPD-.../train1500_test300_original_sr1p0/eval_results.jsonl \
        --baseline-results eval_results/v2/Qwen3VL-2B-Instruct/train1500_test300_original_sr1p0/eval_results.jsonl \
        --output-dir     eval_results/v2/case_analysis/sr1.0-tr0.25_vs_baseline \
        --top-n          20

    # Custom number of cases
    python eval/analyze_cases.py \
        --model-results  .../eval_results.jsonl \
        --baseline-results .../eval_results.jsonl \
        --output-dir     .../case_analysis \
        --top-n          50
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
from robust_chair_analysis import (
    parse_official_synonyms,
    build_double_word_dict,
    caption_to_words,
    compute_per_sample,
)


def load_eval_results(jsonl_path):
    """Load eval_results.jsonl into a dict keyed by image_id."""
    records = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = record.get("image_id")
            if image_id is None:
                continue
            caption = record.get("generated_caption", "")
            if caption and not caption.startswith("[ERROR]"):
                records[image_id] = record
    return records


def compute_sample_f1(per_sample_dict, image_id):
    """Compute ObjF1 for a single sample from per_sample dict."""
    sample = per_sample_dict.get(image_id)
    if sample is None:
        return None
    mentioned = sample["mentioned"]
    correct = sample["correct"]
    gt_count = sample["gt_count"]
    precision = correct / max(mentioned, 1)
    recall = correct / max(gt_count, 1)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mentioned": mentioned,
        "correct": correct,
        "hallucinated": sample["hallucinated"],
        "gt_count": gt_count,
    }


def extract_mentioned_objects(caption, mscoco_objects, inverse_synonym_dict,
                              double_word_dict):
    """Extract the set of canonical COCO objects mentioned in a caption.

    Returns a sorted list of unique canonical object names.
    """
    _, node_words = caption_to_words(
        caption, mscoco_objects, inverse_synonym_dict, double_word_dict
    )
    return sorted(set(node_words))


def compute_object_level_diff(gt_objects, model_mentioned, baseline_mentioned):
    """Compute object-level fine-grained comparison between model, baseline, and GT.

    Args:
        gt_objects: list of GT canonical object names
        model_mentioned: list of canonical objects mentioned by model
        baseline_mentioned: list of canonical objects mentioned by baseline

    Returns dict with per-object hit/miss/hallucination details.
    """
    gt_set = set(gt_objects)
    model_set = set(model_mentioned)
    baseline_set = set(baseline_mentioned)

    model_hit_gt = sorted(gt_set & model_set)
    model_missed_gt = sorted(gt_set - model_set)
    model_hallucinated = sorted(model_set - gt_set)

    baseline_hit_gt = sorted(gt_set & baseline_set)
    baseline_missed_gt = sorted(gt_set - baseline_set)
    baseline_hallucinated = sorted(baseline_set - gt_set)

    # Objects that model gained vs lost relative to baseline
    model_gained_gt = sorted(set(model_hit_gt) - set(baseline_hit_gt))
    model_lost_gt = sorted(set(baseline_hit_gt) - set(model_hit_gt))

    # Hallucinations unique to model / shared / removed
    model_only_hallucinations = sorted(model_set - baseline_set - gt_set)
    removed_hallucinations = sorted(baseline_set - model_set - gt_set)
    shared_hallucinations = sorted((model_set & baseline_set) - gt_set)

    return {
        "model_hit_gt": model_hit_gt,
        "model_missed_gt": model_missed_gt,
        "model_hallucinated": model_hallucinated,
        "baseline_hit_gt": baseline_hit_gt,
        "baseline_missed_gt": baseline_missed_gt,
        "baseline_hallucinated": baseline_hallucinated,
        "model_gained_gt": model_gained_gt,
        "model_lost_gt": model_lost_gt,
        "model_only_hallucinations": model_only_hallucinations,
        "removed_hallucinations": removed_hallucinations,
        "shared_hallucinations": shared_hallucinations,
        "gt_coverage_model": len(model_hit_gt) / max(len(gt_set), 1),
        "gt_coverage_baseline": len(baseline_hit_gt) / max(len(gt_set), 1),
        "gt_coverage_delta": (len(model_hit_gt) - len(baseline_hit_gt)) / max(len(gt_set), 1),
    }


def classify_behavior_pattern(obj_diff, model_f1_info, baseline_f1_info):
    """Classify the behavior pattern of model vs baseline for a single sample.

    Categories:
      - silent_on_gt: model missed more GT objects than baseline (recall drop)
      - over_suppressed: model mentions far fewer total objects AND recall dropped
      - hallucination_reduced: fewer hallucinations, recall stable or improved
      - hallucination_increased: more hallucinations than baseline
      - gt_coverage_improved: model hit strictly more GT objects than baseline
      - object_substitution: model lost GT objects but gained non-GT objects
      - no_change: identical object sets
      - mixed: multiple patterns co-occur
    """
    model_mentioned_count = model_f1_info["mentioned"]
    baseline_mentioned_count = baseline_f1_info["mentioned"]
    recall_delta = model_f1_info["recall"] - baseline_f1_info["recall"]
    halluc_delta = model_f1_info["hallucinated"] - baseline_f1_info["hallucinated"]

    gained = len(obj_diff["model_gained_gt"])
    lost = len(obj_diff["model_lost_gt"])
    new_halluc = len(obj_diff["model_only_hallucinations"])
    removed_halluc = len(obj_diff["removed_hallucinations"])

    patterns = []

    if lost > 0 and gained == 0:
        patterns.append("silent_on_gt")
    if lost > 0 and new_halluc > 0:
        patterns.append("object_substitution")
    if gained > 0 and lost == 0:
        patterns.append("gt_coverage_improved")
    if halluc_delta < 0 and recall_delta >= 0:
        patterns.append("hallucination_reduced")
    if halluc_delta > 0:
        patterns.append("hallucination_increased")

    mention_ratio = model_mentioned_count / max(baseline_mentioned_count, 1)
    if mention_ratio < 0.6 and recall_delta < 0:
        patterns.append("over_suppressed")

    if not patterns:
        if (set(obj_diff["model_hit_gt"]) == set(obj_diff["baseline_hit_gt"])
                and set(obj_diff["model_hallucinated"]) == set(obj_diff["baseline_hallucinated"])):
            return "no_change"
        return "minor_variation"

    if len(patterns) == 1:
        return patterns[0]
    return "mixed:" + "+".join(sorted(patterns))


def build_case_record(image_id, model_record, baseline_record,
                      model_f1_info, baseline_f1_info,
                      model_mentioned_objects, baseline_mentioned_objects,
                      mscoco_objects, inverse_synonym_dict, double_word_dict):
    """Build a structured case record with object-level detail."""
    gt_objects = model_record.get("gt_objects", [])

    obj_diff = compute_object_level_diff(
        gt_objects, model_mentioned_objects, baseline_mentioned_objects
    )
    behavior = classify_behavior_pattern(obj_diff, model_f1_info, baseline_f1_info)

    return {
        "image_id": image_id,
        "file_name": model_record.get("file_name", ""),
        "image_path": _resolve_image_path(model_record),
        "gt_objects": gt_objects,
        "gt_captions": model_record.get("gt_captions", []),
        "model_caption": model_record.get("generated_caption", ""),
        "baseline_caption": baseline_record.get("generated_caption", ""),
        "model_metrics": model_f1_info,
        "baseline_metrics": baseline_f1_info,
        "f1_delta": model_f1_info["f1"] - baseline_f1_info["f1"],
        "precision_delta": model_f1_info["precision"] - baseline_f1_info["precision"],
        "recall_delta": model_f1_info["recall"] - baseline_f1_info["recall"],
        # --- Object-level detail (NEW v2) ---
        "model_mentioned_objects": model_mentioned_objects,
        "baseline_mentioned_objects": baseline_mentioned_objects,
        "object_level_diff": obj_diff,
        "behavior_pattern": behavior,
    }


def _resolve_image_path(record):
    """Try to get image path from record or construct from file_name."""
    if "image_path" in record:
        return record["image_path"]
    file_name = record.get("file_name", "")
    if file_name:
        return f"/home/liuyanlin.lyl/notebook/data/COCO/coco2017val/val2017/{file_name}"
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Analyze badcases and goodcases vs baseline"
    )
    parser.add_argument(
        "--model-results", required=True,
        help="Path to model's eval_results.jsonl"
    )
    parser.add_argument(
        "--baseline-results", required=True,
        help="Path to baseline's eval_results.jsonl"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of badcases and goodcases to extract (default: 20)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model results: {args.model_results}")
    model_records = load_eval_results(args.model_results)
    print(f"  Loaded {len(model_records)} samples")

    print(f"Loading baseline results: {args.baseline_results}")
    baseline_records = load_eval_results(args.baseline_results)
    print(f"  Loaded {len(baseline_records)} samples")

    # Find common image_ids
    common_ids = set(model_records.keys()) & set(baseline_records.keys())
    print(f"Common samples: {len(common_ids)}")

    if not common_ids:
        print("ERROR: No common samples found between model and baseline!")
        sys.exit(1)

    # Compute per-sample CHAIR metrics
    print("Computing per-sample metrics ...")
    mscoco_objects, inverse_synonym_dict = parse_official_synonyms()
    double_word_dict = build_double_word_dict()

    model_eval_records = [
        {"image_id": iid, "generated_text": model_records[iid]["generated_caption"],
         "gt_objects": model_records[iid].get("gt_objects", [])}
        for iid in common_ids
    ]
    baseline_eval_records = [
        {"image_id": iid, "generated_text": baseline_records[iid]["generated_caption"],
         "gt_objects": baseline_records[iid].get("gt_objects", [])}
        for iid in common_ids
    ]

    model_per_sample = compute_per_sample(
        model_eval_records, mscoco_objects, inverse_synonym_dict, double_word_dict
    )
    baseline_per_sample = compute_per_sample(
        baseline_eval_records, mscoco_objects, inverse_synonym_dict, double_word_dict
    )

    # Compute F1 deltas and object-level analysis
    print("Computing object-level diffs and behavior patterns ...")
    case_records = []
    for image_id in common_ids:
        model_f1_info = compute_sample_f1(model_per_sample, image_id)
        baseline_f1_info = compute_sample_f1(baseline_per_sample, image_id)
        if model_f1_info is None or baseline_f1_info is None:
            continue

        model_mentioned = extract_mentioned_objects(
            model_records[image_id]["generated_caption"],
            mscoco_objects, inverse_synonym_dict, double_word_dict
        )
        baseline_mentioned = extract_mentioned_objects(
            baseline_records[image_id]["generated_caption"],
            mscoco_objects, inverse_synonym_dict, double_word_dict
        )

        case_records.append(
            build_case_record(
                image_id, model_records[image_id], baseline_records[image_id],
                model_f1_info, baseline_f1_info,
                model_mentioned, baseline_mentioned,
                mscoco_objects, inverse_synonym_dict, double_word_dict
            )
        )

    # Sort by F1 delta
    case_records.sort(key=lambda x: x["f1_delta"])

    badcases = case_records[:args.top_n]
    goodcases = case_records[-args.top_n:][::-1]  # Best first

    # --- Enhanced summary statistics (v2) ---
    all_deltas = [c["f1_delta"] for c in case_records]
    all_recall_deltas = [c["recall_delta"] for c in case_records]
    all_precision_deltas = [c["precision_delta"] for c in case_records]

    # Behavior pattern distribution
    pattern_counter = Counter(c["behavior_pattern"] for c in case_records)
    pattern_distribution = dict(pattern_counter.most_common())

    # GT coverage statistics
    coverage_deltas = [c["object_level_diff"]["gt_coverage_delta"] for c in case_records]
    total_gt_objects = sum(len(c["gt_objects"]) for c in case_records)
    total_model_hit = sum(len(c["object_level_diff"]["model_hit_gt"]) for c in case_records)
    total_baseline_hit = sum(len(c["object_level_diff"]["baseline_hit_gt"]) for c in case_records)
    total_model_halluc = sum(len(c["object_level_diff"]["model_hallucinated"]) for c in case_records)
    total_baseline_halluc = sum(len(c["object_level_diff"]["baseline_hallucinated"]) for c in case_records)

    # Per-pattern F1 delta averages
    pattern_f1_stats = {}
    for pattern in pattern_distribution:
        pattern_cases = [c for c in case_records if c["behavior_pattern"] == pattern]
        pattern_deltas = [c["f1_delta"] for c in pattern_cases]
        pattern_f1_stats[pattern] = {
            "count": len(pattern_cases),
            "f1_delta_mean": sum(pattern_deltas) / len(pattern_deltas),
            "recall_delta_mean": sum(c["recall_delta"] for c in pattern_cases) / len(pattern_cases),
            "precision_delta_mean": sum(c["precision_delta"] for c in pattern_cases) / len(pattern_cases),
        }

    # Most frequently lost/gained GT objects across all cases
    lost_object_counter = Counter()
    gained_object_counter = Counter()
    new_halluc_counter = Counter()
    removed_halluc_counter = Counter()
    for case in case_records:
        diff = case["object_level_diff"]
        for obj in diff["model_lost_gt"]:
            lost_object_counter[obj] += 1
        for obj in diff["model_gained_gt"]:
            gained_object_counter[obj] += 1
        for obj in diff["model_only_hallucinations"]:
            new_halluc_counter[obj] += 1
        for obj in diff["removed_hallucinations"]:
            removed_halluc_counter[obj] += 1

    summary = {
        "num_common_samples": len(common_ids),
        "top_n": args.top_n,
        "model_results": args.model_results,
        "baseline_results": args.baseline_results,
        # F1 delta stats
        "f1_delta_mean": sum(all_deltas) / len(all_deltas),
        "f1_delta_median": sorted(all_deltas)[len(all_deltas) // 2],
        "f1_delta_min": min(all_deltas),
        "f1_delta_max": max(all_deltas),
        "num_worse": sum(1 for d in all_deltas if d < 0),
        "num_better": sum(1 for d in all_deltas if d > 0),
        "num_equal": sum(1 for d in all_deltas if d == 0),
        # Recall / Precision delta stats (NEW v2)
        "recall_delta_mean": sum(all_recall_deltas) / len(all_recall_deltas),
        "precision_delta_mean": sum(all_precision_deltas) / len(all_precision_deltas),
        # GT coverage (NEW v2)
        "gt_coverage_model_overall": total_model_hit / max(total_gt_objects, 1),
        "gt_coverage_baseline_overall": total_baseline_hit / max(total_gt_objects, 1),
        "gt_coverage_delta_overall": (total_model_hit - total_baseline_hit) / max(total_gt_objects, 1),
        "gt_coverage_delta_mean": sum(coverage_deltas) / len(coverage_deltas),
        # Hallucination totals (NEW v2)
        "total_model_hallucinated_objects": total_model_halluc,
        "total_baseline_hallucinated_objects": total_baseline_halluc,
        "hallucination_delta_total": total_model_halluc - total_baseline_halluc,
        # Behavior pattern distribution (NEW v2)
        "behavior_pattern_distribution": pattern_distribution,
        "behavior_pattern_f1_stats": pattern_f1_stats,
        # Top object-level changes (NEW v2)
        "top_lost_gt_objects": dict(lost_object_counter.most_common(20)),
        "top_gained_gt_objects": dict(gained_object_counter.most_common(20)),
        "top_new_hallucination_objects": dict(new_halluc_counter.most_common(20)),
        "top_removed_hallucination_objects": dict(removed_halluc_counter.most_common(20)),
    }

    # Save outputs
    badcases_path = os.path.join(args.output_dir, "badcases.json")
    goodcases_path = os.path.join(args.output_dir, "goodcases.json")
    summary_path = os.path.join(args.output_dir, "summary.json")
    all_cases_path = os.path.join(args.output_dir, "all_cases_sorted.json")

    with open(badcases_path, "w", encoding="utf-8") as f:
        json.dump(badcases, f, indent=2, ensure_ascii=False)
    with open(goodcases_path, "w", encoding="utf-8") as f:
        json.dump(goodcases, f, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(all_cases_path, "w", encoding="utf-8") as f:
        json.dump(case_records, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"  Case Analysis Summary (Enhanced v2)")
    print(f"{'=' * 70}")
    print(f"  Common samples:       {summary['num_common_samples']}")
    print(f"  Worse than baseline:  {summary['num_worse']}")
    print(f"  Better than baseline: {summary['num_better']}")
    print(f"  Equal:                {summary['num_equal']}")
    print(f"  F1 delta:      mean={summary['f1_delta_mean']:.4f}  "
          f"median={summary['f1_delta_median']:.4f}  "
          f"range=[{summary['f1_delta_min']:.4f}, {summary['f1_delta_max']:.4f}]")
    print(f"  Recall delta:  mean={summary['recall_delta_mean']:.4f}")
    print(f"  Precision delta: mean={summary['precision_delta_mean']:.4f}")
    print(f"{'─' * 70}")
    print(f"  GT Coverage:  model={summary['gt_coverage_model_overall']:.4f}  "
          f"baseline={summary['gt_coverage_baseline_overall']:.4f}  "
          f"delta={summary['gt_coverage_delta_overall']:+.4f}")
    print(f"  Hallucinations: model={summary['total_model_hallucinated_objects']}  "
          f"baseline={summary['total_baseline_hallucinated_objects']}  "
          f"delta={summary['hallucination_delta_total']:+d}")
    print(f"{'─' * 70}")
    print(f"  Behavior Pattern Distribution:")
    for pattern, count in summary["behavior_pattern_distribution"].items():
        stats = summary["behavior_pattern_f1_stats"][pattern]
        print(f"    {pattern:30s}  n={count:4d}  "
              f"F1Δ={stats['f1_delta_mean']:+.4f}  "
              f"RecΔ={stats['recall_delta_mean']:+.4f}  "
              f"PrecΔ={stats['precision_delta_mean']:+.4f}")
    print(f"{'─' * 70}")
    if summary["top_lost_gt_objects"]:
        print(f"  Top Lost GT Objects (model missed but baseline hit):")
        for obj, cnt in list(summary["top_lost_gt_objects"].items())[:10]:
            print(f"    {obj:25s}  {cnt} times")
    if summary["top_gained_gt_objects"]:
        print(f"  Top Gained GT Objects (model hit but baseline missed):")
        for obj, cnt in list(summary["top_gained_gt_objects"].items())[:10]:
            print(f"    {obj:25s}  {cnt} times")
    if summary["top_new_hallucination_objects"]:
        print(f"  Top New Hallucination Objects (model only):")
        for obj, cnt in list(summary["top_new_hallucination_objects"].items())[:10]:
            print(f"    {obj:25s}  {cnt} times")
    if summary["top_removed_hallucination_objects"]:
        print(f"  Top Removed Hallucination Objects (baseline only):")
        for obj, cnt in list(summary["top_removed_hallucination_objects"].items())[:10]:
            print(f"    {obj:25s}  {cnt} times")
    print(f"{'=' * 70}")
    print(f"\nTop-{args.top_n} badcases saved to: {badcases_path}")
    print(f"Top-{args.top_n} goodcases saved to: {goodcases_path}")
    print(f"All cases sorted saved to: {all_cases_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
