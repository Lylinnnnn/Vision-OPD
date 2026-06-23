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
  - Selective suppression diagnostics: separates useful hallucination
    suppression from harmful GT suppression, and surfaces candidate cases for
    selective distillation / delta-based filtering.
  - Optional uncertainty hooks: if eval_results.jsonl contains token logprob,
    entropy, or top-k logprob fields, group-level uncertainty stats are
    reported automatically.

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
import math
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


def _safe_mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _safe_rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _word_count(text):
    return len(re.findall(r"[A-Za-z0-9]+", text or ""))


def _fmt_signed(value, digits=4):
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def extract_mentioned_objects(caption, mscoco_objects, inverse_synonym_dict,
                              double_word_dict):
    """Extract the set of canonical COCO objects mentioned in a caption.

    Returns a sorted list of unique canonical object names.
    """
    _, node_words = caption_to_words(
        caption, mscoco_objects, inverse_synonym_dict, double_word_dict
    )
    return sorted(set(node_words))


def _as_numeric(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
    return None


def _flatten_numeric(value):
    """Flatten nested numeric structures from optional uncertainty fields."""
    if value is None:
        return []
    number = _as_numeric(value)
    if number is not None:
        return [number]
    if isinstance(value, dict):
        values = []
        for item in value.values():
            values.extend(_flatten_numeric(item))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_flatten_numeric(item))
        return values
    return []


def _first_present(record, names):
    for name in names:
        if name in record:
            return record[name]
    return None


def _mean_field(record, scalar_names, list_names):
    scalar_value = _first_present(record, scalar_names)
    scalar_number = _as_numeric(scalar_value)
    if scalar_number is not None:
        return scalar_number

    list_value = _first_present(record, list_names)
    values = _flatten_numeric(list_value)
    return _safe_mean(values)


def _top_logprob_values(top_logprobs):
    """Return per-token sorted top-k logprobs from common serialized formats."""
    if not isinstance(top_logprobs, (list, tuple)):
        return []

    per_token = []
    for entry in top_logprobs:
        if isinstance(entry, dict):
            vals = []
            for value in entry.values():
                number = _as_numeric(value)
                if number is not None:
                    vals.append(number)
                elif isinstance(value, dict):
                    nested = _as_numeric(value.get("logprob"))
                    if nested is not None:
                        vals.append(nested)
            if vals:
                per_token.append(sorted(vals, reverse=True))
        elif isinstance(entry, (list, tuple)):
            vals = []
            for item in entry:
                if isinstance(item, dict):
                    number = _as_numeric(item.get("logprob"))
                else:
                    number = _as_numeric(item)
                if number is not None:
                    vals.append(number)
            if vals:
                per_token.append(sorted(vals, reverse=True))
    return per_token


def _entropy_from_logprobs(logprobs):
    if not logprobs:
        return None
    max_logprob = max(logprobs)
    weights = [math.exp(lp - max_logprob) for lp in logprobs]
    total = sum(weights)
    if total <= 0:
        return None
    probs = [w / total for w in weights]
    return -sum(p * math.log(max(p, 1e-12)) for p in probs)


def extract_uncertainty_features(record):
    """Extract optional confidence/uncertainty statistics if present.

    The current eval files may not contain these fields. This function is
    intentionally permissive so future runs can add token_logprobs,
    token_entropies, top_logprobs, top1_probs, etc. without changing this
    analyzer.
    """
    features = {}

    field_specs = {
        "mean_token_logprob": (
            ["mean_token_logprob", "avg_token_logprob", "mean_logprob", "avg_logprob"],
            ["token_logprobs", "generated_token_logprobs", "per_token_logprobs"],
        ),
        "mean_token_entropy": (
            ["mean_token_entropy", "avg_token_entropy", "mean_entropy", "avg_entropy"],
            ["token_entropies", "generated_token_entropies", "per_token_entropies"],
        ),
        "mean_top1_prob": (
            ["mean_top1_prob", "avg_top1_prob"],
            ["top1_probs", "token_top1_probs"],
        ),
        "mean_top1_top2_margin": (
            ["mean_top1_top2_margin", "avg_top1_top2_margin", "mean_top2_margin"],
            ["top1_top2_margins", "token_top1_top2_margins", "top2_margins"],
        ),
    }

    for metric, (scalar_names, list_names) in field_specs.items():
        value = _mean_field(record, scalar_names, list_names)
        if value is not None:
            features[metric] = value

    top_logprobs = _first_present(
        record,
        ["top_logprobs", "generated_top_logprobs", "token_top_logprobs"],
    )
    per_token_top = _top_logprob_values(top_logprobs)
    if per_token_top:
        top1_logprobs = [vals[0] for vals in per_token_top if vals]
        top2_logprob_margins = [
            vals[0] - vals[1] for vals in per_token_top if len(vals) >= 2
        ]
        topk_entropies = [
            _entropy_from_logprobs(vals) for vals in per_token_top if len(vals) >= 2
        ]
        features.setdefault("mean_token_logprob", _safe_mean(top1_logprobs))
        margin_mean = _safe_mean(top2_logprob_margins)
        entropy_mean = _safe_mean(topk_entropies)
        if margin_mean is not None:
            features.setdefault("mean_top1_top2_logprob_margin", margin_mean)
        if entropy_mean is not None:
            features.setdefault("mean_topk_entropy", entropy_mean)

    return features


def compute_uncertainty_delta(model_uncertainty, baseline_uncertainty):
    delta = {}
    for key in sorted(set(model_uncertainty) | set(baseline_uncertainty)):
        if key in model_uncertainty and key in baseline_uncertainty:
            delta[key] = model_uncertainty[key] - baseline_uncertainty[key]
    return delta


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


def compute_case_diagnostics(case_record):
    """Compute analysis flags that directly test the low-res-teacher hypothesis."""
    obj_diff = case_record["object_level_diff"]
    model_metrics = case_record["model_metrics"]
    baseline_metrics = case_record["baseline_metrics"]

    lost_gt_count = len(obj_diff["model_lost_gt"])
    gained_gt_count = len(obj_diff["model_gained_gt"])
    new_hallucination_count = len(obj_diff["model_only_hallucinations"])
    removed_hallucination_count = len(obj_diff["removed_hallucinations"])
    shared_hallucination_count = len(obj_diff["shared_hallucinations"])

    correct_delta = model_metrics["correct"] - baseline_metrics["correct"]
    hallucination_delta = model_metrics["hallucinated"] - baseline_metrics["hallucinated"]
    mentioned_delta = model_metrics["mentioned"] - baseline_metrics["mentioned"]

    strict_selective = (
        lost_gt_count == 0
        and removed_hallucination_count > 0
        and new_hallucination_count == 0
    )
    pure_hallucination_reduction = correct_delta == 0 and hallucination_delta < 0
    recall_preserved_hallucination_reduced = (
        case_record["recall_delta"] >= 0 and hallucination_delta < 0
    )
    coverage_improved_no_new_hallucination = correct_delta > 0 and hallucination_delta <= 0
    harmful_gt_suppression = correct_delta < 0 and hallucination_delta >= 0
    gt_hallucination_tradeoff = correct_delta < 0 and hallucination_delta < 0
    risky_coverage_tradeoff = correct_delta > 0 and hallucination_delta > 0
    over_suppression_no_hallucination_gain = (
        mentioned_delta < 0 and case_record["recall_delta"] < 0
        and hallucination_delta >= 0
    )

    # Positive values mean "useful teacher signal"; negative values mean GT risk.
    # Losing GT is weighted more heavily because the intended next step is to
    # preserve recall while still suppressing hallucination.
    distill_utility_score = (
        removed_hallucination_count
        - new_hallucination_count
        + gained_gt_count
        - 2 * lost_gt_count
    )

    if strict_selective:
        recommendation = "strong_positive_selective_suppression"
    elif coverage_improved_no_new_hallucination:
        recommendation = "positive_coverage_gain"
    elif recall_preserved_hallucination_reduced:
        recommendation = "positive_recall_preserved_hallucination_reduced"
    elif harmful_gt_suppression or over_suppression_no_hallucination_gain:
        recommendation = "negative_gt_suppression"
    elif gt_hallucination_tradeoff:
        recommendation = "tradeoff_needs_delta_or_filtering"
    elif risky_coverage_tradeoff or new_hallucination_count > 0:
        recommendation = "negative_new_hallucination_risk"
    else:
        recommendation = "neutral_or_minor"

    return {
        "correct_delta": correct_delta,
        "hallucination_delta": hallucination_delta,
        "mentioned_delta": mentioned_delta,
        "lost_gt_count": lost_gt_count,
        "gained_gt_count": gained_gt_count,
        "removed_hallucination_count": removed_hallucination_count,
        "new_hallucination_count": new_hallucination_count,
        "shared_hallucination_count": shared_hallucination_count,
        "strict_selective_suppression": strict_selective,
        "pure_hallucination_reduction": pure_hallucination_reduction,
        "recall_preserved_hallucination_reduced": recall_preserved_hallucination_reduced,
        "coverage_improved_no_new_hallucination": coverage_improved_no_new_hallucination,
        "harmful_gt_suppression": harmful_gt_suppression,
        "gt_hallucination_tradeoff": gt_hallucination_tradeoff,
        "risky_coverage_tradeoff": risky_coverage_tradeoff,
        "over_suppression_no_hallucination_gain": over_suppression_no_hallucination_gain,
        "distill_utility_score": distill_utility_score,
        "distill_recommendation": recommendation,
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
    model_caption = model_record.get("generated_caption", "")
    baseline_caption = baseline_record.get("generated_caption", "")
    model_uncertainty = extract_uncertainty_features(model_record)
    baseline_uncertainty = extract_uncertainty_features(baseline_record)

    case_record = {
        "image_id": image_id,
        "file_name": model_record.get("file_name", ""),
        "image_path": _resolve_image_path(model_record),
        "gt_objects": gt_objects,
        "gt_captions": model_record.get("gt_captions", []),
        "model_caption": model_caption,
        "baseline_caption": baseline_caption,
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
        "caption_length": {
            "model_words": _word_count(model_caption),
            "baseline_words": _word_count(baseline_caption),
            "word_delta": _word_count(model_caption) - _word_count(baseline_caption),
        },
        "model_uncertainty": model_uncertainty,
        "baseline_uncertainty": baseline_uncertainty,
        "uncertainty_delta": compute_uncertainty_delta(
            model_uncertainty, baseline_uncertainty
        ),
    }
    case_record["case_diagnostics"] = compute_case_diagnostics(case_record)
    case_record["distill_recommendation"] = (
        case_record["case_diagnostics"]["distill_recommendation"]
    )
    return case_record


def _resolve_image_path(record):
    """Try to get image path from record or construct from file_name."""
    if "image_path" in record:
        return record["image_path"]
    file_name = record.get("file_name", "")
    if file_name:
        return f"/home/liuyanlin.lyl/notebook/data/COCO/coco2017val/val2017/{file_name}"
    return ""


def _diagnostic_group_stats(group_cases, total_cases):
    diagnostics = [c["case_diagnostics"] for c in group_cases]
    return {
        "count": len(group_cases),
        "rate": _safe_rate(len(group_cases), total_cases),
        "f1_delta_mean": _safe_mean(c["f1_delta"] for c in group_cases),
        "recall_delta_mean": _safe_mean(c["recall_delta"] for c in group_cases),
        "precision_delta_mean": _safe_mean(c["precision_delta"] for c in group_cases),
        "word_delta_mean": _safe_mean(
            c.get("caption_length", {}).get("word_delta") for c in group_cases
        ),
        "mentioned_delta_sum": sum(d["mentioned_delta"] for d in diagnostics),
        "correct_delta_sum": sum(d["correct_delta"] for d in diagnostics),
        "hallucination_delta_sum": sum(d["hallucination_delta"] for d in diagnostics),
        "lost_gt_total": sum(d["lost_gt_count"] for d in diagnostics),
        "gained_gt_total": sum(d["gained_gt_count"] for d in diagnostics),
        "removed_hallucination_total": sum(
            d["removed_hallucination_count"] for d in diagnostics
        ),
        "new_hallucination_total": sum(
            d["new_hallucination_count"] for d in diagnostics
        ),
        "distill_utility_mean": _safe_mean(
            d["distill_utility_score"] for d in diagnostics
        ),
    }


def _counter_from_cases(cases, diff_key):
    counter = Counter()
    for case in cases:
        for obj in case["object_level_diff"][diff_key]:
            counter[obj] += 1
    return counter


def _top_case_ids(cases, n=30, reverse=True):
    ordered = sorted(
        cases,
        key=lambda c: (
            c["case_diagnostics"]["distill_utility_score"],
            c["f1_delta"],
            c["recall_delta"],
        ),
        reverse=reverse,
    )
    return [c["image_id"] for c in ordered[:n]]


def _recommendation_distribution(case_records):
    counter = Counter(c["distill_recommendation"] for c in case_records)
    return dict(counter.most_common())


def _build_uncertainty_group_stats(case_records, group_map):
    metrics = set()
    for case in case_records:
        metrics.update(case.get("model_uncertainty", {}).keys())
        metrics.update(case.get("baseline_uncertainty", {}).keys())
        metrics.update(case.get("uncertainty_delta", {}).keys())
    if not metrics:
        return {}

    group_stats = {}
    for group_name, group_cases in group_map.items():
        metric_stats = {}
        for metric in sorted(metrics):
            model_values = [
                c.get("model_uncertainty", {}).get(metric) for c in group_cases
            ]
            baseline_values = [
                c.get("baseline_uncertainty", {}).get(metric) for c in group_cases
            ]
            delta_values = [
                c.get("uncertainty_delta", {}).get(metric) for c in group_cases
            ]
            if any(v is not None for v in model_values + baseline_values + delta_values):
                metric_stats[metric] = {
                    "model_mean": _safe_mean(model_values),
                    "baseline_mean": _safe_mean(baseline_values),
                    "delta_mean": _safe_mean(delta_values),
                }
        group_stats[group_name] = metric_stats
    return group_stats


def build_selective_suppression_summary(case_records, badcases, goodcases):
    """Aggregate stats that test whether low-res supervision is selective."""
    total_cases = len(case_records)
    group_defs = {
        "strict_selective_suppression": lambda c: (
            c["case_diagnostics"]["strict_selective_suppression"]
        ),
        "pure_hallucination_reduction": lambda c: (
            c["case_diagnostics"]["pure_hallucination_reduction"]
        ),
        "recall_preserved_hallucination_reduced": lambda c: (
            c["case_diagnostics"]["recall_preserved_hallucination_reduced"]
        ),
        "coverage_improved_no_new_hallucination": lambda c: (
            c["case_diagnostics"]["coverage_improved_no_new_hallucination"]
        ),
        "harmful_gt_suppression": lambda c: (
            c["case_diagnostics"]["harmful_gt_suppression"]
        ),
        "gt_hallucination_tradeoff": lambda c: (
            c["case_diagnostics"]["gt_hallucination_tradeoff"]
        ),
        "risky_coverage_tradeoff": lambda c: (
            c["case_diagnostics"]["risky_coverage_tradeoff"]
        ),
        "over_suppression_no_hallucination_gain": lambda c: (
            c["case_diagnostics"]["over_suppression_no_hallucination_gain"]
        ),
    }

    groups = {
        name: [case for case in case_records if pred(case)]
        for name, pred in group_defs.items()
    }
    group_stats = {
        name: _diagnostic_group_stats(group_cases, total_cases)
        for name, group_cases in groups.items()
    }

    diagnostics = [c["case_diagnostics"] for c in case_records]
    total_lost_gt = sum(d["lost_gt_count"] for d in diagnostics)
    total_gained_gt = sum(d["gained_gt_count"] for d in diagnostics)
    total_removed_halluc = sum(d["removed_hallucination_count"] for d in diagnostics)
    total_new_halluc = sum(d["new_hallucination_count"] for d in diagnostics)

    strict_count = len(groups["strict_selective_suppression"])
    harmful_count = len(groups["harmful_gt_suppression"])
    tradeoff_count = len(groups["gt_hallucination_tradeoff"])

    group_map_for_uncertainty = {
        "all": case_records,
        "strict_selective_suppression": groups["strict_selective_suppression"],
        "harmful_gt_suppression": groups["harmful_gt_suppression"],
        "gt_hallucination_tradeoff": groups["gt_hallucination_tradeoff"],
        "top_badcases": badcases,
        "top_goodcases": goodcases,
    }

    return {
        "definition": {
            "strict_selective_suppression": (
                "model loses no baseline-hit GT objects, removes at least one "
                "baseline-only hallucination, and adds no model-only hallucination"
            ),
            "harmful_gt_suppression": (
                "model loses GT coverage while hallucination count is not reduced"
            ),
            "gt_hallucination_tradeoff": (
                "model removes hallucinations but also loses GT coverage"
            ),
            "distill_utility_score": (
                "removed_hallucinations - new_hallucinations + gained_gt - 2*lost_gt"
            ),
        },
        "group_stats": group_stats,
        "object_transition_totals": {
            "lost_gt_total": total_lost_gt,
            "gained_gt_total": total_gained_gt,
            "removed_hallucination_total": total_removed_halluc,
            "new_hallucination_total": total_new_halluc,
            "net_gt_delta": total_gained_gt - total_lost_gt,
            "net_hallucination_delta": total_new_halluc - total_removed_halluc,
            "removed_hallucination_per_lost_gt": (
                total_removed_halluc / max(total_lost_gt, 1)
            ),
            "strict_selective_removed_hallucination_total": sum(
                c["case_diagnostics"]["removed_hallucination_count"]
                for c in groups["strict_selective_suppression"]
            ),
        },
        "selectivity_ratios": {
            "strict_vs_harmful": strict_count / max(strict_count + harmful_count, 1),
            "strict_vs_harmful_plus_tradeoff": (
                strict_count / max(strict_count + harmful_count + tradeoff_count, 1)
            ),
            "recall_preserved_hallucination_reduced_vs_harmful": (
                len(groups["recall_preserved_hallucination_reduced"])
                / max(len(groups["recall_preserved_hallucination_reduced"])
                      + harmful_count, 1)
            ),
        },
        "recommendation_distribution": _recommendation_distribution(case_records),
        "candidate_case_ids": {
            "strong_positive_selective": _top_case_ids(
                groups["strict_selective_suppression"], n=50, reverse=True
            ),
            "positive_coverage_gain": _top_case_ids(
                groups["coverage_improved_no_new_hallucination"], n=50, reverse=True
            ),
            "negative_gt_suppression": _top_case_ids(
                groups["harmful_gt_suppression"], n=50, reverse=False
            ),
            "tradeoff_needs_delta_or_filtering": _top_case_ids(
                groups["gt_hallucination_tradeoff"], n=50, reverse=False
            ),
        },
        "top_objects_by_group": {
            "strict_selectively_removed_hallucinations": dict(
                _counter_from_cases(
                    groups["strict_selective_suppression"],
                    "removed_hallucinations",
                ).most_common(20)
            ),
            "strict_selective_gt_context": dict(
                Counter(
                    obj
                    for c in groups["strict_selective_suppression"]
                    for obj in c["gt_objects"]
                ).most_common(20)
            ),
            "harmful_lost_gt_objects": dict(
                _counter_from_cases(
                    groups["harmful_gt_suppression"],
                    "model_lost_gt",
                ).most_common(20)
            ),
            "tradeoff_lost_gt_objects": dict(
                _counter_from_cases(
                    groups["gt_hallucination_tradeoff"],
                    "model_lost_gt",
                ).most_common(20)
            ),
            "tradeoff_removed_hallucinations": dict(
                _counter_from_cases(
                    groups["gt_hallucination_tradeoff"],
                    "removed_hallucinations",
                ).most_common(20)
            ),
            "coverage_gain_gt_objects": dict(
                _counter_from_cases(
                    groups["coverage_improved_no_new_hallucination"],
                    "model_gained_gt",
                ).most_common(20)
            ),
        },
        "top_n_case_stats": {
            "badcases": _diagnostic_group_stats(badcases, total_cases),
            "goodcases": _diagnostic_group_stats(goodcases, total_cases),
            "badcase_recommendation_distribution": _recommendation_distribution(badcases),
            "goodcase_recommendation_distribution": _recommendation_distribution(goodcases),
        },
        "caption_length_stats": {
            "model_words_mean": _safe_mean(
                c["caption_length"]["model_words"] for c in case_records
            ),
            "baseline_words_mean": _safe_mean(
                c["caption_length"]["baseline_words"] for c in case_records
            ),
            "word_delta_mean": _safe_mean(
                c["caption_length"]["word_delta"] for c in case_records
            ),
            "word_delta_median": _median(
                c["caption_length"]["word_delta"] for c in case_records
            ),
            "model_shorter_count": sum(
                1 for c in case_records if c["caption_length"]["word_delta"] < 0
            ),
            "model_longer_count": sum(
                1 for c in case_records if c["caption_length"]["word_delta"] > 0
            ),
            "model_equal_length_count": sum(
                1 for c in case_records if c["caption_length"]["word_delta"] == 0
            ),
        },
        "uncertainty_group_stats": _build_uncertainty_group_stats(
            case_records, group_map_for_uncertainty
        ),
    }


def enrich_existing_case_record(case):
    """Backfill v3 diagnostics for an existing all_cases_sorted.json record."""
    model_caption = case.get("model_caption", "")
    baseline_caption = case.get("baseline_caption", "")
    case.setdefault("caption_length", {
        "model_words": _word_count(model_caption),
        "baseline_words": _word_count(baseline_caption),
        "word_delta": _word_count(model_caption) - _word_count(baseline_caption),
    })
    case.setdefault("model_uncertainty", {})
    case.setdefault("baseline_uncertainty", {})
    case.setdefault("uncertainty_delta", compute_uncertainty_delta(
        case["model_uncertainty"], case["baseline_uncertainty"]
    ))
    case["case_diagnostics"] = compute_case_diagnostics(case)
    case["distill_recommendation"] = (
        case["case_diagnostics"]["distill_recommendation"]
    )
    return case


def refresh_from_all_cases(all_cases_path, output_dir, top_n):
    with open(all_cases_path, "r", encoding="utf-8") as f:
        case_records = json.load(f)

    case_records = [enrich_existing_case_record(c) for c in case_records]
    case_records.sort(key=lambda x: x["f1_delta"])
    badcases = case_records[:top_n]
    goodcases = case_records[-top_n:][::-1]

    all_deltas = [c["f1_delta"] for c in case_records]
    all_recall_deltas = [c["recall_delta"] for c in case_records]
    all_precision_deltas = [c["precision_delta"] for c in case_records]
    pattern_distribution = dict(
        Counter(c["behavior_pattern"] for c in case_records).most_common()
    )
    recommendation_distribution = dict(
        Counter(c["distill_recommendation"] for c in case_records).most_common()
    )
    selective_suppression_summary = build_selective_suppression_summary(
        case_records, badcases, goodcases
    )

    summary = {
        "num_common_samples": len(case_records),
        "top_n": top_n,
        "source_all_cases": all_cases_path,
        "f1_delta_mean": _safe_mean(all_deltas),
        "f1_delta_median": _median(all_deltas),
        "f1_delta_min": min(all_deltas) if all_deltas else None,
        "f1_delta_max": max(all_deltas) if all_deltas else None,
        "num_worse": sum(1 for d in all_deltas if d < 0),
        "num_better": sum(1 for d in all_deltas if d > 0),
        "num_equal": sum(1 for d in all_deltas if d == 0),
        "recall_delta_mean": _safe_mean(all_recall_deltas),
        "precision_delta_mean": _safe_mean(all_precision_deltas),
        "behavior_pattern_distribution": pattern_distribution,
        "distill_recommendation_distribution": recommendation_distribution,
        "selective_suppression_summary": selective_suppression_summary,
    }

    os.makedirs(output_dir, exist_ok=True)
    badcases_path = os.path.join(output_dir, "badcases.json")
    goodcases_path = os.path.join(output_dir, "goodcases.json")
    summary_path = os.path.join(output_dir, "summary.json")
    all_cases_path = os.path.join(output_dir, "all_cases_sorted.json")

    with open(badcases_path, "w", encoding="utf-8") as f:
        json.dump(badcases, f, indent=2, ensure_ascii=False)
    with open(goodcases_path, "w", encoding="utf-8") as f:
        json.dump(goodcases, f, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(all_cases_path, "w", encoding="utf-8") as f:
        json.dump(case_records, f, indent=2, ensure_ascii=False)

    strict = selective_suppression_summary["group_stats"]["strict_selective_suppression"]
    harmful = selective_suppression_summary["group_stats"]["harmful_gt_suppression"]
    ratio = selective_suppression_summary["selectivity_ratios"]["strict_vs_harmful"]
    print(f"Refreshed diagnostics from: {all_cases_path}")
    print(f"  strict selective: {strict['count']}")
    print(f"  harmful GT suppression: {harmful['count']}")
    print(f"  strict/(strict+harmful): {ratio:.3f}")
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze badcases and goodcases vs baseline"
    )
    parser.add_argument(
        "--model-results",
        help="Path to model's eval_results.jsonl"
    )
    parser.add_argument(
        "--baseline-results",
        help="Path to baseline's eval_results.jsonl"
    )
    parser.add_argument(
        "--from-all-cases",
        help="Refresh v3 diagnostics from an existing all_cases_sorted.json"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save analysis results. Defaults to <model-results dir>/case_analysis/",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of badcases and goodcases to extract (default: 20)"
    )
    args = parser.parse_args()

    # Derive default output-dir from --model-results when not specified
    if args.output_dir is None:
        if args.model_results:
            model_dir = os.path.dirname(os.path.abspath(args.model_results))
            args.output_dir = os.path.join(model_dir, "case_analysis")
        elif args.from_all_cases:
            cases_dir = os.path.dirname(os.path.abspath(args.from_all_cases))
            args.output_dir = cases_dir
        else:
            parser.error("--output-dir is required when neither --model-results nor --from-all-cases is provided")
        print(f"[Default] --output-dir not specified, using: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.from_all_cases:
        refresh_from_all_cases(args.from_all_cases, args.output_dir, args.top_n)
        return

    if not args.model_results or not args.baseline_results:
        parser.error("--model-results and --baseline-results are required unless --from-all-cases is used")

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

    selective_suppression_summary = build_selective_suppression_summary(
        case_records, badcases, goodcases
    )

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
        # Hypothesis-oriented diagnostics (NEW v3)
        "selective_suppression_summary": selective_suppression_summary,
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
    print(f"{'─' * 70}")
    selective = summary["selective_suppression_summary"]
    group_stats = selective["group_stats"]
    strict = group_stats["strict_selective_suppression"]
    recall_preserved = group_stats["recall_preserved_hallucination_reduced"]
    coverage_gain = group_stats["coverage_improved_no_new_hallucination"]
    harmful = group_stats["harmful_gt_suppression"]
    tradeoff = group_stats["gt_hallucination_tradeoff"]
    ratios = selective["selectivity_ratios"]
    transitions = selective["object_transition_totals"]
    print(f"  Selective Suppression Diagnostics:")
    print(f"    strict selective:       n={strict['count']:4d}  "
          f"rate={strict['rate']:.3f}  F1Δ={_fmt_signed(strict['f1_delta_mean'])}")
    print(f"    recall-preserved h↓:    n={recall_preserved['count']:4d}  "
          f"rate={recall_preserved['rate']:.3f}  "
          f"F1Δ={_fmt_signed(recall_preserved['f1_delta_mean'])}")
    print(f"    coverage gain, h not ↑: n={coverage_gain['count']:4d}  "
          f"rate={coverage_gain['rate']:.3f}  "
          f"F1Δ={_fmt_signed(coverage_gain['f1_delta_mean'])}")
    print(f"    harmful GT suppression: n={harmful['count']:4d}  "
          f"rate={harmful['rate']:.3f}  F1Δ={_fmt_signed(harmful['f1_delta_mean'])}")
    print(f"    tradeoff h↓ + GT↓:      n={tradeoff['count']:4d}  "
          f"rate={tradeoff['rate']:.3f}  F1Δ={_fmt_signed(tradeoff['f1_delta_mean'])}")
    print(f"    selectivity strict/(strict+harmful): "
          f"{ratios['strict_vs_harmful']:.3f}")
    print(f"    transitions: lost_gt={transitions['lost_gt_total']}  "
          f"gained_gt={transitions['gained_gt_total']}  "
          f"removed_h={transitions['removed_hallucination_total']}  "
          f"new_h={transitions['new_hallucination_total']}")
    print(f"{'=' * 70}")
    print(f"\nTop-{args.top_n} badcases saved to: {badcases_path}")
    print(f"Top-{args.top_n} goodcases saved to: {goodcases_path}")
    print(f"All cases sorted saved to: {all_cases_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
