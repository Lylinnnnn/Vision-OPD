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

Note:
    --case-analysis is only meaningful when the trace records and case-analysis
    records cover the same image split. Training rollout traces usually do not
    overlap with eval/test case-analysis outputs; the script reports overlap
    diagnostics instead of silently treating that as a parser failure.
"""

import argparse
import glob
import json
import math
import os
import re
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
    try_singularize,
)


OBJECT_METRIC_KEYS = [
    "student_selected_logprob_mean",
    "teacher_selected_logprob_mean",
    "teacher_minus_student_selected_logprob_mean",
    "student_selected_logprob_sum",
    "teacher_selected_logprob_sum",
    "teacher_minus_student_selected_logprob_sum",
    "student_entropy_mean",
    "teacher_entropy_mean",
    "teacher_minus_student_entropy_mean",
    "student_topk_mass_mean",
    "teacher_topk_mass_mean",
    "student_top1_top2_margin_mean",
    "teacher_top1_top2_margin_mean",
    "topk_overlap_ratio_mean",
    "topk_jaccard_mean",
    "top1_match_frac",
    "selected_token_rank_in_student_topk_mean",
    "selected_token_rank_in_teacher_topk_mean",
    "selected_token_in_teacher_topk_frac",
    "teacher_selected_logprob_lt_student_frac",
]

OBJECT_TOKEN_METRIC_KEYS = [
    "student_selected_logprob",
    "teacher_selected_logprob",
    "_teacher_minus_student_selected_logprob",
    "student_entropy",
    "teacher_entropy",
    "_teacher_minus_student_entropy",
    "student_topk_mass",
    "teacher_topk_mass",
    "student_top1_top2_margin",
    "teacher_top1_top2_margin",
    "topk_overlap_ratio",
    "topk_jaccard",
    "top1_match",
    "selected_token_rank_in_student_topk",
    "selected_token_rank_in_teacher_topk",
]

DEFAULT_ENTROPY_BIN_EDGES = [0.0, 0.5, 1.0, 1.5, float("inf")]
DEFAULT_GATE_ENTROPY_THRESHOLDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
DEFAULT_GATE_LOGP_MARGINS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
DEFAULT_QUADRANT_CONFIG = {
    "support_delta_min": -0.01,
    "reject_delta_max": -0.05,
    "teacher_entropy_max": 1.5,
    "teacher_margin_min": 0.0,
    "support_rank_max": 5.0,
    "reject_rank_min": 20.0,
    "support_top1_min": 0.5,
    "reject_top1_max": 0.25,
    "reject_in_teacher_topk_max": 0.5,
}
DELTA_DISTRIBUTION_FIELDS = [
    "teacher_minus_student_selected_logprob_mean",
    "teacher_minus_student_selected_logprob_sum",
]
DELTA_RATE_THRESHOLDS = [-0.2, -0.1, -0.05, -0.01, 0.0]
DELTA_PERCENTILES = [10, 25, 50, 75, 90]


def safe_mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def finite_values(rows, key):
    return [float(row.get(key)) for row in rows if is_finite_number(row.get(key))]


def quantile(values, percentile):
    values = sorted(float(value) for value in values if is_finite_number(value))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * (percentile / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize_value_distribution(rows, key, rate_thresholds=None):
    values = finite_values(rows, key)
    summary = {
        "count": len(values),
        "mean": safe_mean(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }
    for percentile in DELTA_PERCENTILES:
        summary[f"p{percentile}"] = quantile(values, percentile)
    for threshold in rate_thresholds or []:
        label = str(threshold).replace("-", "neg").replace(".", "p")
        summary[f"frac_lt_{label}"] = safe_rate(
            len([value for value in values if value < threshold]),
            len(values),
        )
    return summary


def parse_float_sequence(raw, default):
    if raw is None or str(raw).strip() == "":
        return list(default)
    values = []
    for part in str(raw).split(","):
        text = part.strip().lower()
        if not text:
            continue
        if text in {"inf", "+inf", "infinity", "+infinity"}:
            values.append(float("inf"))
        else:
            values.append(float(text))
    return values or list(default)


def build_entropy_bins(edges):
    clean_edges = []
    for edge in edges:
        if edge == float("inf"):
            clean_edges.append(edge)
        elif is_finite_number(edge):
            clean_edges.append(float(edge))
    if len(clean_edges) < 2:
        clean_edges = list(DEFAULT_ENTROPY_BIN_EDGES)
    bins = []
    for lower, upper in zip(clean_edges[:-1], clean_edges[1:]):
        upper_value = None if upper == float("inf") else upper
        if upper_value is None:
            label = f"[{lower:g},inf)"
        else:
            label = f"[{lower:g},{upper_value:g})"
        bins.append({"lower": lower, "upper": upper_value, "label": label})
    return bins


def entropy_bin_for_value(value, bins):
    if not is_finite_number(value):
        return "missing"
    value = float(value)
    for bin_info in bins:
        lower = bin_info["lower"]
        upper = bin_info["upper"]
        if value >= lower and (upper is None or value < upper):
            return bin_info["label"]
    return "out_of_range"


def step_sort_key(step):
    if step is None:
        return (-1, "")
    try:
        return (0, int(step))
    except (TypeError, ValueError):
        return (1, str(step))


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
            "file_name": case.get("file_name", ""),
            "behavior_pattern": case.get("behavior_pattern"),
            "distill_recommendation": case.get("distill_recommendation"),
            "gt_objects": case.get("gt_objects", []),
            "model_mentioned_objects": case.get("model_mentioned_objects", []),
            "baseline_mentioned_objects": case.get("baseline_mentioned_objects", []),
            "model_caption": case.get("model_caption"),
            "baseline_caption": case.get("baseline_caption"),
            "object_level_diff": case.get("object_level_diff", {}),
            "case_diagnostics": case.get("case_diagnostics", {}),
        }
    return groups


def get_image_id(record):
    metadata = record.get("metadata") or {}
    image_id = record.get("image_id") or metadata.get("image_id")
    if image_id is None:
        extra = metadata.get("extra_info")
        if isinstance(extra, dict):
            image_id = extra.get("image_id")
    try:
        return int(image_id)
    except (TypeError, ValueError):
        return None


def get_file_name(record):
    metadata = record.get("metadata") or {}
    file_name = record.get("file_name") or metadata.get("file_name")
    if not file_name:
        extra = metadata.get("extra_info")
        if isinstance(extra, dict):
            file_name = extra.get("file_name")
    if not file_name:
        return None
    return os.path.basename(str(file_name))


def get_caption_source(record):
    metadata = record.get("metadata") or {}
    caption_source = record.get("caption_source") or metadata.get("caption_source")
    if not caption_source:
        extra = metadata.get("extra_info")
        if isinstance(extra, dict):
            caption_source = extra.get("caption_source")
    return caption_source or "train_rollout"


def sample_values(values, limit=20):
    clean_values = [value for value in values if value is not None]
    try:
        return sorted(clean_values)[:limit]
    except TypeError:
        return sorted(str(value) for value in clean_values)[:limit]


def summarize_case_overlap(records, case_groups):
    trace_image_ids = [get_image_id(record) for record in records]
    trace_image_id_set = {image_id for image_id in trace_image_ids if image_id is not None}
    case_image_id_set = set(case_groups.keys())
    image_id_overlap = trace_image_id_set & case_image_id_set

    trace_file_names = [get_file_name(record) for record in records]
    trace_file_name_set = {file_name for file_name in trace_file_names if file_name}
    case_file_name_set = {
        os.path.basename(str(info.get("file_name")))
        for info in case_groups.values()
        if info.get("file_name")
    }
    file_name_overlap = trace_file_name_set & case_file_name_set

    data_source_counter = Counter()
    for record in records:
        metadata = record.get("metadata") or {}
        data_source = metadata.get("data_source")
        if not data_source:
            extra = metadata.get("extra_info")
            if isinstance(extra, dict):
                data_source = extra.get("data_source")
        if data_source:
            data_source_counter[str(data_source)] += 1

    matched_records_by_image_id = sum(1 for image_id in trace_image_ids if image_id in case_image_id_set)
    matched_records_by_file_name = sum(
        1 for file_name in trace_file_names if file_name and file_name in case_file_name_set
    )
    disjoint = bool(case_groups) and not image_id_overlap and not file_name_overlap
    return {
        "case_analysis_loaded": bool(case_groups),
        "trace_num_unique_image_ids": len(trace_image_id_set),
        "case_num_unique_image_ids": len(case_image_id_set),
        "image_id_overlap_count": len(image_id_overlap),
        "image_id_overlap_samples": sample_values(image_id_overlap),
        "trace_image_id_samples": sample_values(trace_image_id_set),
        "case_image_id_samples": sample_values(case_image_id_set),
        "matched_records_by_image_id": matched_records_by_image_id,
        "matched_record_rate_by_image_id": safe_rate(matched_records_by_image_id, len(records)),
        "trace_num_unique_file_names": len(trace_file_name_set),
        "case_num_unique_file_names": len(case_file_name_set),
        "file_name_overlap_count": len(file_name_overlap),
        "file_name_overlap_samples": sample_values(file_name_overlap),
        "trace_file_name_samples": sample_values(trace_file_name_set),
        "case_file_name_samples": sample_values(case_file_name_set),
        "matched_records_by_file_name": matched_records_by_file_name,
        "matched_record_rate_by_file_name": safe_rate(matched_records_by_file_name, len(records)),
        "trace_data_sources": dict(data_source_counter.most_common()),
        "is_disjoint_from_case_analysis": disjoint,
        "likely_reason": (
            "trace records and case analysis appear to come from different splits; "
            "training rollout traces cannot be grouped by eval/test case outcomes"
            if disjoint
            else None
        ),
    }


def canonicalize_object_name(obj, inverse_synonym_dict):
    if obj is None:
        return None
    text = str(obj).strip().lower()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    if text in inverse_synonym_dict:
        return inverse_synonym_dict[text]

    words = [try_singularize(word) for word in text.split()]
    singular = " ".join(words)
    return inverse_synonym_dict.get(singular, singular)


def coerce_object_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, dict):
        for key in ("objects", "gt_objects", "present_objects", "categories"):
            if key in value:
                return coerce_object_list(value.get(key))
        return list(value.keys())
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                return coerce_object_list(parsed)
            except json.JSONDecodeError:
                pass
        if "," in stripped:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return [stripped]
    return [value]


def extract_gt_objects(record, case_groups, inverse_synonym_dict):
    image_id = get_image_id(record)
    metadata = record.get("metadata") or {}
    candidates = []
    extra_info = metadata.get("extra_info")
    for container_name, container in (("metadata", metadata), ("metadata.extra_info", extra_info)):
        if not isinstance(container, dict):
            continue
        for field in ("gt_objects", "objects", "present_objects", "coco_objects", "categories"):
            if field in container:
                candidates.append((f"{container_name}.{field}", container.get(field)))

    case_info = case_groups.get(image_id, {}) if image_id is not None else {}
    if case_info.get("gt_objects"):
        candidates.append(("case_analysis.gt_objects", case_info.get("gt_objects")))

    for source, value in candidates:
        objects = {
            canonicalize_object_name(obj, inverse_synonym_dict)
            for obj in coerce_object_list(value)
        }
        objects = {obj for obj in objects if obj}
        if objects:
            return objects, source
    return set(), "missing"


def plural_variants(term):
    words = term.split()
    if not words:
        return set()

    last = words[-1]
    variants = set()
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        variants.add(" ".join(words[:-1] + [last[:-1] + "ies"]))
    elif last.endswith(("s", "x", "z", "ch", "sh")):
        variants.add(" ".join(words[:-1] + [last + "es"]))
    elif not last.endswith("s"):
        variants.add(" ".join(words[:-1] + [last + "s"]))

    irregular_plurals = {
        "person": "people",
        "man": "men",
        "woman": "women",
        "child": "children",
        "mouse": "mice",
        "knife": "knives",
    }
    if last in irregular_plurals:
        variants.add(" ".join(words[:-1] + [irregular_plurals[last]]))
    return variants


def build_canonical_synonym_map(inverse_synonym_dict):
    canonical_to_terms = defaultdict(set)
    for term, canonical in inverse_synonym_dict.items():
        canonical_to_terms[canonical].add(term)
        singular = " ".join(try_singularize(part) for part in term.split())
        canonical_to_terms[canonical].add(singular)
        canonical_to_terms[canonical].update(plural_variants(term))
        canonical_to_terms[canonical].update(plural_variants(singular))
    return {
        canonical: sorted({term for term in terms if term}, key=lambda term: (-len(term), term))
        for canonical, terms in canonical_to_terms.items()
    }


def build_token_char_spans(record):
    token_records = record.get("token_records", []) or []
    text_parts = []
    spans = []
    cursor = 0
    for token in token_records:
        token_text = token.get("token_text")
        if token_text is None:
            token_text = ""
        token_text = str(token_text)
        token_text = token_text.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\n")
        start = cursor
        text_parts.append(token_text)
        cursor += len(token_text)
        spans.append((start, cursor))
    token_text = "".join(text_parts)
    return token_text, spans


def overlaps(span_a, span_b):
    return span_a[0] < span_b[1] and span_b[0] < span_a[1]


def token_indices_for_char_span(token_spans, char_start, char_end):
    indices = []
    for idx, (token_start, token_end) in enumerate(token_spans):
        if token_start < char_end and char_start < token_end:
            indices.append(idx)
    return indices


def bounded_context(text, start, end, window=80):
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].replace("\n", " ")
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet += "..."
    return snippet


def classify_object_role(canonical, object_type, caption_source, case_info):
    object_diff = case_info.get("object_level_diff") or {}
    if not object_diff:
        return object_type

    def has(field):
        return canonical in set(object_diff.get(field, []) or [])

    if caption_source == "model_caption":
        if has("model_only_hallucinations"):
            return "model_only_hallucination"
        if has("shared_hallucinations"):
            return "persistent_hallucination"
        if has("model_gained_gt"):
            return "model_gained_gt"
        if has("model_hit_gt"):
            return "model_correct_object"
        if has("model_hallucinated"):
            return "model_hallucinated_object"
        return object_type

    if caption_source == "baseline_caption":
        if has("removed_hallucinations"):
            return "removed_hallucination_candidate"
        if has("shared_hallucinations"):
            return "persistent_hallucination"
        if has("model_lost_gt"):
            return "lost_gt_candidate"
        if has("baseline_hit_gt"):
            return "baseline_correct_object"
        if has("baseline_hallucinated"):
            return "baseline_hallucinated_object"
        return object_type

    return object_type


def metric_values(tokens, key):
    values = []
    for token in tokens:
        if key == "_teacher_minus_student_selected_logprob":
            value = token_delta(token)
        elif key == "_teacher_minus_student_entropy":
            teacher = token.get("teacher_entropy")
            student = token.get("student_entropy")
            value = teacher - student if teacher is not None and student is not None else None
        elif key == "top1_match":
            raw = token.get("top1_match")
            value = 1.0 if raw is True else 0.0 if raw is False else None
        else:
            value = token.get(key)
        if value is not None:
            values.append(float(value))
    return values


def summarize_mention_metrics(tokens):
    student_logps = metric_values(tokens, "student_selected_logprob")
    teacher_logps = metric_values(tokens, "teacher_selected_logprob")
    logp_deltas = metric_values(tokens, "_teacher_minus_student_selected_logprob")
    student_entropy = metric_values(tokens, "student_entropy")
    teacher_entropy = metric_values(tokens, "teacher_entropy")
    entropy_deltas = metric_values(tokens, "_teacher_minus_student_entropy")

    result = {
        "student_selected_logprob_mean": safe_mean(student_logps),
        "teacher_selected_logprob_mean": safe_mean(teacher_logps),
        "teacher_minus_student_selected_logprob_mean": safe_mean(logp_deltas),
        "student_selected_logprob_sum": sum(student_logps) if student_logps else None,
        "teacher_selected_logprob_sum": sum(teacher_logps) if teacher_logps else None,
        "teacher_minus_student_selected_logprob_sum": sum(logp_deltas) if logp_deltas else None,
        "student_entropy_mean": safe_mean(student_entropy),
        "teacher_entropy_mean": safe_mean(teacher_entropy),
        "teacher_minus_student_entropy_mean": safe_mean(entropy_deltas),
        "student_topk_mass_mean": safe_mean(metric_values(tokens, "student_topk_mass")),
        "teacher_topk_mass_mean": safe_mean(metric_values(tokens, "teacher_topk_mass")),
        "student_top1_top2_margin_mean": safe_mean(metric_values(tokens, "student_top1_top2_margin")),
        "teacher_top1_top2_margin_mean": safe_mean(metric_values(tokens, "teacher_top1_top2_margin")),
        "topk_overlap_ratio_mean": safe_mean(metric_values(tokens, "topk_overlap_ratio")),
        "topk_jaccard_mean": safe_mean(metric_values(tokens, "topk_jaccard")),
        "top1_match_frac": safe_mean(metric_values(tokens, "top1_match")),
        "selected_token_rank_in_student_topk_mean": safe_mean(
            metric_values(tokens, "selected_token_rank_in_student_topk")
        ),
        "selected_token_rank_in_teacher_topk_mean": safe_mean(
            metric_values(tokens, "selected_token_rank_in_teacher_topk")
        ),
    }
    teacher_rank_presence = [
        1.0 if token.get("selected_token_rank_in_teacher_topk") is not None else 0.0
        for token in tokens
    ]
    result["selected_token_in_teacher_topk_frac"] = safe_mean(teacher_rank_presence)
    result["teacher_selected_logprob_lt_student_frac"] = safe_rate(
        len([delta for delta in logp_deltas if delta < 0]),
        len(logp_deltas),
    )

    token_metric_sums = {}
    token_metric_counts = {}
    for key in OBJECT_TOKEN_METRIC_KEYS:
        values = metric_values(tokens, key)
        token_metric_sums[key] = sum(values) if values else None
        token_metric_counts[key] = len(values)
    result["_token_metric_sums"] = token_metric_sums
    result["_token_metric_counts"] = token_metric_counts
    return result


def find_object_mentions(record, canonical_synonym_map, gt_objects, gt_source, case_group, case_info):
    token_text, token_spans = build_token_char_spans(record)
    search_text = token_text or record.get("response_text", "")
    if not search_text:
        return []
    lowered = search_text.lower()
    token_records = record.get("token_records", []) or []
    mentions = []
    selected_spans_by_canonical = defaultdict(list)
    caption_source = get_caption_source(record)
    metadata = record.get("metadata") or {}
    behavior_pattern = metadata.get("behavior_pattern") or case_info.get("behavior_pattern")
    distill_recommendation = metadata.get("distill_recommendation") or case_info.get("distill_recommendation")

    candidates = []
    for canonical, terms in canonical_synonym_map.items():
        for term in terms:
            if term:
                candidates.append((canonical, term))
    candidates.sort(key=lambda item: (-len(item[1]), item[0], item[1]))

    for canonical, term in candidates:
        escaped_term = re.escape(term).replace(r"\ ", r"\s+")
        pattern = r"(?<![a-z0-9])" + escaped_term + r"(?![a-z0-9])"
        for match in re.finditer(pattern, lowered):
            char_span = (match.start(), match.end())
            if any(overlaps(char_span, existing) for existing in selected_spans_by_canonical[canonical]):
                continue
            token_indices = token_indices_for_char_span(token_spans, match.start(), match.end())
            if not token_indices:
                continue
            selected_spans_by_canonical[canonical].append(char_span)
            tokens = [token_records[idx] for idx in token_indices if idx < len(token_records)]
            if not tokens:
                continue
            if gt_objects:
                object_type = "correct_object" if canonical in gt_objects else "hallucinated_object"
            else:
                object_type = "unknown_object"
            object_role = classify_object_role(canonical, object_type, caption_source, case_info)

            metrics = summarize_mention_metrics(tokens)
            mention = {
                "image_id": get_image_id(record),
                "global_step": record.get("global_step"),
                "trace_scope": record.get("trace_scope", "train"),
                "caption_source": caption_source,
                "rank": record.get("rank"),
                "sample_index_in_rank_batch": record.get("sample_index_in_rank_batch"),
                "case_group": case_group,
                "behavior_pattern": behavior_pattern,
                "distill_recommendation": distill_recommendation,
                "canonical_object": canonical,
                "source_term": term,
                "mention_text": search_text[match.start():match.end()],
                "context": bounded_context(search_text, match.start(), match.end()),
                "object_type": object_type,
                "object_role": object_role,
                "gt_source": gt_source,
                "char_start": match.start(),
                "char_end": match.end(),
                "token_start": min(token_indices),
                "token_end": max(token_indices) + 1,
                "num_tokens": len(tokens),
                "token_text": "".join(str(token.get("token_text", "")) for token in tokens),
                "_trace_file": record.get("_trace_file"),
            }
            mention.update(metrics)
            mentions.append(mention)

    mentions.sort(
        key=lambda item: (
            item.get("char_start", 0),
            -(item.get("char_end", 0) - item.get("char_start", 0)),
            item.get("canonical_object", ""),
        )
    )
    return mentions


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


def clean_mention_for_output(mention):
    keep_keys = [
        "image_id",
        "global_step",
        "trace_scope",
        "caption_source",
        "rank",
        "sample_index_in_rank_batch",
        "case_group",
        "behavior_pattern",
        "distill_recommendation",
        "canonical_object",
        "source_term",
        "mention_text",
        "context",
        "object_type",
        "object_role",
        "gt_source",
        "char_start",
        "char_end",
        "token_start",
        "token_end",
        "num_tokens",
        "token_text",
        "_trace_file",
    ] + OBJECT_METRIC_KEYS
    return {key: mention.get(key) for key in keep_keys if key in mention}


def summarize_mention_rows(rows):
    object_counter = Counter(row.get("canonical_object") for row in rows if row.get("canonical_object"))
    source_term_counter = Counter(row.get("source_term") for row in rows if row.get("source_term"))
    gt_source_counter = Counter(row.get("gt_source") for row in rows if row.get("gt_source"))
    case_group_counter = Counter(row.get("case_group") for row in rows if row.get("case_group"))
    caption_source_counter = Counter(row.get("caption_source") for row in rows if row.get("caption_source"))
    object_role_counter = Counter(row.get("object_role") for row in rows if row.get("object_role"))
    trace_files = {row.get("_trace_file") for row in rows if row.get("_trace_file")}
    image_ids = {row.get("image_id") for row in rows if row.get("image_id") is not None}
    record_keys = {
        (
            row.get("image_id"),
            row.get("global_step"),
            row.get("rank"),
            row.get("sample_index_in_rank_batch"),
            row.get("_trace_file"),
        )
        for row in rows
    }
    valid_record_keys = {key for key in record_keys if any(value is not None for value in key)}

    summary = {
        "num_mentions": len(rows),
        "num_mention_tokens": sum(row.get("num_tokens", 0) or 0 for row in rows),
        "num_unique_images": len(image_ids),
        "num_unique_records": len(valid_record_keys),
        "num_unique_objects": len(object_counter),
        "num_trace_files": len(trace_files),
        "mean_tokens_per_mention": safe_mean(row.get("num_tokens") for row in rows),
        "top_objects": dict(object_counter.most_common(30)),
        "top_source_terms": dict(source_term_counter.most_common(30)),
        "gt_source_counts": dict(gt_source_counter.most_common()),
        "case_group_counts": dict(case_group_counter.most_common()),
        "caption_source_counts": dict(caption_source_counter.most_common()),
        "object_role_counts": dict(object_role_counter.most_common()),
    }
    for key in OBJECT_METRIC_KEYS:
        summary[key] = safe_mean(row.get(key) for row in rows)

    summary["teacher_penalized_mention_frac"] = safe_rate(
        len(
            [
                row
                for row in rows
                if row.get("teacher_minus_student_selected_logprob_sum") is not None
                and row.get("teacher_minus_student_selected_logprob_sum") < 0
            ]
        ),
        len([row for row in rows if row.get("teacher_minus_student_selected_logprob_sum") is not None]),
    )

    token_weighted = {}
    for token_key in OBJECT_TOKEN_METRIC_KEYS:
        total = 0.0
        count = 0
        for row in rows:
            sums = row.get("_token_metric_sums") or {}
            counts = row.get("_token_metric_counts") or {}
            value_sum = sums.get(token_key)
            value_count = counts.get(token_key, 0) or 0
            if value_sum is None or value_count <= 0:
                continue
            total += value_sum
            count += value_count
        if count:
            out_key = token_key.strip("_").replace("_teacher_minus_student", "teacher_minus_student")
            token_weighted[f"{out_key}_token_weighted_mean"] = total / count
    summary["token_weighted_metrics"] = token_weighted
    return summary


def diff_summary(left, right, metric_keys):
    if not left or not right:
        return {}
    diff = {}
    for key in metric_keys:
        left_value = left.get(key)
        right_value = right.get(key)
        diff[f"{key}_diff"] = (
            left_value - right_value
            if left_value is not None and right_value is not None
            else None
        )
    return diff


def top_mentions(rows, key, reverse=False, limit=30, object_type=None):
    filtered = [
        row for row in rows
        if row.get(key) is not None and (object_type is None or row.get("object_type") == object_type)
    ]
    filtered.sort(key=lambda row: row.get(key), reverse=reverse)
    return [clean_mention_for_output(row) for row in filtered[:limit]]


def top_mentions_where(rows, predicate, key, reverse=False, limit=30):
    filtered = [row for row in rows if row.get(key) is not None and predicate(row)]
    filtered.sort(key=lambda row: row.get(key), reverse=reverse)
    return [clean_mention_for_output(row) for row in filtered[:limit]]


def summarize_by_object(rows):
    by_object_type = defaultdict(list)
    for row in rows:
        object_type = row.get("object_type", "unknown_object")
        canonical = row.get("canonical_object", "unknown")
        by_object_type[(object_type, canonical)].append(row)

    object_summary = {}
    for (object_type, canonical), obj_rows in sorted(by_object_type.items()):
        object_summary.setdefault(object_type, {})[canonical] = summarize_mention_rows(obj_rows)
    return object_summary


def summarize_by_field(rows, field_name):
    by_value = defaultdict(list)
    for row in rows:
        by_value[row.get(field_name, "unknown")].append(row)
    return {
        str(value): summarize_mention_rows(value_rows)
        for value, value_rows in sorted(by_value.items(), key=lambda item: str(item[0]))
    }


def summarize_by_step(rows):
    by_step_type = defaultdict(list)
    for row in rows:
        step = row.get("global_step")
        object_type = row.get("object_type", "unknown_object")
        by_step_type[(step, object_type)].append(row)

    by_step = {}
    for (step, object_type), step_rows in sorted(by_step_type.items(), key=lambda item: step_sort_key(item[0][0])):
        step_key = str(step)
        by_step.setdefault(step_key, {})[object_type] = summarize_mention_rows(step_rows)

    contrast_keys = [
        "teacher_minus_student_selected_logprob_mean",
        "teacher_minus_student_selected_logprob_sum",
        "teacher_selected_logprob_lt_student_frac",
        "teacher_penalized_mention_frac",
        "student_entropy_mean",
        "teacher_entropy_mean",
        "teacher_minus_student_entropy_mean",
        "topk_overlap_ratio_mean",
        "top1_match_frac",
    ]
    step_contrast = {}
    for step_key, step_summary in by_step.items():
        hallucinated = step_summary.get("hallucinated_object")
        correct = step_summary.get("correct_object")
        if hallucinated and correct:
            step_contrast[step_key] = {
                "hallucinated_minus_correct": diff_summary(hallucinated, correct, contrast_keys),
                "hallucinated_num_mentions": hallucinated.get("num_mentions", 0),
                "correct_num_mentions": correct.get("num_mentions", 0),
            }
    return by_step, step_contrast


def summarize_entropy_bins(rows, entropy_bins):
    by_bin_type = defaultdict(list)
    for row in rows:
        object_type = row.get("object_type", "unknown_object")
        bin_label = entropy_bin_for_value(row.get("student_entropy_mean"), entropy_bins)
        by_bin_type[(bin_label, object_type)].append(row)

    bin_labels = [bin_info["label"] for bin_info in entropy_bins] + ["missing", "out_of_range"]
    by_bin = {}
    for bin_label in bin_labels:
        type_summary = {}
        for object_type in ("correct_object", "hallucinated_object", "unknown_object"):
            type_rows = by_bin_type.get((bin_label, object_type), [])
            if type_rows:
                type_summary[object_type] = summarize_mention_rows(type_rows)
        if type_summary:
            by_bin[bin_label] = type_summary

    contrast_keys = [
        "teacher_minus_student_selected_logprob_mean",
        "teacher_minus_student_selected_logprob_sum",
        "teacher_selected_logprob_lt_student_frac",
        "teacher_penalized_mention_frac",
        "student_entropy_mean",
        "teacher_entropy_mean",
        "teacher_minus_student_entropy_mean",
        "topk_overlap_ratio_mean",
        "top1_match_frac",
    ]
    contrast = {}
    for bin_label, type_summary in by_bin.items():
        hallucinated = type_summary.get("hallucinated_object")
        correct = type_summary.get("correct_object")
        if hallucinated and correct:
            contrast[bin_label] = {
                "hallucinated_minus_correct": diff_summary(hallucinated, correct, contrast_keys),
                "hallucinated_num_mentions": hallucinated.get("num_mentions", 0),
                "correct_num_mentions": correct.get("num_mentions", 0),
            }

    return {
        "bin_edges": [
            ("inf" if edge == float("inf") else edge)
            for edge in [entropy_bins[0]["lower"]] + [
                (bin_info["upper"] if bin_info["upper"] is not None else float("inf"))
                for bin_info in entropy_bins
            ]
        ] if entropy_bins else [],
        "by_bin": by_bin,
        "hallucinated_minus_correct_by_bin": contrast,
    }


def summarize_delta_distributions(rows):
    by_type = defaultdict(list)
    for row in rows:
        by_type[row.get("object_type", "unknown_object")].append(row)

    output = {}
    for field in DELTA_DISTRIBUTION_FIELDS:
        field_summary = {}
        for object_type in ("correct_object", "hallucinated_object", "unknown_object"):
            type_rows = by_type.get(object_type, [])
            if type_rows:
                field_summary[object_type] = summarize_value_distribution(
                    type_rows,
                    field,
                    rate_thresholds=DELTA_RATE_THRESHOLDS,
                )

        correct = field_summary.get("correct_object")
        hallucinated = field_summary.get("hallucinated_object")
        if correct and hallucinated:
            diff = {}
            for key in ["mean", "min", "max"] + [f"p{p}" for p in DELTA_PERCENTILES]:
                left = hallucinated.get(key)
                right = correct.get(key)
                diff[f"{key}_diff"] = left - right if left is not None and right is not None else None
            for threshold in DELTA_RATE_THRESHOLDS:
                label = str(threshold).replace("-", "neg").replace(".", "p")
                key = f"frac_lt_{label}"
                left = hallucinated.get(key)
                right = correct.get(key)
                diff[f"{key}_diff"] = left - right if left is not None and right is not None else None
            field_summary["hallucinated_minus_correct"] = diff
        output[field] = field_summary
    return output


def summarize_gate_sweep(rows, entropy_thresholds, logp_margins):
    labeled_rows = [
        row for row in rows
        if row.get("object_type") in {"correct_object", "hallucinated_object"}
        and is_finite_number(row.get("student_entropy_mean"))
        and is_finite_number(row.get("teacher_minus_student_selected_logprob_mean"))
    ]
    total_hallucinated = len([row for row in labeled_rows if row.get("object_type") == "hallucinated_object"])
    total_correct = len([row for row in labeled_rows if row.get("object_type") == "correct_object"])
    total_labeled = total_hallucinated + total_correct
    base_precision = safe_rate(total_hallucinated, total_labeled)

    gates = []
    for entropy_threshold in entropy_thresholds:
        for logp_margin in logp_margins:
            selected = [
                row for row in labeled_rows
                if row.get("student_entropy_mean") > entropy_threshold
                and row.get("teacher_minus_student_selected_logprob_mean") < -logp_margin
            ]
            selected_hallucinated = len(
                [row for row in selected if row.get("object_type") == "hallucinated_object"]
            )
            selected_correct = len([row for row in selected if row.get("object_type") == "correct_object"])
            selected_total = selected_hallucinated + selected_correct
            precision = safe_rate(selected_hallucinated, selected_total)
            recall = safe_rate(selected_hallucinated, total_hallucinated)
            correct_false_positive_rate = safe_rate(selected_correct, total_correct)
            f1 = safe_rate(2 * precision * recall, precision + recall)
            gates.append(
                {
                    "student_entropy_gt": entropy_threshold,
                    "teacher_minus_student_logp_lt": -logp_margin,
                    "selected_total": selected_total,
                    "selected_hallucinated": selected_hallucinated,
                    "selected_correct": selected_correct,
                    "hallucination_precision": precision,
                    "hallucination_recall": recall,
                    "correct_false_positive_rate": correct_false_positive_rate,
                    "f1": f1,
                    "precision_lift_vs_base": precision / base_precision if base_precision > 0 else None,
                }
            )

    def gate_rank(gate):
        return (
            gate.get("f1") or 0.0,
            gate.get("hallucination_precision") or 0.0,
            gate.get("hallucination_recall") or 0.0,
            -(gate.get("correct_false_positive_rate") or 0.0),
        )

    def precision_rank(gate):
        return (
            gate.get("hallucination_precision") or 0.0,
            gate.get("hallucination_recall") or 0.0,
            -(gate.get("correct_false_positive_rate") or 0.0),
        )

    return {
        "num_labeled_mentions": total_labeled,
        "num_hallucinated_mentions": total_hallucinated,
        "num_correct_mentions": total_correct,
        "base_hallucination_rate": base_precision,
        "entropy_thresholds": entropy_thresholds,
        "logp_margins": logp_margins,
        "all_gates": gates,
        "top_by_f1": sorted(gates, key=gate_rank, reverse=True)[:30],
        "top_precision_recall_ge_0p1": sorted(
            [gate for gate in gates if gate.get("hallucination_recall", 0.0) >= 0.1],
            key=precision_rank,
            reverse=True,
        )[:30],
    }


def build_quadrant_config(**overrides):
    config = dict(DEFAULT_QUADRANT_CONFIG)
    for key, value in overrides.items():
        if value is not None:
            config[key] = float(value)
    return config


def lowres_reliable(row, config):
    teacher_entropy = row.get("teacher_entropy_mean")
    teacher_margin = row.get("teacher_top1_top2_margin_mean")
    entropy_ok = (
        is_finite_number(teacher_entropy)
        and float(teacher_entropy) <= config["teacher_entropy_max"]
    )
    margin_ok = (
        not is_finite_number(teacher_margin)
        or float(teacher_margin) >= config["teacher_margin_min"]
    )
    return entropy_ok and margin_ok


def lowres_uncertain(row, config):
    teacher_entropy = row.get("teacher_entropy_mean")
    teacher_margin = row.get("teacher_top1_top2_margin_mean")
    entropy_uncertain = (
        not is_finite_number(teacher_entropy)
        or float(teacher_entropy) > config["teacher_entropy_max"]
    )
    margin_uncertain = (
        is_finite_number(teacher_margin)
        and float(teacher_margin) < config["teacher_margin_min"]
    )
    return entropy_uncertain or margin_uncertain


def lowres_support_evidence(row, config):
    delta = row.get("teacher_minus_student_selected_logprob_mean")
    if not is_finite_number(delta) or float(delta) < config["support_delta_min"]:
        return False

    rank = row.get("selected_token_rank_in_teacher_topk_mean")
    top1 = row.get("top1_match_frac")
    in_topk = row.get("selected_token_in_teacher_topk_frac")
    rank_ok = is_finite_number(rank) and float(rank) <= config["support_rank_max"]
    top1_ok = is_finite_number(top1) and float(top1) >= config["support_top1_min"]
    in_topk_ok = is_finite_number(in_topk) and float(in_topk) >= 0.8
    return rank_ok or top1_ok or in_topk_ok


def lowres_reject_evidence(row, config):
    delta = row.get("teacher_minus_student_selected_logprob_mean")
    if not is_finite_number(delta) or float(delta) > config["reject_delta_max"]:
        return False

    rank = row.get("selected_token_rank_in_teacher_topk_mean")
    top1 = row.get("top1_match_frac")
    in_topk = row.get("selected_token_in_teacher_topk_frac")
    rank_bad = not is_finite_number(rank) or float(rank) >= config["reject_rank_min"]
    top1_bad = is_finite_number(top1) and float(top1) <= config["reject_top1_max"]
    in_topk_bad = (
        not is_finite_number(in_topk)
        or float(in_topk) <= config["reject_in_teacher_topk_max"]
    )
    return rank_bad or top1_bad or in_topk_bad


def classify_lowres_quadrant(row, config):
    reliable = lowres_reliable(row, config)
    uncertain = lowres_uncertain(row, config)
    support = reliable and lowres_support_evidence(row, config)
    reject = reliable and lowres_reject_evidence(row, config)

    if support and not reject:
        return "lowres_confident_support"
    if reject and not support:
        return "lowres_confident_reject"
    if support and reject:
        return "lowres_conflicting"
    if uncertain:
        return "lowres_uncertain"
    return "lowres_ambiguous"


def summarize_lowres_quadrants(rows, config):
    labeled_rows = [
        row for row in rows
        if row.get("object_type") in {"correct_object", "hallucinated_object"}
    ]
    total_correct = len([row for row in labeled_rows if row.get("object_type") == "correct_object"])
    total_hallucinated = len([row for row in labeled_rows if row.get("object_type") == "hallucinated_object"])
    total_labeled = total_correct + total_hallucinated
    base_hallucination_rate = safe_rate(total_hallucinated, total_labeled)

    by_bucket = defaultdict(list)
    for row in rows:
        row = dict(row)
        bucket = classify_lowres_quadrant(row, config)
        row["lowres_quadrant"] = bucket
        by_bucket[bucket].append(row)

    bucket_summary = {}
    preferred_order = [
        "lowres_confident_support",
        "lowres_confident_reject",
        "lowres_uncertain",
        "lowres_ambiguous",
        "lowres_conflicting",
    ]
    metric_keys = [
        "teacher_minus_student_selected_logprob_mean",
        "teacher_minus_student_selected_logprob_sum",
        "teacher_selected_logprob_lt_student_frac",
        "teacher_penalized_mention_frac",
        "student_entropy_mean",
        "teacher_entropy_mean",
        "teacher_top1_top2_margin_mean",
        "selected_token_rank_in_teacher_topk_mean",
        "selected_token_in_teacher_topk_frac",
        "top1_match_frac",
        "topk_jaccard_mean",
    ]
    for bucket in preferred_order + sorted(set(by_bucket) - set(preferred_order)):
        bucket_rows = by_bucket.get(bucket, [])
        if not bucket_rows:
            continue
        correct_rows = [row for row in bucket_rows if row.get("object_type") == "correct_object"]
        hallucinated_rows = [
            row for row in bucket_rows if row.get("object_type") == "hallucinated_object"
        ]
        unknown_rows = [row for row in bucket_rows if row.get("object_type") == "unknown_object"]
        labeled_count = len(correct_rows) + len(hallucinated_rows)
        hallucination_rate = safe_rate(len(hallucinated_rows), labeled_count)
        bucket_summary[bucket] = {
            "num_mentions": len(bucket_rows),
            "num_labeled_mentions": labeled_count,
            "num_correct_mentions": len(correct_rows),
            "num_hallucinated_mentions": len(hallucinated_rows),
            "num_unknown_mentions": len(unknown_rows),
            "hallucination_rate": hallucination_rate,
            "precision_lift_vs_base": (
                hallucination_rate / base_hallucination_rate
                if base_hallucination_rate > 0
                else None
            ),
            "hallucinated_recall": safe_rate(len(hallucinated_rows), total_hallucinated),
            "correct_capture_rate": safe_rate(len(correct_rows), total_correct),
            "correct_false_positive_rate": safe_rate(len(correct_rows), total_correct),
            "correct_to_hallucinated_ratio": safe_rate(len(correct_rows), len(hallucinated_rows)),
            "metrics": {
                key: safe_mean(row.get(key) for row in bucket_rows)
                for key in metric_keys
            },
            "by_type": {
                object_type: summarize_mention_rows(type_rows)
                for object_type, type_rows in (
                    ("correct_object", correct_rows),
                    ("hallucinated_object", hallucinated_rows),
                    ("unknown_object", unknown_rows),
                )
                if type_rows
            },
        }

    support = bucket_summary.get("lowres_confident_support", {})
    reject = bucket_summary.get("lowres_confident_reject", {})
    return {
        "config": config,
        "num_labeled_mentions": total_labeled,
        "num_correct_mentions": total_correct,
        "num_hallucinated_mentions": total_hallucinated,
        "base_hallucination_rate": base_hallucination_rate,
        "bucket_summary": bucket_summary,
        "interpretation": {
            "support_bucket_correct_enrichment": (
                support.get("num_correct_mentions", 0)
                / support.get("num_labeled_mentions", 1)
                if support.get("num_labeled_mentions")
                else None
            ),
            "reject_bucket_hallucination_enrichment": reject.get("hallucination_rate"),
            "reject_bucket_precision_lift_vs_base": reject.get("precision_lift_vs_base"),
            "support_bucket_hallucination_rate": support.get("hallucination_rate"),
            "support_bucket_correct_capture_rate": support.get("correct_capture_rate"),
            "reject_bucket_hallucinated_recall": reject.get("hallucinated_recall"),
            "reject_bucket_correct_false_positive_rate": reject.get("correct_false_positive_rate"),
        },
    }


def build_selective_suppression_signal(type_summary):
    hallucinated = type_summary.get("hallucinated_object")
    correct = type_summary.get("correct_object")
    if not hallucinated or not correct:
        return {
            "available": False,
            "reason": "Need both hallucinated_object and correct_object mentions with GT labels.",
        }

    contrast_keys = [
        "teacher_minus_student_selected_logprob_mean",
        "teacher_minus_student_selected_logprob_sum",
        "teacher_selected_logprob_lt_student_frac",
        "teacher_penalized_mention_frac",
        "student_entropy_mean",
        "teacher_entropy_mean",
        "teacher_minus_student_entropy_mean",
        "topk_overlap_ratio_mean",
        "top1_match_frac",
    ]
    contrast = diff_summary(hallucinated, correct, contrast_keys)
    logp_gap = contrast.get("teacher_minus_student_selected_logprob_mean_diff")
    penalized_gap = contrast.get("teacher_selected_logprob_lt_student_frac_diff")
    mention_penalty_gap = contrast.get("teacher_penalized_mention_frac_diff")
    return {
        "available": True,
        "interpretation": (
            "Positive evidence means hallucinated objects have more negative teacher-student "
            "logprob deltas, or a higher teacher<student penalty rate, than correct objects."
        ),
        "hallucinated_minus_correct": contrast,
        "selective_logprob_suppression_signal": logp_gap is not None and logp_gap < 0,
        "selective_teacher_lt_student_signal": penalized_gap is not None and penalized_gap > 0,
        "selective_mention_penalty_signal": mention_penalty_gap is not None and mention_penalty_gap > 0,
    }


def summarize_object_trace(
    records,
    case_groups,
    entropy_bins,
    gate_entropy_thresholds,
    gate_logp_margins,
    quadrant_config=None,
):
    _, inverse_synonym_dict = parse_official_synonyms()
    canonical_synonym_map = build_canonical_synonym_map(inverse_synonym_dict)

    rows = []
    gt_source_counts = Counter()
    records_with_gt = 0
    records_without_gt = 0
    records_with_mentions = 0
    records_without_mentions = 0
    record_type_counts = Counter()
    record_role_counts = Counter()
    caption_source_counts = Counter()

    for record in records:
        image_id = get_image_id(record)
        case_info = case_groups.get(image_id, {}) if image_id is not None else {}
        case_group = case_info.get("group", "unmatched_or_no_case")
        caption_source_counts[get_caption_source(record)] += 1
        gt_objects, gt_source = extract_gt_objects(record, case_groups, inverse_synonym_dict)
        gt_source_counts[gt_source] += 1
        if gt_objects:
            records_with_gt += 1
        else:
            records_without_gt += 1

        mentions = find_object_mentions(record, canonical_synonym_map, gt_objects, gt_source, case_group, case_info)
        if mentions:
            records_with_mentions += 1
        else:
            records_without_mentions += 1
        for mention in mentions:
            record_type_counts[mention.get("object_type", "unknown_object")] += 1
            record_role_counts[mention.get("object_role", "unknown_role")] += 1
        rows.extend(mentions)

    by_type = defaultdict(list)
    for row in rows:
        by_type[row.get("object_type", "unknown_object")].append(row)
    mention_type_summary = {
        object_type: summarize_mention_rows(type_rows)
        for object_type, type_rows in sorted(by_type.items())
    }
    by_step, step_contrast = summarize_by_step(rows)

    correct_rows = [row for row in rows if row.get("object_type") == "correct_object"]
    hallucinated_rows = [row for row in rows if row.get("object_type") == "hallucinated_object"]
    quadrant_config = quadrant_config or build_quadrant_config()

    return {
        "num_object_mentions": len(rows),
        "records_with_gt_count": records_with_gt,
        "records_without_gt_count": records_without_gt,
        "records_with_object_mentions_count": records_with_mentions,
        "records_without_object_mentions_count": records_without_mentions,
        "gt_source_counts": dict(gt_source_counts.most_common()),
        "caption_source_counts": dict(caption_source_counts.most_common()),
        "mention_counts_by_type": dict(record_type_counts.most_common()),
        "mention_counts_by_role": dict(record_role_counts.most_common()),
        "mention_type_summary": mention_type_summary,
        "mention_role_summary": summarize_by_field(rows, "object_role"),
        "mention_caption_source_summary": summarize_by_field(rows, "caption_source"),
        "correct_vs_hallucinated_signal": build_selective_suppression_signal(mention_type_summary),
        "entropy_bin_summary": summarize_entropy_bins(rows, entropy_bins),
        "logp_delta_distribution": summarize_delta_distributions(rows),
        "gate_sweep": summarize_gate_sweep(rows, gate_entropy_thresholds, gate_logp_margins),
        "lowres_quadrant_summary": summarize_lowres_quadrants(rows, quadrant_config),
        "step_summary_by_type": by_step,
        "step_contrast_hallucinated_minus_correct": step_contrast,
        "object_summary_by_type": summarize_by_object(rows),
        "top_teacher_suppressed_hallucination_mentions": top_mentions(
            hallucinated_rows,
            "teacher_minus_student_selected_logprob_sum",
            reverse=False,
            limit=30,
        ),
        "top_teacher_suppressed_correct_mentions": top_mentions(
            correct_rows,
            "teacher_minus_student_selected_logprob_sum",
            reverse=False,
            limit=30,
        ),
        "top_teacher_supported_correct_mentions": top_mentions(
            correct_rows,
            "teacher_minus_student_selected_logprob_sum",
            reverse=True,
            limit=30,
        ),
        "top_teacher_supported_hallucination_mentions": top_mentions(
            hallucinated_rows,
            "teacher_minus_student_selected_logprob_sum",
            reverse=True,
            limit=30,
        ),
        "top_teacher_suppressed_removed_hallucination_candidates": top_mentions_where(
            rows,
            lambda row: row.get("object_role") == "removed_hallucination_candidate",
            "teacher_minus_student_selected_logprob_sum",
            reverse=False,
            limit=30,
        ),
        "top_teacher_suppressed_lost_gt_candidates": top_mentions_where(
            rows,
            lambda row: row.get("object_role") == "lost_gt_candidate",
            "teacher_minus_student_selected_logprob_sum",
            reverse=False,
            limit=30,
        ),
    }


def fmt_value(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        clean_row = [str(value).replace("\n", " ").replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(clean_row) + " |")
    return "\n".join(lines)


def mention_summary_table_rows(type_summary):
    rows = []
    for object_type in ("correct_object", "hallucinated_object", "unknown_object"):
        stats = type_summary.get(object_type)
        if not stats:
            continue
        rows.append(
            [
                object_type,
                stats.get("num_mentions", 0),
                stats.get("num_unique_images", 0),
                fmt_value(stats.get("teacher_minus_student_selected_logprob_mean")),
                fmt_value(stats.get("teacher_minus_student_selected_logprob_sum")),
                fmt_value(stats.get("teacher_selected_logprob_lt_student_frac")),
                fmt_value(stats.get("teacher_penalized_mention_frac")),
                fmt_value(stats.get("student_entropy_mean")),
                fmt_value(stats.get("teacher_entropy_mean")),
                fmt_value(stats.get("topk_overlap_ratio_mean")),
                fmt_value(stats.get("top1_match_frac")),
            ]
        )
    return rows


def generic_summary_table_rows(summary, preferred_order=None):
    rows = []
    seen = set()
    keys = list(preferred_order or []) + sorted(summary.keys())
    for key in keys:
        if key in seen or key not in summary:
            continue
        seen.add(key)
        stats = summary.get(key) or {}
        rows.append(
            [
                key,
                stats.get("num_mentions", 0),
                stats.get("num_unique_images", 0),
                fmt_value(stats.get("teacher_minus_student_selected_logprob_mean")),
                fmt_value(stats.get("teacher_minus_student_selected_logprob_sum")),
                fmt_value(stats.get("teacher_selected_logprob_lt_student_frac")),
                fmt_value(stats.get("teacher_penalized_mention_frac")),
                fmt_value(stats.get("student_entropy_mean")),
                fmt_value(stats.get("teacher_entropy_mean")),
                fmt_value(stats.get("topk_overlap_ratio_mean")),
                fmt_value(stats.get("top1_match_frac")),
            ]
        )
    return rows


def step_contrast_table_rows(step_contrast):
    rows = []
    for step, stats in sorted(step_contrast.items(), key=lambda item: step_sort_key(item[0])):
        diff = stats.get("hallucinated_minus_correct", {})
        rows.append(
            [
                step,
                stats.get("hallucinated_num_mentions", 0),
                stats.get("correct_num_mentions", 0),
                fmt_value(diff.get("teacher_minus_student_selected_logprob_mean_diff")),
                fmt_value(diff.get("teacher_minus_student_selected_logprob_sum_diff")),
                fmt_value(diff.get("teacher_selected_logprob_lt_student_frac_diff")),
                fmt_value(diff.get("teacher_penalized_mention_frac_diff")),
                fmt_value(diff.get("teacher_minus_student_entropy_mean_diff")),
                fmt_value(diff.get("topk_overlap_ratio_mean_diff")),
                fmt_value(diff.get("top1_match_frac_diff")),
            ]
        )
    return rows


def entropy_bin_table_rows(entropy_bin_summary):
    rows = []
    contrast_by_bin = entropy_bin_summary.get("hallucinated_minus_correct_by_bin", {})
    for bin_label, stats in contrast_by_bin.items():
        diff = stats.get("hallucinated_minus_correct", {})
        rows.append(
            [
                bin_label,
                stats.get("hallucinated_num_mentions", 0),
                stats.get("correct_num_mentions", 0),
                fmt_value(diff.get("teacher_minus_student_selected_logprob_mean_diff")),
                fmt_value(diff.get("teacher_minus_student_selected_logprob_sum_diff")),
                fmt_value(diff.get("teacher_selected_logprob_lt_student_frac_diff")),
                fmt_value(diff.get("teacher_penalized_mention_frac_diff")),
                fmt_value(diff.get("topk_overlap_ratio_mean_diff")),
                fmt_value(diff.get("top1_match_frac_diff")),
            ]
        )
    return rows


def delta_distribution_table_rows(distribution_summary, field):
    rows = []
    field_summary = distribution_summary.get(field, {})
    for object_type in ("correct_object", "hallucinated_object", "unknown_object"):
        stats = field_summary.get(object_type)
        if not stats:
            continue
        rows.append(
            [
                object_type,
                stats.get("count", 0),
                fmt_value(stats.get("mean")),
                fmt_value(stats.get("p10")),
                fmt_value(stats.get("p25")),
                fmt_value(stats.get("p50")),
                fmt_value(stats.get("p75")),
                fmt_value(stats.get("p90")),
                fmt_value(stats.get("frac_lt_neg0p1")),
                fmt_value(stats.get("frac_lt_neg0p05")),
                fmt_value(stats.get("frac_lt_0p0")),
            ]
        )
    contrast = field_summary.get("hallucinated_minus_correct")
    if contrast:
        rows.append(
            [
                "hallucinated_minus_correct",
                "",
                fmt_value(contrast.get("mean_diff")),
                fmt_value(contrast.get("p10_diff")),
                fmt_value(contrast.get("p25_diff")),
                fmt_value(contrast.get("p50_diff")),
                fmt_value(contrast.get("p75_diff")),
                fmt_value(contrast.get("p90_diff")),
                fmt_value(contrast.get("frac_lt_neg0p1_diff")),
                fmt_value(contrast.get("frac_lt_neg0p05_diff")),
                fmt_value(contrast.get("frac_lt_0p0_diff")),
            ]
        )
    return rows


def gate_sweep_table_rows(gates, limit=20):
    rows = []
    for gate in gates[:limit]:
        rows.append(
            [
                fmt_value(gate.get("student_entropy_gt")),
                fmt_value(gate.get("teacher_minus_student_logp_lt")),
                gate.get("selected_total", 0),
                gate.get("selected_hallucinated", 0),
                gate.get("selected_correct", 0),
                fmt_value(gate.get("hallucination_precision")),
                fmt_value(gate.get("hallucination_recall")),
                fmt_value(gate.get("correct_false_positive_rate")),
                fmt_value(gate.get("f1")),
                fmt_value(gate.get("precision_lift_vs_base")),
            ]
        )
    return rows


def lowres_quadrant_table_rows(quadrant_summary):
    rows = []
    buckets = quadrant_summary.get("bucket_summary", {})
    preferred_order = [
        "lowres_confident_support",
        "lowres_confident_reject",
        "lowres_uncertain",
        "lowres_ambiguous",
        "lowres_conflicting",
    ]
    for bucket in preferred_order + sorted(set(buckets) - set(preferred_order)):
        stats = buckets.get(bucket)
        if not stats:
            continue
        metrics = stats.get("metrics", {})
        rows.append(
            [
                bucket,
                stats.get("num_labeled_mentions", 0),
                stats.get("num_correct_mentions", 0),
                stats.get("num_hallucinated_mentions", 0),
                fmt_value(stats.get("hallucination_rate")),
                fmt_value(stats.get("precision_lift_vs_base")),
                fmt_value(stats.get("hallucinated_recall")),
                fmt_value(stats.get("correct_false_positive_rate")),
                fmt_value(metrics.get("teacher_minus_student_selected_logprob_mean")),
                fmt_value(metrics.get("teacher_entropy_mean")),
                fmt_value(metrics.get("teacher_top1_top2_margin_mean")),
                fmt_value(metrics.get("selected_token_rank_in_teacher_topk_mean")),
                fmt_value(metrics.get("selected_token_in_teacher_topk_frac")),
                fmt_value(metrics.get("top1_match_frac")),
            ]
        )
    return rows


def mention_example_table_rows(mentions, limit=20):
    rows = []
    for mention in mentions[:limit]:
        rows.append(
            [
                mention.get("global_step"),
                mention.get("image_id"),
                mention.get("caption_source"),
                mention.get("object_role"),
                mention.get("canonical_object"),
                mention.get("mention_text"),
                fmt_value(mention.get("teacher_minus_student_selected_logprob_sum")),
                fmt_value(mention.get("teacher_minus_student_selected_logprob_mean")),
                fmt_value(mention.get("student_entropy_mean")),
                fmt_value(mention.get("teacher_entropy_mean")),
                mention.get("context", "")[:180],
            ]
        )
    return rows


def write_markdown_summary(output, output_md):
    object_trace = output.get("object_trace_summary", {})
    case_overlap = output.get("case_overlap_summary", {})
    type_summary = object_trace.get("mention_type_summary", {})
    signal = object_trace.get("correct_vs_hallucinated_signal", {})
    lines = [
        "# OPD Object-Level Trace Summary",
        "",
        "## Inputs",
        "",
        f"- trace_dir: `{output.get('trace_dir')}`",
        f"- experiment_name: `{output.get('experiment_name')}`",
        f"- case_analysis: `{output.get('case_analysis')}`",
        f"- records: {output.get('num_records')}  tokens: {output.get('num_tokens')}  "
        f"unique_images: {output.get('num_unique_images')}",
        f"- trace_scope_counts: `{json.dumps(output.get('trace_scope_counts', {}), ensure_ascii=False)}`",
        "",
        "## Case Analysis Overlap",
        "",
        f"- case_analysis_loaded: {case_overlap.get('case_analysis_loaded')}",
        f"- matched_records_by_image_id: {case_overlap.get('matched_records_by_image_id')} / "
        f"{output.get('num_records')}",
        f"- image_id_overlap_count: {case_overlap.get('image_id_overlap_count')}",
        f"- matched_records_by_file_name: {case_overlap.get('matched_records_by_file_name')} / "
        f"{output.get('num_records')}",
        f"- file_name_overlap_count: {case_overlap.get('file_name_overlap_count')}",
        f"- trace_data_sources: `{json.dumps(case_overlap.get('trace_data_sources', {}), ensure_ascii=False)}`",
        f"- trace_image_id_samples: `{case_overlap.get('trace_image_id_samples')}`",
        f"- case_image_id_samples: `{case_overlap.get('case_image_id_samples')}`",
        f"- likely_reason: {case_overlap.get('likely_reason')}",
        "",
        "## GT Coverage",
        "",
        f"- records_with_gt: {object_trace.get('records_with_gt_count')}",
        f"- records_without_gt: {object_trace.get('records_without_gt_count')}",
        f"- records_with_object_mentions: {object_trace.get('records_with_object_mentions_count')}",
        f"- records_without_object_mentions: {object_trace.get('records_without_object_mentions_count')}",
        f"- gt_source_counts: `{json.dumps(object_trace.get('gt_source_counts', {}), ensure_ascii=False)}`",
        "",
        "## Correct vs Hallucinated Mentions",
        "",
        markdown_table(
            [
                "type",
                "mentions",
                "images",
                "mean teacher-student logp",
                "mean mention-sum teacher-student logp",
                "teacher<student frac",
                "penalized mention frac",
                "student entropy",
                "teacher entropy",
                "topk overlap",
                "top1 match",
            ],
            mention_summary_table_rows(type_summary),
        ),
        "",
        "## Eval Object Roles",
        "",
        markdown_table(
            [
                "role",
                "mentions",
                "images",
                "mean teacher-student logp",
                "mean mention-sum teacher-student logp",
                "teacher<student frac",
                "penalized mention frac",
                "student entropy",
                "teacher entropy",
                "topk overlap",
                "top1 match",
            ],
            generic_summary_table_rows(
                object_trace.get("mention_role_summary", {}),
                preferred_order=[
                    "removed_hallucination_candidate",
                    "lost_gt_candidate",
                    "model_only_hallucination",
                    "persistent_hallucination",
                    "model_correct_object",
                    "baseline_correct_object",
                ],
            ),
        ),
        "",
        "## Selective Suppression Signal",
        "",
    ]

    if signal.get("available"):
        contrast = signal.get("hallucinated_minus_correct", {})
        lines.extend(
            [
                "- selective_logprob_suppression_signal: "
                f"{signal.get('selective_logprob_suppression_signal')}",
                "- selective_teacher_lt_student_signal: "
                f"{signal.get('selective_teacher_lt_student_signal')}",
                "- selective_mention_penalty_signal: "
                f"{signal.get('selective_mention_penalty_signal')}",
                "- hallucinated_minus_correct teacher-student logp mean diff: "
                f"{fmt_value(contrast.get('teacher_minus_student_selected_logprob_mean_diff'))}",
                "- hallucinated_minus_correct teacher<student frac diff: "
                f"{fmt_value(contrast.get('teacher_selected_logprob_lt_student_frac_diff'))}",
                "- hallucinated_minus_correct entropy diff: "
                f"{fmt_value(contrast.get('teacher_minus_student_entropy_mean_diff'))}",
            ]
        )
    else:
        lines.append(f"- unavailable: {signal.get('reason')}")

    entropy_rows = entropy_bin_table_rows(object_trace.get("entropy_bin_summary", {}))
    if entropy_rows:
        lines.extend(
            [
                "",
                "## Student Entropy Bins: Hallucinated Minus Correct",
                "",
                markdown_table(
                    [
                        "student entropy bin",
                        "halluc mentions",
                        "correct mentions",
                        "mean logp gap",
                        "mention-sum logp gap",
                        "teacher<student gap",
                        "penalized mention gap",
                        "topk overlap gap",
                        "top1 match gap",
                    ],
                    entropy_rows,
                ),
            ]
        )

    delta_distribution = object_trace.get("logp_delta_distribution", {})
    delta_mean_rows = delta_distribution_table_rows(
        delta_distribution,
        "teacher_minus_student_selected_logprob_mean",
    )
    if delta_mean_rows:
        lines.extend(
            [
                "",
                "## Logprob Delta Distribution: Mention Mean",
                "",
                markdown_table(
                    [
                        "type",
                        "n",
                        "mean",
                        "p10",
                        "p25",
                        "p50",
                        "p75",
                        "p90",
                        "frac < -0.1",
                        "frac < -0.05",
                        "frac < 0",
                    ],
                    delta_mean_rows,
                ),
            ]
        )

    delta_sum_rows = delta_distribution_table_rows(
        delta_distribution,
        "teacher_minus_student_selected_logprob_sum",
    )
    if delta_sum_rows:
        lines.extend(
            [
                "",
                "## Logprob Delta Distribution: Mention Sum",
                "",
                markdown_table(
                    [
                        "type",
                        "n",
                        "mean",
                        "p10",
                        "p25",
                        "p50",
                        "p75",
                        "p90",
                        "frac < -0.1",
                        "frac < -0.05",
                        "frac < 0",
                    ],
                    delta_sum_rows,
                ),
            ]
        )

    gate_sweep = object_trace.get("gate_sweep", {})
    top_gates = gate_sweep.get("top_by_f1", [])
    if top_gates:
        lines.extend(
            [
                "",
                "## Selective Gate Sweep: Top By F1",
                "",
                f"- base_hallucination_rate: {fmt_value(gate_sweep.get('base_hallucination_rate'))}",
                "",
                markdown_table(
                    [
                        "student entropy >",
                        "teacher-student logp <",
                        "selected",
                        "halluc selected",
                        "correct selected",
                        "halluc precision",
                        "halluc recall",
                        "correct FPR",
                        "F1",
                        "precision lift",
                    ],
                    gate_sweep_table_rows(top_gates),
                ),
            ]
        )

    precision_gates = gate_sweep.get("top_precision_recall_ge_0p1", [])
    if precision_gates:
        lines.extend(
            [
                "",
                "## Selective Gate Sweep: Top Precision With Recall >= 0.1",
                "",
                markdown_table(
                    [
                        "student entropy >",
                        "teacher-student logp <",
                        "selected",
                        "halluc selected",
                        "correct selected",
                        "halluc precision",
                        "halluc recall",
                        "correct FPR",
                        "F1",
                        "precision lift",
                    ],
                    gate_sweep_table_rows(precision_gates),
                ),
            ]
        )

    quadrant_summary = object_trace.get("lowres_quadrant_summary", {})
    quadrant_rows = lowres_quadrant_table_rows(quadrant_summary)
    if quadrant_rows:
        interpretation = quadrant_summary.get("interpretation", {})
        lines.extend(
            [
                "",
                "## Low-Resolution Support/Reject Quadrants",
                "",
                f"- base_hallucination_rate: {fmt_value(quadrant_summary.get('base_hallucination_rate'))}",
                f"- support_bucket_hallucination_rate: "
                f"{fmt_value(interpretation.get('support_bucket_hallucination_rate'))}",
                f"- reject_bucket_precision_lift_vs_base: "
                f"{fmt_value(interpretation.get('reject_bucket_precision_lift_vs_base'))}",
                f"- reject_bucket_correct_false_positive_rate: "
                f"{fmt_value(interpretation.get('reject_bucket_correct_false_positive_rate'))}",
                "",
                markdown_table(
                    [
                        "bucket",
                        "labeled",
                        "correct",
                        "halluc",
                        "halluc rate",
                        "precision lift",
                        "halluc recall",
                        "correct FPR",
                        "mean logp delta",
                        "teacher entropy",
                        "teacher margin",
                        "teacher rank",
                        "in teacher topk",
                        "top1 match",
                    ],
                    quadrant_rows,
                ),
            ]
        )

    step_rows = step_contrast_table_rows(object_trace.get("step_contrast_hallucinated_minus_correct", {}))
    if step_rows:
        lines.extend(
            [
                "",
                "## Step Trend: Hallucinated Minus Correct",
                "",
                markdown_table(
                    [
                        "step",
                        "halluc mentions",
                        "correct mentions",
                        "mean logp gap",
                        "mean mention-sum logp gap",
                        "teacher<student gap",
                        "penalized mention gap",
                        "entropy gap",
                        "topk overlap gap",
                        "top1 match gap",
                    ],
                    step_rows,
                ),
            ]
        )

    examples = [
        (
            "Top Teacher-Suppressed Hallucination Mentions",
            object_trace.get("top_teacher_suppressed_hallucination_mentions", []),
        ),
        (
            "Top Teacher-Suppressed Correct Mentions",
            object_trace.get("top_teacher_suppressed_correct_mentions", []),
        ),
        (
            "Top Teacher-Supported Correct Mentions",
            object_trace.get("top_teacher_supported_correct_mentions", []),
        ),
        (
            "Top Teacher-Supported Hallucination Mentions",
            object_trace.get("top_teacher_supported_hallucination_mentions", []),
        ),
        (
            "Top Teacher-Suppressed Removed Hallucination Candidates",
            object_trace.get("top_teacher_suppressed_removed_hallucination_candidates", []),
        ),
        (
            "Top Teacher-Suppressed Lost GT Candidates",
            object_trace.get("top_teacher_suppressed_lost_gt_candidates", []),
        ),
    ]
    for title, mentions in examples:
        if not mentions:
            continue
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                markdown_table(
                    [
                        "step",
                        "image_id",
                        "source",
                        "role",
                        "object",
                        "mention",
                        "sum logp delta",
                        "mean logp delta",
                        "student entropy",
                        "teacher entropy",
                        "context",
                    ],
                    mention_example_table_rows(mentions),
                ),
            ]
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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


def build_trace_analysis_summary(
    trace_dir,
    case_analysis=None,
    experiment_name=None,
    oss_name=None,
    oss_base=None,
    oss_trace_path=None,
    fetched_from_oss=False,
    max_records=0,
    entropy_bin_edges=None,
    gate_entropy_thresholds=None,
    gate_logp_margins=None,
    quadrant_config=None,
):
    """Build the full OPD trace analysis summary without parsing CLI args."""
    records, skipped_non_trace_records = load_trace_records(trace_dir, max_records=max_records)
    if not records:
        hint = ""
        if experiment_name and oss_name and oss_base:
            hint = (
                f"\nTried local trace dir: {trace_dir}\n"
                f"Expected OSS trace path: {oss_base.rstrip('/')}/{oss_name}/training_artifacts/traces/"
            )
        raise SystemExit(f"No trace records found under {trace_dir}.{hint}")

    oss_base = oss_base or os.environ.get("OSS_BASE", "oss://industry-algo/yanlin/ckpt/OPD/v4")
    case_groups = load_case_groups(case_analysis)
    case_overlap_summary = summarize_case_overlap(records, case_groups)
    tokens = flatten_token_records(records, case_groups)
    entropy_bins = build_entropy_bins(parse_float_sequence(entropy_bin_edges, DEFAULT_ENTROPY_BIN_EDGES))
    parsed_gate_entropy_thresholds = parse_float_sequence(
        gate_entropy_thresholds,
        DEFAULT_GATE_ENTROPY_THRESHOLDS,
    )
    parsed_gate_logp_margins = parse_float_sequence(gate_logp_margins, DEFAULT_GATE_LOGP_MARGINS)
    object_trace_summary = summarize_object_trace(
        records,
        case_groups,
        entropy_bins,
        parsed_gate_entropy_thresholds,
        parsed_gate_logp_margins,
        quadrant_config=quadrant_config,
    )

    image_ids = [get_image_id(record) for record in records]
    trace_scope_counts = Counter(record.get("trace_scope", "train") for record in records)
    matched_records = case_overlap_summary["matched_records_by_image_id"]
    trace_coverage = summarize_trace_coverage(records)
    return {
        "trace_dir": trace_dir,
        "experiment_name": experiment_name,
        "oss_name": oss_name,
        "oss_base": oss_base,
        "oss_trace_path": oss_trace_path
        or (f"{oss_base.rstrip('/')}/{oss_name}/training_artifacts/traces" if oss_name else None),
        "fetched_from_oss": fetched_from_oss,
        "case_analysis": case_analysis,
        "num_records": len(records),
        "skipped_non_trace_records": skipped_non_trace_records,
        "num_tokens": len(tokens),
        "num_unique_images": len({image_id for image_id in image_ids if image_id is not None}),
        "trace_scope_counts": dict(trace_scope_counts.most_common()),
        "case_matched_records": matched_records,
        "case_matched_record_rate": safe_rate(matched_records, len(records)),
        "case_overlap_summary": case_overlap_summary,
        "trace_coverage": trace_coverage,
        "global_steps": trace_coverage["global_steps"],
        "record_group_summary": summarize_records(records, case_groups),
        "token_group_summary": summarize_tokens(tokens),
        "object_mention_summary": summarize_object_mentions(records, case_groups),
        "object_trace_summary": object_trace_summary,
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
    parser.add_argument(
        "--entropy-bin-edges",
        default="0,0.5,1,1.5,inf",
        help="Comma-separated student entropy bin edges for object-mention summaries.",
    )
    parser.add_argument(
        "--gate-entropy-thresholds",
        default="0.5,0.75,1,1.25,1.5,2",
        help="Comma-separated student-entropy thresholds for selective gate sweep.",
    )
    parser.add_argument(
        "--gate-logp-margins",
        default="0,0.01,0.02,0.05,0.1,0.2",
        help=(
            "Comma-separated positive margins m for gate condition "
            "teacher_minus_student_logp < -m."
        ),
    )
    parser.add_argument(
        "--quadrant-support-delta-min",
        type=float,
        default=DEFAULT_QUADRANT_CONFIG["support_delta_min"],
        help="Low-res confident-support requires teacher-student logp >= this value.",
    )
    parser.add_argument(
        "--quadrant-reject-delta-max",
        type=float,
        default=DEFAULT_QUADRANT_CONFIG["reject_delta_max"],
        help="Low-res confident-reject requires teacher-student logp <= this value.",
    )
    parser.add_argument(
        "--quadrant-teacher-entropy-max",
        type=float,
        default=DEFAULT_QUADRANT_CONFIG["teacher_entropy_max"],
        help="Low-res critic is treated as reliable only below this teacher entropy.",
    )
    parser.add_argument(
        "--quadrant-teacher-margin-min",
        type=float,
        default=DEFAULT_QUADRANT_CONFIG["teacher_margin_min"],
        help="Low-res critic is treated as reliable only above this top1-top2 margin.",
    )
    parser.add_argument(
        "--quadrant-support-rank-max",
        type=float,
        default=DEFAULT_QUADRANT_CONFIG["support_rank_max"],
        help="Low-res support evidence when selected token rank is at most this value.",
    )
    parser.add_argument(
        "--quadrant-reject-rank-min",
        type=float,
        default=DEFAULT_QUADRANT_CONFIG["reject_rank_min"],
        help="Low-res reject evidence when selected token rank is at least this value.",
    )
    parser.add_argument(
        "--quadrant-support-top1-min",
        type=float,
        default=DEFAULT_QUADRANT_CONFIG["support_top1_min"],
        help="Low-res support evidence when mention top1-match fraction is at least this value.",
    )
    parser.add_argument(
        "--quadrant-reject-top1-max",
        type=float,
        default=DEFAULT_QUADRANT_CONFIG["reject_top1_max"],
        help="Low-res reject evidence when mention top1-match fraction is at most this value.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Where to save summary JSON. Defaults to <trace-dir>/opd_trace_summary.json",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional Markdown report path. Defaults to <trace-dir>/opd_trace_summary.md",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Debug cap; 0 = all")
    args = parser.parse_args()

    trace_dir = args.trace_dir
    if not trace_dir and args.experiment_name:
        trace_dir = os.path.join(RES_OPD_ROOT, "traces", args.experiment_name)
    if not trace_dir:
        parser.error("--trace-dir is required unless --experiment-name is provided")

    # Derive default output paths from trace_dir when not specified
    if args.output_json is None:
        if os.path.isfile(trace_dir):
            base_dir = os.path.dirname(os.path.abspath(trace_dir))
        else:
            base_dir = os.path.abspath(trace_dir)
        args.output_json = os.path.join(base_dir, "opd_trace_summary.json")
        print(f"[Default] --output-json not specified, using: {args.output_json}")
    if args.output_md is None:
        if os.path.isfile(trace_dir):
            base_dir = os.path.dirname(os.path.abspath(trace_dir))
        else:
            base_dir = os.path.abspath(trace_dir)
        args.output_md = os.path.join(base_dir, "opd_trace_summary.md")
        print(f"[Default] --output-md not specified, using: {args.output_md}")

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

    quadrant_config = build_quadrant_config(
        support_delta_min=args.quadrant_support_delta_min,
        reject_delta_max=args.quadrant_reject_delta_max,
        teacher_entropy_max=args.quadrant_teacher_entropy_max,
        teacher_margin_min=args.quadrant_teacher_margin_min,
        support_rank_max=args.quadrant_support_rank_max,
        reject_rank_min=args.quadrant_reject_rank_min,
        support_top1_min=args.quadrant_support_top1_min,
        reject_top1_max=args.quadrant_reject_top1_max,
    )

    output = build_trace_analysis_summary(
        trace_dir,
        case_analysis=args.case_analysis,
        experiment_name=args.experiment_name,
        oss_name=oss_name,
        oss_base=args.oss_base,
        oss_trace_path=oss_trace_path,
        fetched_from_oss=fetched_from_oss,
        max_records=args.max_records,
        entropy_bin_edges=args.entropy_bin_edges,
        gate_entropy_thresholds=args.gate_entropy_thresholds,
        gate_logp_margins=args.gate_logp_margins,
        quadrant_config=quadrant_config,
    )
    object_trace_summary = output["object_trace_summary"]
    case_overlap_summary = output["case_overlap_summary"]
    trace_coverage = output["trace_coverage"]
    matched_records = output["case_matched_records"]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if args.output_md:
        write_markdown_summary(output, args.output_md)

    print(f"Loaded trace records: {output['num_records']}")
    print(f"Loaded tokens: {output['num_tokens']}")
    print(
        "Trace steps: "
        f"{trace_coverage['min_global_step']}..{trace_coverage['max_global_step']} "
        f"({trace_coverage['num_global_steps']} unique steps)"
    )
    if fetched_from_oss:
        print(f"Fetched traces from OSS: {output['oss_trace_path']}/")
    if output["skipped_non_trace_records"]:
        print(f"Skipped non-trace JSON records: {output['skipped_non_trace_records']}")
    print(f"Matched records to case analysis by image_id: {matched_records}/{output['num_records']}")
    if args.case_analysis:
        print(
            "Case-analysis overlap: "
            f"image_ids={case_overlap_summary['image_id_overlap_count']} "
            f"file_names={case_overlap_summary['file_name_overlap_count']} "
            f"trace_data_sources={case_overlap_summary['trace_data_sources']}"
        )
        if case_overlap_summary["is_disjoint_from_case_analysis"]:
            print(
                "WARNING: trace records and case analysis are disjoint. "
                "This usually means training rollout traces are being compared with eval/test case analysis; "
                "case-group summaries will stay under unmatched_or_no_case."
            )
    print(
        "Object-level GT coverage: "
        f"with_gt={object_trace_summary['records_with_gt_count']} "
        f"without_gt={object_trace_summary['records_without_gt_count']} "
        f"gt_sources={object_trace_summary['gt_source_counts']}"
    )
    object_type_summary = object_trace_summary.get("mention_type_summary", {})
    for object_type in ("correct_object", "hallucinated_object", "unknown_object"):
        stats = object_type_summary.get(object_type)
        if not stats:
            continue
        print(
            f"{object_type}: mentions={stats['num_mentions']} "
            f"teacher-student logp mean={fmt_value(stats.get('teacher_minus_student_selected_logprob_mean'))} "
            f"teacher<student={fmt_value(stats.get('teacher_selected_logprob_lt_student_frac'))} "
            f"student_entropy={fmt_value(stats.get('student_entropy_mean'))} "
            f"teacher_entropy={fmt_value(stats.get('teacher_entropy_mean'))}"
        )
    signal = object_trace_summary.get("correct_vs_hallucinated_signal", {})
    if signal.get("available"):
        contrast = signal.get("hallucinated_minus_correct", {})
        print(
            "Hallucinated-correct contrast: "
            f"logp_gap={fmt_value(contrast.get('teacher_minus_student_selected_logprob_mean_diff'))} "
            f"teacher<student_gap={fmt_value(contrast.get('teacher_selected_logprob_lt_student_frac_diff'))} "
            f"entropy_gap={fmt_value(contrast.get('teacher_minus_student_entropy_mean_diff'))}"
        )
    else:
        print(f"Hallucinated-correct contrast unavailable: {signal.get('reason')}")
    gate_sweep = object_trace_summary.get("gate_sweep", {})
    best_gate = (gate_sweep.get("top_by_f1") or [None])[0]
    if best_gate:
        print(
            "Best selective gate by F1: "
            f"student_entropy>{fmt_value(best_gate.get('student_entropy_gt'))} "
            f"teacher-student_logp<{fmt_value(best_gate.get('teacher_minus_student_logp_lt'))} "
            f"precision={fmt_value(best_gate.get('hallucination_precision'))} "
            f"recall={fmt_value(best_gate.get('hallucination_recall'))} "
            f"correct_fpr={fmt_value(best_gate.get('correct_false_positive_rate'))} "
            f"f1={fmt_value(best_gate.get('f1'))}"
        )
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
    if args.output_md:
        print(f"Saved Markdown report to: {args.output_md}")


if __name__ == "__main__":
    main()
