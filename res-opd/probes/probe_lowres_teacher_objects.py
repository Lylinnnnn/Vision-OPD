#!/usr/bin/env python3
"""
Low-resolution teacher object probe for Res-OPD.

This script scores fixed student captions with both student/high-res and
teacher/low-res image views, then runs POPE-style yes/no forced scoring for
object-level low-res support. It is designed for the base-model probe on
res-opd/data/test_1000.json.
"""

import argparse
import json
import glob
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict

from PIL import Image


PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_OPD_ROOT = os.path.dirname(PROBE_DIR)
EVAL_DIR = os.path.join(RES_OPD_ROOT, "eval")
sys.path.insert(0, EVAL_DIR)

from robust_chair_analysis import (  # noqa: E402
    build_double_word_dict,
    parse_official_synonyms,
    try_singularize,
)
from score_opd_eval_trace import (  # noqa: E402
    DEFAULT_COCO_VAL_ROOT,
    PROMPT_TEXT,
    as_int,
    bool_finite,
    combine_student_teacher,
    load_jsonl,
    load_model_and_processor,
    load_test_index,
    load_view_image,
    normalize_generation_record,
    resolve_image_path,
    safe_mean,
    score_view,
)


DEFAULT_INSTANCES_JSON = (
    "/home/liuyanlin.lyl/notebook/data/COCO/coco2017val/annotations/instances_val2017.json"
)
DEFAULT_TEST_JSON = os.path.join(RES_OPD_ROOT, "data", "test_1000.json")
DEFAULT_RESULTS_DIR = os.path.join(PROBE_DIR, "results", "lowres_teacher_objects")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe low-res teacher behavior on correct, hallucinated, and visually uncertain objects."
    )
    parser.add_argument("--model-path", help="Base model or checkpoint path used for forced scoring")
    parser.add_argument(
        "--eval-results",
        help="Student/high-res eval_chair.py eval_results.jsonl. Required unless --validate-coco-only.",
    )
    parser.add_argument("--test-json", default=DEFAULT_TEST_JSON)
    parser.add_argument("--instances-json", default=DEFAULT_INSTANCES_JSON)
    parser.add_argument("--image-root", default=DEFAULT_COCO_VAL_ROOT)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--coco-structure-json", default=None)
    parser.add_argument("--validate-coco-only", action="store_true")
    parser.add_argument(
        "--merge-jsonl-glob",
        default=None,
        help="Merge completed shard JSONL files and write a combined summary; skips model loading.",
    )
    parser.add_argument("--merge-output-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)

    parser.add_argument("--prompt", default=PROMPT_TEXT)
    parser.add_argument("--caption-field", default="auto")
    parser.add_argument("--degradation-mode", choices=["square", "original"], default="original")
    parser.add_argument("--student-px", type=int, default=0)
    parser.add_argument("--teacher-px", type=int, default=0)
    parser.add_argument("--target-px", type=int, default=448)
    parser.add_argument("--student-ratio", type=float, default=1.0)
    parser.add_argument("--teacher-ratio", type=float, default=0.75)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument(
        "--topk-breaks",
        default="1,10,20,30",
        help="Comma-separated top-k cutoffs summarized from the saved topk logprobs.",
    )
    parser.add_argument("--entropy", dest="entropy", action="store_true", default=True)
    parser.add_argument("--no-entropy", dest="entropy", action="store_false")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--max-objects-per-image",
        type=int,
        default=0,
        help="Debug cap for object probes per image. 0 = all candidate objects.",
    )
    parser.add_argument("--max-model-len", type=int, default=9728)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")

    parser.add_argument(
        "--probe-prompt-template",
        default="Is there a {object} in the image? Answer yes or no.",
    )
    parser.add_argument(
        "--probe-views",
        default="teacher",
        help="Comma-separated yes/no probe views: teacher or student,teacher. Caption forced scoring always uses both.",
    )
    parser.add_argument("--support-margin", type=float, default=0.5)
    parser.add_argument("--confident-entropy-max", type=float, default=0.65)
    parser.add_argument("--yes-variants", default="yes,Yes, yes, Yes")
    parser.add_argument("--no-variants", default="no,No, no, No")
    return parser.parse_args()


def safe_path_part(value):
    value = str(value or "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def default_run_name(eval_results):
    if not eval_results:
        return "coco_structure"
    eval_dir = os.path.dirname(os.path.abspath(eval_results))
    dataset_name = os.path.basename(eval_dir)
    experiment_name = os.path.basename(os.path.dirname(eval_dir))
    if experiment_name and dataset_name and experiment_name != dataset_name:
        return safe_path_part(f"{experiment_name}__{dataset_name}")
    return safe_path_part(dataset_name or os.path.splitext(os.path.basename(eval_results))[0])


def default_output_paths(eval_results):
    base_dir = os.path.join(DEFAULT_RESULTS_DIR, default_run_name(eval_results))
    return (
        os.path.join(base_dir, "lowres_teacher_object_probe.jsonl"),
        os.path.join(base_dir, "lowres_teacher_object_probe_summary.json"),
        os.path.join(base_dir, "coco_instance_structure_summary.json"),
    )


def format_sharded_path(path, shard_index, num_shards):
    if num_shards <= 1:
        return path
    shard_tag = f"shard{shard_index:02d}-of-{num_shards:02d}"
    mapping = {
        "shard": shard_tag,
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    if "{" in path and "}" in path:
        return path.format(**mapping)
    root, ext = os.path.splitext(path)
    return f"{root}.{shard_tag}{ext}"


def parse_topk_breaks(value, max_topk):
    breaks = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            k = int(part)
        except ValueError:
            raise ValueError(f"Invalid --topk-breaks entry: {part}") from None
        if k <= 0:
            raise ValueError(f"--topk-breaks values must be positive, got {k}")
        if max_topk > 0 and k > max_topk:
            continue
        breaks.append(k)
    return sorted(set(breaks))


def load_coco_instances(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    categories = {int(cat["id"]): cat["name"] for cat in data.get("categories", [])}
    images = {int(img["id"]): img for img in data.get("images", [])}
    grouped = defaultdict(lambda: defaultdict(list))
    for ann in data.get("annotations", []):
        image_id = as_int(ann.get("image_id"))
        category_id = as_int(ann.get("category_id"))
        if image_id is None or category_id is None:
            continue
        category_name = categories.get(category_id, f"category_{category_id}")
        grouped[image_id][category_name].append(ann)
    return {
        "raw": data,
        "categories": categories,
        "images": images,
        "by_image_object": grouped,
    }


def object_instance_stats(coco, image_id, obj):
    image_id = as_int(image_id)
    if image_id is None:
        return {"has_instance": False}
    image_info = coco["images"].get(image_id, {})
    width = float(image_info.get("width") or 0)
    height = float(image_info.get("height") or 0)
    image_area = width * height
    anns = list(coco["by_image_object"].get(image_id, {}).get(obj, []))
    if not anns:
        return {
            "has_instance": False,
            "num_instances": 0,
            "image_width": width or None,
            "image_height": height or None,
            "max_area_ratio": None,
            "total_area_ratio": None,
            "min_bbox_area_ratio": None,
            "max_bbox_area_ratio": None,
            "iscrowd_count": 0,
        }

    areas = [float(ann.get("area") or 0.0) for ann in anns]
    bbox_areas = []
    for ann in anns:
        bbox = ann.get("bbox") or []
        if len(bbox) == 4:
            bbox_areas.append(float(bbox[2]) * float(bbox[3]))
    denom = image_area if image_area > 0 else 1.0
    return {
        "has_instance": True,
        "num_instances": len(anns),
        "image_width": width or None,
        "image_height": height or None,
        "max_area_ratio": max(areas) / denom if areas else None,
        "total_area_ratio": sum(areas) / denom if areas else None,
        "min_bbox_area_ratio": min(bbox_areas) / denom if bbox_areas else None,
        "max_bbox_area_ratio": max(bbox_areas) / denom if bbox_areas else None,
        "iscrowd_count": sum(1 for ann in anns if int(ann.get("iscrowd") or 0) == 1),
    }


def area_bucket(stats):
    value = stats.get("max_area_ratio")
    if not bool_finite(value):
        return "missing"
    value = float(value)
    if value < 0.001:
        return "tiny(<0.1%)"
    if value < 0.01:
        return "small(<1%)"
    if value < 0.05:
        return "medium(<5%)"
    return "large(>=5%)"


def validate_coco_structure(coco, test_index):
    raw = coco["raw"]
    test_ids = set(test_index.keys())
    instance_image_ids = set(coco["images"].keys())
    test_missing_in_instances = sorted(test_ids - instance_image_ids)

    mismatch_examples = []
    object_count_deltas = []
    object_set_mismatch_count = 0
    for image_id, sample in test_index.items():
        expected = set(sample.get("objects") or sample.get("gt_objects") or [])
        from_instances = set(coco["by_image_object"].get(image_id, {}).keys())
        if expected != from_instances:
            object_set_mismatch_count += 1
            if len(mismatch_examples) < 20:
                mismatch_examples.append({
                    "image_id": image_id,
                    "test_only": sorted(expected - from_instances),
                    "instances_only": sorted(from_instances - expected),
                })
        object_count_deltas.append(len(from_instances) - len(expected))

    category_counter = Counter()
    for obj_map in coco["by_image_object"].values():
        for obj, anns in obj_map.items():
            category_counter[obj] += len(anns)

    summary = {
        "instances_json": {
            "top_level_keys": sorted(raw.keys()),
            "num_images": len(raw.get("images", [])),
            "num_annotations": len(raw.get("annotations", [])),
            "num_categories": len(raw.get("categories", [])),
            "first_image_keys": sorted((raw.get("images") or [{}])[0].keys()) if raw.get("images") else [],
            "first_annotation_keys": sorted((raw.get("annotations") or [{}])[0].keys()) if raw.get("annotations") else [],
            "first_category_keys": sorted((raw.get("categories") or [{}])[0].keys()) if raw.get("categories") else [],
        },
        "test_json": {
            "num_samples": len(test_index),
            "missing_image_ids_in_instances": len(test_missing_in_instances),
            "missing_image_id_examples": test_missing_in_instances[:20],
            "object_set_mismatch_count": object_set_mismatch_count,
            "object_count_delta_mean": safe_mean(object_count_deltas),
            "mismatch_examples": mismatch_examples,
        },
        "annotation_category_top20": category_counter.most_common(20),
    }
    return summary


def normalize_token(word):
    return try_singularize(word.lower())


def caption_tokens_with_spans(text):
    tokens = []
    for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+", text):
        raw = match.group(0)
        tokens.append({
            "raw": raw,
            "norm": normalize_token(raw),
            "start": match.start(),
            "end": match.end(),
        })
    return tokens


def extract_object_mentions(text, inverse_synonym_dict, double_word_dict):
    tokens = caption_tokens_with_spans(text)
    mentions = []
    i = 0
    while i < len(tokens):
        two = None
        if i + 1 < len(tokens):
            two = f"{tokens[i]['norm']} {tokens[i + 1]['norm']}"
        if two and two in double_word_dict:
            surface = text[tokens[i]["start"]:tokens[i + 1]["end"]]
            mapped = double_word_dict[two]
            canonical = inverse_synonym_dict.get(mapped, mapped)
            mentions.append({
                "object": canonical,
                "surface": surface,
                "char_start": tokens[i]["start"],
                "char_end": tokens[i + 1]["end"],
            })
            i += 2
            continue
        one = tokens[i]["norm"]
        if one in inverse_synonym_dict:
            mentions.append({
                "object": inverse_synonym_dict[one],
                "surface": text[tokens[i]["start"]:tokens[i]["end"]],
                "char_start": tokens[i]["start"],
                "char_end": tokens[i]["end"],
            })
        i += 1

    # Mirror the official CHAIR toilet-seat exception.
    objects = {mention["object"] for mention in mentions}
    if "toilet" in objects:
        mentions = [
            mention for mention in mentions
            if not (mention["object"] == "chair" and mention["surface"].lower() == "seat")
        ]
    return mentions


def decoded_response_and_offsets(processor, response_token_ids):
    decoded = ""
    offsets = []
    for idx in range(len(response_token_ids)):
        current = processor.decode(response_token_ids[: idx + 1], skip_special_tokens=True)
        if current.startswith(decoded):
            start = len(decoded)
        else:
            start = max(0, len(current) - len(processor.decode([response_token_ids[idx]], skip_special_tokens=True)))
        end = len(current)
        offsets.append((start, end))
        decoded = current
    return decoded, offsets


def overlapping_token_indices(char_start, char_end, token_offsets):
    indices = []
    for idx, (tok_start, tok_end) in enumerate(token_offsets):
        if tok_end <= char_start or tok_start >= char_end:
            continue
        if tok_end > tok_start:
            indices.append(idx)
    return indices


def _topk_logprobs(token_record, prefix, k=None):
    vals = token_record.get(f"{prefix}_topk_logprobs") or []
    vals = [float(v) for v in vals if bool_finite(v)]
    if k is not None:
        vals = vals[:k]
    return vals


def mean_topk_logprob(token_record, prefix, k=None):
    vals = _topk_logprobs(token_record, prefix, k)
    return safe_mean(vals)


def mean_topk_mass(token_record, prefix, k=None):
    vals = _topk_logprobs(token_record, prefix, k)
    if not vals:
        return None
    return sum(math.exp(v) for v in vals)


def mean_topk_entropy(token_record, prefix, k=None):
    vals = _topk_logprobs(token_record, prefix, k)
    if not vals:
        return None
    max_val = max(vals)
    weights = [math.exp(v - max_val) for v in vals]
    total = sum(weights)
    if total <= 0:
        return None
    probs = [w / total for w in weights]
    return -sum(p * math.log(max(p, 1e-12)) for p in probs)


def add_topk_break_metrics(out, selected, prefix, topk_breaks):
    for k in topk_breaks:
        out[f"{prefix}_top{k}_logprob_mean"] = safe_mean(
            mean_topk_logprob(record, prefix, k) for record in selected
        )
        out[f"{prefix}_top{k}_entropy_mean"] = safe_mean(
            mean_topk_entropy(record, prefix, k) for record in selected
        )
        out[f"{prefix}_top{k}_mass_mean"] = safe_mean(
            mean_topk_mass(record, prefix, k) for record in selected
        )


def summarize_token_span(token_records, token_indices, topk_breaks):
    selected = [token_records[i] for i in token_indices if 0 <= i < len(token_records)]
    out = {"num_tokens": len(selected)}
    for prefix in ("student", "teacher"):
        out[f"{prefix}_selected_logprob_mean"] = safe_mean(
            record.get(f"{prefix}_selected_logprob") for record in selected
        )
        out[f"{prefix}_entropy_mean"] = safe_mean(record.get(f"{prefix}_entropy") for record in selected)
        out[f"{prefix}_topk_logprob_mean"] = safe_mean(mean_topk_logprob(record, prefix) for record in selected)
        out[f"{prefix}_topk_entropy_mean"] = safe_mean(mean_topk_entropy(record, prefix) for record in selected)
        out[f"{prefix}_topk_mass_mean"] = safe_mean(mean_topk_mass(record, prefix) for record in selected)
        add_topk_break_metrics(out, selected, prefix, topk_breaks)
        out[f"{prefix}_top1_top2_margin_mean"] = safe_mean(
            record.get(f"{prefix}_top1_top2_margin") for record in selected
        )
    if bool_finite(out.get("teacher_selected_logprob_mean")) and bool_finite(out.get("student_selected_logprob_mean")):
        out["teacher_minus_student_logprob_mean"] = (
            out["teacher_selected_logprob_mean"] - out["student_selected_logprob_mean"]
        )
    else:
        out["teacher_minus_student_logprob_mean"] = None
    if bool_finite(out.get("teacher_entropy_mean")) and bool_finite(out.get("student_entropy_mean")):
        out["teacher_minus_student_entropy_mean"] = out["teacher_entropy_mean"] - out["student_entropy_mean"]
    else:
        out["teacher_minus_student_entropy_mean"] = None
    return out


def parse_variants(value):
    return [item for item in (part.strip("\n") for part in value.split(",")) if item != ""]


def one_token_variant_ids(processor, variants):
    token_to_variant = {}
    tokenizer = getattr(processor, "tokenizer", processor)
    for variant in variants:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            token_to_variant[int(ids[0])] = variant
    return token_to_variant


def tensorize_prompt(processor, image, prompt, device):
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    return inputs, prompt_text


def logsumexp(values):
    values = [float(v) for v in values if bool_finite(v)]
    if not values:
        return None
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def binary_entropy_from_margin(margin):
    if not bool_finite(margin):
        return None
    margin = float(margin)
    if margin >= 0:
        p_yes = 1.0 / (1.0 + math.exp(-margin))
    else:
        exp_m = math.exp(margin)
        p_yes = exp_m / (1.0 + exp_m)
    p_no = 1.0 - p_yes
    return -sum(p * math.log(max(p, 1e-12)) for p in (p_yes, p_no))


def score_yes_no_first_token(model, processor, device, image, prompt, yes_token_ids, no_token_ids):
    import torch

    if not yes_token_ids or not no_token_ids:
        raise ValueError("yes/no variants must include at least one single-token variant")
    inputs, prompt_text = tensorize_prompt(processor, image, prompt, device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    yes_logprob = logsumexp([float(log_probs[token_id].detach().cpu().item()) for token_id in yes_token_ids])
    no_logprob = logsumexp([float(log_probs[token_id].detach().cpu().item()) for token_id in no_token_ids])
    margin = yes_logprob - no_logprob if yes_logprob is not None and no_logprob is not None else None
    return {
        "prompt_text": prompt_text,
        "yes_logprob": yes_logprob,
        "no_logprob": no_logprob,
        "yes_minus_no_logprob": margin,
        "binary_entropy": binary_entropy_from_margin(margin),
    }


def classify_teacher_state(margin, entropy, support_margin, confident_entropy_max):
    if not bool_finite(margin) or not bool_finite(entropy):
        return "unknown"
    if entropy > confident_entropy_max:
        return "uncertain"
    if margin >= support_margin:
        return "support"
    if margin <= -support_margin:
        return "reject"
    return "uncertain"


def object_label(obj, gt_objects, mentioned_objects):
    if obj in mentioned_objects and obj in gt_objects:
        return "correct"
    if obj in mentioned_objects and obj not in gt_objects:
        return "hallucinated"
    if obj in gt_objects and obj not in mentioned_objects:
        return "missed_gt"
    return "other_candidate"


def update_numeric_lists(bucket, record, fields=None):
    bucket["count"] += 1
    fields = fields or record.keys()
    for field in fields:
        value = record.get(field)
        if bool_finite(value):
            bucket.setdefault(field, []).append(float(value))


def aggregate_numeric_bucket(bucket):
    out = {"count": bucket.get("count", 0)}
    for field, values in bucket.items():
        if field == "count":
            continue
        out[f"{field}_mean"] = safe_mean(values)
    return out


def summarize_records(sample_records):
    mention_buckets = defaultdict(lambda: {"count": 0})
    probe_buckets = defaultdict(lambda: {"count": 0})
    state_by_label = defaultdict(Counter)
    size_by_label_state = defaultdict(lambda: defaultdict(Counter))
    totals = Counter()

    for sample in sample_records:
        totals["samples"] += 1
        totals["mentioned_objects"] += len(sample.get("mentioned_objects", []))
        totals["correct_objects"] += len(sample.get("correct_objects", []))
        totals["hallucinated_objects"] += len(sample.get("hallucinated_objects", []))
        totals["missed_gt_objects"] += len(sample.get("missed_gt_objects", []))
        for mention in sample.get("object_mentions", []):
            label = mention.get("label")
            update_numeric_lists(mention_buckets[label], mention.get("token_metrics", {}))
        for probe in sample.get("object_probes", []):
            label = probe.get("label")
            teacher_state = probe.get("teacher_state", "unknown")
            flat = dict(probe)
            flat.update(probe.get("instance_stats", {}))
            update_numeric_lists(probe_buckets[label], flat)
            state_by_label[label][teacher_state] += 1
            size_by_label_state[label][probe.get("area_bucket", "missing")][teacher_state] += 1

    return {
        "totals": dict(totals),
        "mention_token_metrics_by_label": {
            label: aggregate_numeric_bucket(bucket) for label, bucket in sorted(mention_buckets.items())
        },
        "object_probe_metrics_by_label": {
            label: aggregate_numeric_bucket(bucket) for label, bucket in sorted(probe_buckets.items())
        },
        "teacher_state_by_label": {
            label: {
                "count": sum(counter.values()),
                "states": dict(counter),
                "rates": {
                    state: count / max(sum(counter.values()), 1)
                    for state, count in sorted(counter.items())
                },
            }
            for label, counter in sorted(state_by_label.items())
        },
        "teacher_state_by_label_and_size": {
            label: {
                bucket: {
                    "count": sum(counter.values()),
                    "states": dict(counter),
                    "rates": {
                        state: count / max(sum(counter.values()), 1)
                        for state, count in sorted(counter.items())
                    },
                }
                for bucket, counter in sorted(size_map.items())
            }
            for label, size_map in sorted(size_by_label_state.items())
        },
    }


def load_probe_jsonl(path):
    records = []
    for record in load_jsonl(path):
        if isinstance(record, dict):
            records.append(record)
    return records


def merge_probe_outputs(jsonl_glob, output_json, overwrite=False):
    paths = sorted(glob.glob(jsonl_glob))
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched: {jsonl_glob}")
    if output_json and os.path.exists(output_json) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing summary: {output_json}")

    records = []
    per_file_counts = {}
    for path in paths:
        current = load_probe_jsonl(path)
        records.extend(current)
        per_file_counts[path] = len(current)

    summary = summarize_records(records)
    summary.update({
        "merged_from": paths,
        "per_file_counts": per_file_counts,
        "output_record_count": len(records),
    })
    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved merged summary to: {output_json}")
    print(f"Merged {len(records)} records from {len(paths)} JSONL files.")
    return summary


def main():
    args = parse_args()
    if args.merge_jsonl_glob:
        output_json = args.merge_output_json or os.path.join(
            DEFAULT_RESULTS_DIR,
            "merged_lowres_teacher_object_probe_summary.json",
        )
        merge_probe_outputs(args.merge_jsonl_glob, output_json, overwrite=args.overwrite)
        return

    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )

    test_index = load_test_index(args.test_json)
    coco = load_coco_instances(args.instances_json)

    default_jsonl, default_summary, default_coco = default_output_paths(args.eval_results)
    args.output_jsonl = args.output_jsonl or default_jsonl
    args.summary_json = args.summary_json or default_summary
    args.coco_structure_json = args.coco_structure_json or default_coco
    args.output_jsonl = format_sharded_path(args.output_jsonl, args.shard_index, args.num_shards)
    args.summary_json = format_sharded_path(args.summary_json, args.shard_index, args.num_shards)

    coco_summary = validate_coco_structure(coco, test_index)
    os.makedirs(os.path.dirname(os.path.abspath(args.coco_structure_json)), exist_ok=True)
    with open(args.coco_structure_json, "w", encoding="utf-8") as f:
        json.dump(coco_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved COCO structure summary to: {args.coco_structure_json}")
    print(
        "COCO structure: "
        f"images={coco_summary['instances_json']['num_images']} "
        f"annotations={coco_summary['instances_json']['num_annotations']} "
        f"categories={coco_summary['instances_json']['num_categories']} "
        f"test_missing={coco_summary['test_json']['missing_image_ids_in_instances']} "
        f"object_mismatch={coco_summary['test_json']['object_set_mismatch_count']}"
    )

    if args.validate_coco_only:
        return
    if not args.model_path or not args.eval_results:
        raise ValueError("--model-path and --eval-results are required unless --validate-coco-only")

    if args.overwrite:
        for path in (args.output_jsonl, args.summary_json):
            if path and os.path.exists(path):
                os.remove(path)

    mscoco_objects, inverse_synonym_dict = parse_official_synonyms()
    double_word_dict = build_double_word_dict()
    del mscoco_objects

    eval_records = [
        normalize_generation_record(record, test_index, args.caption_field)
        for record in load_jsonl(args.eval_results)
    ]
    if args.max_samples > 0:
        eval_records = eval_records[: args.max_samples]
    total_eval_records = len(eval_records)
    if args.num_shards > 1:
        eval_records = [
            record for idx, record in enumerate(eval_records)
            if idx % args.num_shards == args.shard_index
        ]
        print(
            f"Shard {args.shard_index}/{args.num_shards}: "
            f"selected {len(eval_records)} of {total_eval_records} records"
        )
    topk_breaks = parse_topk_breaks(args.topk_breaks, args.topk)
    print(f"topk breaks summarized from top{args.topk}: {topk_breaks}")

    model, processor, device = load_model_and_processor(args.model_path, args.torch_dtype)
    yes_ids = one_token_variant_ids(processor, parse_variants(args.yes_variants))
    no_ids = one_token_variant_ids(processor, parse_variants(args.no_variants))
    print(f"yes variants used: {yes_ids}")
    print(f"no variants used: {no_ids}")
    probe_views = {item.strip() for item in args.probe_views.split(",") if item.strip()}
    if not probe_views <= {"student", "teacher"}:
        raise ValueError(f"Unsupported --probe-views: {args.probe_views}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    started = time.time()
    sample_records = []
    failures = 0
    with open(args.output_jsonl, "a", encoding="utf-8") as f:
        for idx, record in enumerate(eval_records, start=1):
            image_id = record.get("image_id")
            image_path = resolve_image_path(record, test_index, args.image_root)
            caption = record.get("generated_caption", "")
            if not image_path or not caption or str(caption).startswith("[ERROR]"):
                failures += 1
                continue

            try:
                student_image = load_view_image(
                    image_path,
                    args.degradation_mode,
                    args.student_px,
                    args.target_px,
                    args.student_ratio,
                    blank_when_px_zero=False,
                )
                teacher_image = load_view_image(
                    image_path,
                    args.degradation_mode,
                    args.teacher_px,
                    args.target_px,
                    args.teacher_ratio,
                    blank_when_px_zero=True,
                )
                student = score_view(
                    model, processor, device, student_image, args.prompt, caption,
                    args.topk, args.entropy, args.max_model_len,
                )
                teacher = score_view(
                    model, processor, device, teacher_image, args.prompt, caption,
                    args.topk, args.entropy, args.max_model_len,
                )
                combined = combine_student_teacher(student, teacher)
                caption_forced_summary = dict(combined["summary"])
                caption_forced_summary.update(
                    summarize_token_span(
                        combined["token_records"],
                        range(len(combined["token_records"])),
                        topk_breaks,
                    )
                )
                decoded_caption, token_offsets = decoded_response_and_offsets(
                    processor,
                    combined["response_token_ids"],
                )
                mentions = extract_object_mentions(decoded_caption, inverse_synonym_dict, double_word_dict)
                gt_objects = set(record.get("gt_objects") or [])
                mentioned_objects = {mention["object"] for mention in mentions}
                correct_objects = sorted(mentioned_objects & gt_objects)
                hallucinated_objects = sorted(mentioned_objects - gt_objects)
                missed_gt_objects = sorted(gt_objects - mentioned_objects)
                candidate_objects = sorted(mentioned_objects | gt_objects)
                if args.max_objects_per_image > 0:
                    candidate_objects = candidate_objects[: args.max_objects_per_image]

                object_mentions = []
                for mention in mentions:
                    token_indices = overlapping_token_indices(
                        mention["char_start"], mention["char_end"], token_offsets
                    )
                    stats = object_instance_stats(coco, image_id, mention["object"])
                    token_metrics = summarize_token_span(combined["token_records"], token_indices, topk_breaks)
                    token_metrics.update({
                        "max_area_ratio": stats.get("max_area_ratio"),
                        "total_area_ratio": stats.get("total_area_ratio"),
                    })
                    object_mentions.append({
                        **mention,
                        "label": object_label(mention["object"], gt_objects, mentioned_objects),
                        "token_indices": token_indices,
                        "instance_stats": stats,
                        "area_bucket": area_bucket(stats),
                        "token_metrics": token_metrics,
                    })

                object_probes = []
                for obj in candidate_objects:
                    prompt = args.probe_prompt_template.format(object=obj)
                    stats = object_instance_stats(coco, image_id, obj)
                    probe = {
                        "object": obj,
                        "label": object_label(obj, gt_objects, mentioned_objects),
                        "prompt": prompt,
                        "instance_stats": stats,
                        "area_bucket": area_bucket(stats),
                    }
                    if "student" in probe_views:
                        student_probe = score_yes_no_first_token(
                            model, processor, device, student_image, prompt, yes_ids, no_ids
                        )
                        probe.update({
                            "student_yes_logprob": student_probe["yes_logprob"],
                            "student_no_logprob": student_probe["no_logprob"],
                            "student_yes_minus_no_logprob": student_probe["yes_minus_no_logprob"],
                            "student_binary_entropy": student_probe["binary_entropy"],
                        })
                    if "teacher" in probe_views:
                        teacher_probe = score_yes_no_first_token(
                            model, processor, device, teacher_image, prompt, yes_ids, no_ids
                        )
                        probe.update({
                            "teacher_yes_logprob": teacher_probe["yes_logprob"],
                            "teacher_no_logprob": teacher_probe["no_logprob"],
                            "teacher_yes_minus_no_logprob": teacher_probe["yes_minus_no_logprob"],
                            "teacher_binary_entropy": teacher_probe["binary_entropy"],
                            "teacher_state": classify_teacher_state(
                                teacher_probe["yes_minus_no_logprob"],
                                teacher_probe["binary_entropy"],
                                args.support_margin,
                                args.confident_entropy_max,
                            ),
                        })
                    object_probes.append(probe)

                out = {
                    "image_id": image_id,
                    "file_name": record.get("file_name"),
                    "image_path": image_path,
                    "model_path": args.model_path,
                    "degradation_mode": args.degradation_mode,
                    "student_ratio": args.student_ratio,
                    "teacher_ratio": args.teacher_ratio,
                    "student_px": args.student_px,
                    "teacher_px": args.teacher_px,
                    "target_px": args.target_px,
                    "caption": caption,
                    "decoded_caption": decoded_caption,
                    "gt_objects": sorted(gt_objects),
                    "mentioned_objects": sorted(mentioned_objects),
                    "correct_objects": correct_objects,
                    "hallucinated_objects": hallucinated_objects,
                    "missed_gt_objects": missed_gt_objects,
                    "caption_forced_summary": caption_forced_summary,
                    "object_mentions": object_mentions,
                    "object_probes": object_probes,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
                sample_records.append(out)
            except Exception as exc:
                failures += 1
                print(f"WARNING: failed image_id={image_id}: {exc}", file=sys.stderr)
                continue

            if idx % 10 == 0:
                elapsed = time.time() - started
                print(f"Processed {idx}/{len(eval_records)} samples ({elapsed:.0f}s, failures={failures})")

    summary = summarize_records(sample_records)
    summary.update({
        "config": {
            "model_path": args.model_path,
            "eval_results": args.eval_results,
            "test_json": args.test_json,
            "instances_json": args.instances_json,
            "degradation_mode": args.degradation_mode,
            "student_ratio": args.student_ratio,
            "teacher_ratio": args.teacher_ratio,
            "topk": args.topk,
            "topk_breaks": topk_breaks,
            "probe_views": sorted(probe_views),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "total_eval_records": total_eval_records,
            "shard_eval_records": len(eval_records),
            "support_margin": args.support_margin,
            "confident_entropy_max": args.confident_entropy_max,
            "entropy": args.entropy,
        },
        "failures": failures,
        "output_jsonl": args.output_jsonl,
        "coco_structure_json": args.coco_structure_json,
    })
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved object probe JSONL to: {args.output_jsonl}")
    print(f"Saved object probe summary to: {args.summary_json}")


if __name__ == "__main__":
    main()
