#!/usr/bin/env python3
"""Analyze how an OPD-trained model changes base-caption token distributions.

This is the P2/P3 mechanism probe for the paper:

  caption source: base model captions on original images
  base dist:      base model + original image
  teacher dist:   base model + low-res image
  after dist:     trained checkpoint + original image

The script forced-scores the same caption under all three distributions. It
keeps the training-aligned RKL signal as the first gate, then asks whether the
trained model mostly sharpens the base distribution, reshapes its top-k support,
or suppresses the base-emitted token on hallucinated object spans.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any

PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_OPD_ROOT = os.path.dirname(PROBE_DIR)
EVAL_DIR = os.path.join(RES_OPD_ROOT, "eval")
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, EVAL_DIR)

from probe_same_image_rkl_signal import (  # noqa: E402
    add_non_object_token_roles,
    bool_finite,
    build_canonical_synonym_map,
    build_token_char_spans,
    bounded_context,
    checkpoint_has_weights,
    cleanup_local_checkpoint,
    fetch_checkpoint_from_oss,
    find_object_mentions,
    fmt,
    get_gt_objects,
    load_model_and_processor,
    parse_official_synonyms,
    percentile,
    safe_mean,
    tensorize_inputs_batch,
    forward_target_logits_batch,
)
from score_opd_eval_trace import (  # noqa: E402
    DEFAULT_COCO_VAL_ROOT,
    PROMPT_TEXT,
    as_int,
    load_jsonl,
    load_test_index,
    load_view_image,
    normalize_generation_record,
    resolve_image_path,
)


DEFAULT_OUTPUT_ROOT = os.path.join(PROBE_DIR, "results", "model_delta_logprob")

TOKEN_METRIC_FIELDS = [
    "rkl_after_to_base",
    "fkl_base_to_after",
    "jsd_after_base",
    "rkl_after_to_teacher",
    "fkl_teacher_to_after",
    "jsd_after_teacher",
    "rkl_base_to_teacher",
    "fkl_teacher_to_base",
    "jsd_base_teacher",
    "base_nll",
    "after_nll",
    "teacher_nll",
    "delta_logp_after_base",
    "base_entropy",
    "after_entropy",
    "teacher_entropy",
    "delta_entropy_after_base",
    "base_top1_prob",
    "after_top1_prob",
    "teacher_top1_prob",
    "delta_top1_prob_after_base",
    "base_top1_top2_margin",
    "after_top1_top2_margin",
    "delta_top1_margin_after_base",
    "after_base_topk_jaccard",
    "after_base_topk_overlap_ratio",
    "after_mass_on_base_topk",
    "base_mass_on_after_topk",
    "after_mass_on_teacher_topk",
    "base_mass_on_teacher_topk",
    "delta_after_base_mass_on_teacher_topk",
    "base_selected_rank",
    "after_selected_rank",
    "teacher_selected_rank",
    "riskmask_nll_score",
    "riskmask_entropy_score",
]

BOOLEAN_FIELDS = [
    "after_base_top1_match",
    "after_teacher_top1_match",
    "base_teacher_top1_match",
    "emit_in_base_topk",
    "emit_in_after_topk",
    "emit_in_teacher_topk",
    "riskmask_nll_selected",
    "riskmask_entropy_selected",
]

MECHANISM_LABELS = [
    "sharpen_same_top1",
    "reshape_topk",
    "suppress_emitted",
    "boost_emitted",
    "stable",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe base-vs-trained logprob changes on base-generated captions."
    )
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--after-model-path", required=True)
    parser.add_argument("--after-name", default=None)
    parser.add_argument("--after-oss-checkpoint", default=None)
    parser.add_argument("--cleanup-after", action="store_true")
    parser.add_argument("--eval-results", required=True, help="Base original-image caption eval_results.jsonl.")
    parser.add_argument(
        "--after-eval-results",
        default=None,
        help="Optional trained-model eval_results.jsonl for kept/removed hallucination labels.",
    )
    parser.add_argument("--caption-field", default="auto")
    parser.add_argument("--caption-source-label", default="base_original_caption")
    parser.add_argument("--test-json", default=os.path.join(RES_OPD_ROOT, "data", "test_1000.json"))
    parser.add_argument("--image-root", default=DEFAULT_COCO_VAL_ROOT)
    parser.add_argument("--prompt", default=PROMPT_TEXT)
    parser.add_argument("--student-ratio", type=float, default=1.0)
    parser.add_argument("--teacher-ratio", type=float, default=0.75)
    parser.add_argument("--degradation-mode", choices=["original"], default="original")
    parser.add_argument("--target-px", type=int, default=448)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument("--examples-jsonl", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kl-chunk-size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.30)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--logp-eps", type=float, default=0.05)
    parser.add_argument("--entropy-eps", type=float, default=0.02)
    parser.add_argument("--prob-eps", type=float, default=0.02)
    parser.add_argument("--topk-jaccard-threshold", type=float, default=0.50)
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def safe_div(num: float, den: float) -> float | None:
    return num / den if den else None


def ratio_tag(value: float) -> str:
    return str(value).replace(".", "p")


def safe_path_part(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("_") or "unknown"


def infer_output_paths(args: argparse.Namespace) -> tuple[str, str, str, str]:
    base_name = os.path.basename(os.path.normpath(args.base_model_path))
    after_name = args.after_name or os.path.basename(os.path.normpath(args.after_model_path))
    eval_dir = os.path.basename(os.path.dirname(os.path.abspath(args.eval_results)))
    run_name = safe_path_part(
        f"{base_name}__after_{after_name}__{eval_dir}"
        f"__sr{ratio_tag(args.student_ratio)}_tr{ratio_tag(args.teacher_ratio)}"
    )
    output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, run_name)
    return (
        args.output_jsonl or os.path.join(output_dir, "model_delta_trace.jsonl"),
        args.summary_json or os.path.join(output_dir, "model_delta_summary.json"),
        args.summary_md or os.path.join(output_dir, "model_delta_summary.md"),
        args.examples_jsonl or os.path.join(output_dir, "model_delta_examples.jsonl"),
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        fail("--batch-size must be positive.")
    if args.kl_chunk_size <= 0:
        fail("--kl-chunk-size must be positive.")
    if args.num_shards <= 0:
        fail("--num-shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        fail("--shard-index must satisfy 0 <= shard_index < num_shards.")
    if not 0.0 < args.top_p <= 1.0:
        fail("--top-p must satisfy 0 < top_p <= 1.")
    if not args.analyze_only:
        if not os.path.isdir(args.base_model_path):
            fail(f"--base-model-path does not exist: {args.base_model_path}")
        if args.after_oss_checkpoint:
            if not args.after_model_path:
                fail("--after-model-path is required with --after-oss-checkpoint.")
        elif not os.path.isdir(args.after_model_path):
            fail(f"--after-model-path does not exist: {args.after_model_path}")
    if not os.path.exists(args.eval_results):
        fail(f"--eval-results does not exist: {args.eval_results}")
    if args.after_eval_results and not os.path.exists(args.after_eval_results):
        fail(f"--after-eval-results does not exist: {args.after_eval_results}")
    if args.test_json and not os.path.exists(args.test_json):
        fail(f"--test-json does not exist: {args.test_json}")


def prepare_oss_checkpoint(args: argparse.Namespace) -> list[str]:
    if args.analyze_only or not args.after_oss_checkpoint:
        return []
    did_download = fetch_checkpoint_from_oss(args.after_oss_checkpoint, args.after_model_path)
    if did_download:
        return [os.path.abspath(args.after_model_path)]
    return []


def existing_done_keys(path: str) -> set[tuple[int | None, str]]:
    if not os.path.exists(path):
        return set()
    keys: set[tuple[int | None, str]] = set()
    for record in load_jsonl(path):
        metadata = record.get("metadata") or {}
        keys.add((as_int(metadata.get("image_id")), record.get("caption_source") or "base_original_caption"))
    return keys


def record_done_key(args: argparse.Namespace, record: dict[str, Any]) -> tuple[int | None, str]:
    return (as_int(record.get("image_id")), args.caption_source_label or record.get("caption_source") or "base_original_caption")


def load_caption_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    test_index = load_test_index(args.test_json)
    records = [
        normalize_generation_record(record, test_index, args.caption_field)
        for record in load_jsonl(args.eval_results)
    ]
    records = [
        record
        for record in records
        if record.get("generated_caption") and not str(record.get("generated_caption")).startswith("[ERROR]")
    ]
    if args.max_samples:
        records = records[: args.max_samples]
    if args.num_shards > 1:
        records = [record for idx, record in enumerate(records) if idx % args.num_shards == args.shard_index]
    if not records:
        fail(f"No usable caption records for shard {args.shard_index}/{args.num_shards}: {args.eval_results}")
    return records


def dist_triplet_metrics(after_lp, base_lp, teacher_lp):
    import torch

    after_p = after_lp.exp()
    base_p = base_lp.exp()
    teacher_p = teacher_lp.exp()
    log_m_ab = torch.logaddexp(after_lp, base_lp) - math.log(2.0)
    log_m_at = torch.logaddexp(after_lp, teacher_lp) - math.log(2.0)
    log_m_bt = torch.logaddexp(base_lp, teacher_lp) - math.log(2.0)
    return {
        "rkl_after_to_base": (after_p * (after_lp - base_lp)).sum(dim=-1),
        "fkl_base_to_after": (base_p * (base_lp - after_lp)).sum(dim=-1),
        "jsd_after_base": 0.5 * (after_p * (after_lp - log_m_ab)).sum(dim=-1)
        + 0.5 * (base_p * (base_lp - log_m_ab)).sum(dim=-1),
        "rkl_after_to_teacher": (after_p * (after_lp - teacher_lp)).sum(dim=-1),
        "fkl_teacher_to_after": (teacher_p * (teacher_lp - after_lp)).sum(dim=-1),
        "jsd_after_teacher": 0.5 * (after_p * (after_lp - log_m_at)).sum(dim=-1)
        + 0.5 * (teacher_p * (teacher_lp - log_m_at)).sum(dim=-1),
        "rkl_base_to_teacher": (base_p * (base_lp - teacher_lp)).sum(dim=-1),
        "fkl_teacher_to_base": (teacher_p * (teacher_lp - base_lp)).sum(dim=-1),
        "jsd_base_teacher": 0.5 * (base_p * (base_lp - log_m_bt)).sum(dim=-1)
        + 0.5 * (teacher_p * (teacher_lp - log_m_bt)).sum(dim=-1),
        "after_entropy": -(after_p * after_lp).sum(dim=-1),
        "base_entropy": -(base_p * base_lp).sum(dim=-1),
        "teacher_entropy": -(teacher_p * teacher_lp).sum(dim=-1),
    }


def exact_selected_rank(log_probs, selected):
    return (log_probs > log_probs.gather(1, selected.unsqueeze(1))).sum(dim=-1) + 1


def topk_info(log_probs_row, selected_id: int, k: int) -> dict[str, Any]:
    import torch

    if k <= 0:
        return {}
    k = min(k, int(log_probs_row.shape[-1]))
    vals, ids = torch.topk(log_probs_row, k=k, dim=-1)
    ids_list = [int(value) for value in ids.detach().cpu().tolist()]
    vals_list = [float(value) for value in vals.detach().cpu().tolist()]
    probs = torch.exp(vals)
    return {
        "ids": ids_list,
        "logprobs": vals_list,
        "probs": [float(value) for value in probs.detach().cpu().tolist()],
        "mass": float(probs.sum().detach().cpu().item()),
        "top1_token_id": ids_list[0] if ids_list else None,
        "top1_prob": float(probs[0].detach().cpu().item()) if ids_list else None,
        "top1_top2_margin": vals_list[0] - vals_list[1] if len(vals_list) > 1 else None,
        "selected_rank_in_topk": next((idx + 1 for idx, token_id in enumerate(ids_list) if token_id == selected_id), None),
    }


def mass_on_ids(probs_row, token_ids: list[int]) -> float | None:
    import torch

    if not token_ids:
        return None
    ids = torch.tensor(token_ids, device=probs_row.device, dtype=torch.long)
    return float(probs_row.index_select(0, ids).sum().detach().cpu().item())


def set_overlap_stats(ids_a: list[int], ids_b: list[int]) -> tuple[int, float | None, float | None]:
    set_a = set(ids_a or [])
    set_b = set(ids_b or [])
    if not set_a or not set_b:
        return 0, None, None
    overlap = len(set_a & set_b)
    denom = min(len(set_a), len(set_b))
    union = len(set_a | set_b)
    return overlap, safe_div(overlap, denom), safe_div(overlap, union)


def score_logits_triplet(
    *,
    processor,
    prompt_len: int,
    response_ids,
    prompt_text: str,
    after_target_logits,
    base_target_logits,
    teacher_target_logits,
    after_target_ids,
    base_target_ids,
    teacher_target_ids,
    kl_chunk_size: int,
    topk: int,
) -> dict[str, Any]:
    import torch

    if after_target_ids.detach().cpu().tolist() != base_target_ids.detach().cpu().tolist():
        raise ValueError("after/base target ids differ after tokenization")
    if teacher_target_ids.detach().cpu().tolist() != base_target_ids.detach().cpu().tolist():
        raise ValueError("teacher/base target ids differ after tokenization")
    if int(after_target_logits.shape[0]) != int(base_target_logits.shape[0]):
        raise ValueError("after/base target length mismatch")
    if int(teacher_target_logits.shape[0]) != int(base_target_logits.shape[0]):
        raise ValueError("teacher/base target length mismatch")

    n_tokens = int(base_target_ids.shape[0])
    token_records: list[dict[str, Any]] = []
    selected_token_ids = [int(value) for value in response_ids.tolist()]

    for start in range(0, n_tokens, kl_chunk_size):
        end = min(start + kl_chunk_size, n_tokens)
        selected = base_target_ids[start:end].to(base_target_logits.device)
        base_lp = torch.log_softmax(base_target_logits[start:end].float(), dim=-1)
        after_lp = torch.log_softmax(after_target_logits[start:end].to(base_target_logits.device).float(), dim=-1)
        teacher_lp = torch.log_softmax(teacher_target_logits[start:end].to(base_target_logits.device).float(), dim=-1)
        base_p = base_lp.exp()
        after_p = after_lp.exp()
        teacher_p = teacher_lp.exp()
        dist_metrics = dist_triplet_metrics(after_lp, base_lp, teacher_lp)
        base_selected = base_lp.gather(1, selected.unsqueeze(1)).squeeze(1)
        after_selected = after_lp.gather(1, selected.unsqueeze(1)).squeeze(1)
        teacher_selected = teacher_lp.gather(1, selected.unsqueeze(1)).squeeze(1)
        base_ranks = exact_selected_rank(base_lp, selected)
        after_ranks = exact_selected_rank(after_lp, selected)
        teacher_ranks = exact_selected_rank(teacher_lp, selected)

        for offset in range(end - start):
            pos = start + offset
            token_id = int(selected[offset].detach().cpu().item())
            record: dict[str, Any] = {
                "step": pos,
                "response_position": int(prompt_len + pos),
                "token_id": token_id,
                "token_text": processor.decode([token_id], skip_special_tokens=False),
                "base_selected_logprob": float(base_selected[offset].detach().cpu().item()),
                "after_selected_logprob": float(after_selected[offset].detach().cpu().item()),
                "teacher_selected_logprob": float(teacher_selected[offset].detach().cpu().item()),
                "delta_logp_after_base": float((after_selected[offset] - base_selected[offset]).detach().cpu().item()),
                "base_nll": float((-base_selected[offset]).detach().cpu().item()),
                "after_nll": float((-after_selected[offset]).detach().cpu().item()),
                "teacher_nll": float((-teacher_selected[offset]).detach().cpu().item()),
                "base_selected_rank": int(base_ranks[offset].detach().cpu().item()),
                "after_selected_rank": int(after_ranks[offset].detach().cpu().item()),
                "teacher_selected_rank": int(teacher_ranks[offset].detach().cpu().item()),
            }
            for field, tensor in dist_metrics.items():
                record[field] = float(tensor[offset].detach().cpu().item())
            record["delta_entropy_after_base"] = record["after_entropy"] - record["base_entropy"]

            if topk > 0:
                base_top = topk_info(base_lp[offset], token_id, topk)
                after_top = topk_info(after_lp[offset], token_id, topk)
                teacher_top = topk_info(teacher_lp[offset], token_id, topk)
                for prefix, top in (("base", base_top), ("after", after_top), ("teacher", teacher_top)):
                    for key, value in top.items():
                        record[f"{prefix}_topk_{key}"] = value
                for a_prefix, a_top, b_prefix, b_top in (
                    ("after", after_top, "base", base_top),
                    ("after", after_top, "teacher", teacher_top),
                    ("base", base_top, "teacher", teacher_top),
                ):
                    overlap, ratio, jaccard = set_overlap_stats(a_top.get("ids", []), b_top.get("ids", []))
                    record[f"{a_prefix}_{b_prefix}_topk_overlap_count"] = overlap
                    record[f"{a_prefix}_{b_prefix}_topk_overlap_ratio"] = ratio
                    record[f"{a_prefix}_{b_prefix}_topk_jaccard"] = jaccard
                    record[f"{a_prefix}_{b_prefix}_top1_match"] = (
                        a_top.get("top1_token_id") == b_top.get("top1_token_id")
                    )
                record["base_top1_prob"] = base_top.get("top1_prob")
                record["after_top1_prob"] = after_top.get("top1_prob")
                record["teacher_top1_prob"] = teacher_top.get("top1_prob")
                record["base_top1_top2_margin"] = base_top.get("top1_top2_margin")
                record["after_top1_top2_margin"] = after_top.get("top1_top2_margin")
                record["teacher_top1_top2_margin"] = teacher_top.get("top1_top2_margin")
                if bool_finite(record.get("after_top1_prob")) and bool_finite(record.get("base_top1_prob")):
                    record["delta_top1_prob_after_base"] = record["after_top1_prob"] - record["base_top1_prob"]
                if bool_finite(record.get("after_top1_top2_margin")) and bool_finite(record.get("base_top1_top2_margin")):
                    record["delta_top1_margin_after_base"] = (
                        record["after_top1_top2_margin"] - record["base_top1_top2_margin"]
                    )
                record["after_mass_on_base_topk"] = mass_on_ids(after_p[offset], base_top.get("ids", []))
                record["base_mass_on_after_topk"] = mass_on_ids(base_p[offset], after_top.get("ids", []))
                record["after_mass_on_teacher_topk"] = mass_on_ids(after_p[offset], teacher_top.get("ids", []))
                record["base_mass_on_teacher_topk"] = mass_on_ids(base_p[offset], teacher_top.get("ids", []))
                record["teacher_mass_on_teacher_topk"] = teacher_top.get("mass")
                if bool_finite(record.get("after_mass_on_teacher_topk")) and bool_finite(record.get("base_mass_on_teacher_topk")):
                    record["delta_after_base_mass_on_teacher_topk"] = (
                        record["after_mass_on_teacher_topk"] - record["base_mass_on_teacher_topk"]
                    )
                record["emit_in_base_topk"] = bool(base_top.get("selected_rank_in_topk"))
                record["emit_in_after_topk"] = bool(after_top.get("selected_rank_in_topk"))
                record["emit_in_teacher_topk"] = bool(teacher_top.get("selected_rank_in_topk"))

            token_records.append(record)

    summary = summarize_rows(token_records, TOKEN_METRIC_FIELDS)
    summary["num_tokens"] = len(token_records)
    return {
        "prompt_text": prompt_text,
        "response_token_ids": selected_token_ids,
        "token_records": token_records,
        "summary": summary,
    }


def score_triplet_batch(
    *,
    base_model,
    base_processor,
    base_device,
    after_model,
    after_processor,
    after_device,
    original_images,
    teacher_images,
    prompt: str,
    response_texts: list[str],
    max_model_len: int,
    kl_chunk_size: int,
    topk: int,
) -> list[dict[str, Any]]:
    base_inputs, base_prompt_lens, base_response_ids, base_prompt_texts = tensorize_inputs_batch(
        base_processor, original_images, prompt, response_texts, base_device
    )
    teacher_inputs, teacher_prompt_lens, teacher_response_ids, _teacher_prompt_texts = tensorize_inputs_batch(
        base_processor, teacher_images, prompt, response_texts, base_device
    )
    after_inputs, after_prompt_lens, after_response_ids, _after_prompt_texts = tensorize_inputs_batch(
        after_processor, original_images, prompt, response_texts, after_device
    )
    for idx, (base_ids, teacher_ids, after_ids) in enumerate(zip(base_response_ids, teacher_response_ids, after_response_ids)):
        if base_ids.tolist() != teacher_ids.tolist():
            raise ValueError(f"base/teacher response tokenization differs for batch item {idx}")
        if base_ids.tolist() != after_ids.tolist():
            raise ValueError(f"base/after response tokenization differs for batch item {idx}")
        if base_prompt_lens[idx] != teacher_prompt_lens[idx]:
            raise ValueError(f"base/teacher prompt lengths differ for batch item {idx}")

    base_targets = forward_target_logits_batch(base_model, base_inputs, base_prompt_lens, max_model_len)
    teacher_targets = forward_target_logits_batch(base_model, teacher_inputs, teacher_prompt_lens, max_model_len)
    after_targets = forward_target_logits_batch(after_model, after_inputs, after_prompt_lens, max_model_len)

    scored = []
    for idx in range(len(response_texts)):
        base_logits, base_ids = base_targets[idx]
        teacher_logits, teacher_ids = teacher_targets[idx]
        after_logits, after_ids = after_targets[idx]
        scored.append(
            score_logits_triplet(
                processor=base_processor,
                prompt_len=base_prompt_lens[idx],
                response_ids=base_response_ids[idx],
                prompt_text=base_prompt_texts[idx],
                after_target_logits=after_logits,
                base_target_logits=base_logits,
                teacher_target_logits=teacher_logits,
                after_target_ids=after_ids,
                base_target_ids=base_ids,
                teacher_target_ids=teacher_ids,
                kl_chunk_size=kl_chunk_size,
                topk=topk,
            )
        )
    return scored


def build_trace_record(args: argparse.Namespace, record: dict[str, Any], caption: str, scored: dict[str, Any]) -> dict[str, Any]:
    image_id = record.get("image_id")
    metadata = {
        "uid": record.get("uid"),
        "index": record.get("index"),
        "data_source": record.get("data_source") or "coco_res_opd_eval",
        "image_id": image_id,
        "file_name": record.get("file_name"),
        "extra_info": {
            "image_id": image_id,
            "file_name": record.get("file_name"),
            "image_path": record.get("image_path"),
            "objects": record.get("gt_objects", []),
            "captions": record.get("gt_captions", []),
        },
    }
    return {
        "probe_type": "model_delta_logprob",
        "caption_source": args.caption_source_label or record.get("caption_source") or "base_original_caption",
        "base_model_path": args.base_model_path,
        "after_model_path": args.after_model_path,
        "after_name": args.after_name,
        "degradation_mode": args.degradation_mode,
        "student_ratio": args.student_ratio,
        "teacher_ratio": args.teacher_ratio,
        "target_px": args.target_px,
        "metadata": metadata,
        "response_text": caption,
        "response_token_ids": scored["response_token_ids"],
        "summary": scored["summary"],
        "token_records": scored["token_records"],
    }


def run_scoring(args: argparse.Namespace, output_jsonl: str) -> None:
    test_index = load_test_index(args.test_json)
    records = load_caption_records(args)
    if args.overwrite and os.path.exists(output_jsonl):
        os.remove(output_jsonl)
    done = existing_done_keys(output_jsonl)
    pending = [record for record in records if record_done_key(args, record) not in done]
    print(
        f"Shard {args.shard_index}/{args.num_shards}: records={len(records)} "
        f"pending={len(pending)} output={output_jsonl}"
    )
    if not pending:
        print("No pending records for this shard; skipping model load.")
        return

    base_model, base_processor, base_device = load_model_and_processor(args.base_model_path, args.torch_dtype)
    after_model, after_processor, after_device = load_model_and_processor(args.after_model_path, args.torch_dtype)

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    started = time.time()
    written = 0
    failures = 0
    skipped = 0
    with open(output_jsonl, "a", encoding="utf-8") as f:
        for batch_start in range(0, len(pending), args.batch_size):
            batch_records = pending[batch_start : batch_start + args.batch_size]
            batch_items = []
            for record in batch_records:
                caption = record.get("generated_caption") or ""
                image_path = resolve_image_path(record, test_index, args.image_root)
                if not image_path:
                    print(f"WARNING: missing image path for image_id={record.get('image_id')}", file=sys.stderr)
                    failures += 1
                    continue
                try:
                    original_image = load_view_image(
                        image_path,
                        args.degradation_mode,
                        0,
                        args.target_px,
                        args.student_ratio,
                        blank_when_px_zero=False,
                    )
                    teacher_image = load_view_image(
                        image_path,
                        args.degradation_mode,
                        0,
                        args.target_px,
                        args.teacher_ratio,
                        blank_when_px_zero=False,
                    )
                except Exception as exc:
                    print(f"WARNING: failed loading image_id={record.get('image_id')}: {exc}", file=sys.stderr)
                    failures += 1
                    continue
                batch_items.append(
                    {
                        "record": record,
                        "caption": caption,
                        "original_image": original_image,
                        "teacher_image": teacher_image,
                    }
                )
            if not batch_items:
                continue

            try:
                scored_batch = score_triplet_batch(
                    base_model=base_model,
                    base_processor=base_processor,
                    base_device=base_device,
                    after_model=after_model,
                    after_processor=after_processor,
                    after_device=after_device,
                    original_images=[item["original_image"] for item in batch_items],
                    teacher_images=[item["teacher_image"] for item in batch_items],
                    prompt=args.prompt,
                    response_texts=[item["caption"] for item in batch_items],
                    max_model_len=args.max_model_len,
                    kl_chunk_size=args.kl_chunk_size,
                    topk=args.topk,
                )
            except Exception as batch_exc:
                print(
                    f"WARNING: batch scoring failed at batch_start={batch_start}: {batch_exc}; "
                    "falling back to per-sample scoring",
                    file=sys.stderr,
                )
                scored_batch = []
                for item in batch_items:
                    try:
                        scored = score_triplet_batch(
                            base_model=base_model,
                            base_processor=base_processor,
                            base_device=base_device,
                            after_model=after_model,
                            after_processor=after_processor,
                            after_device=after_device,
                            original_images=[item["original_image"]],
                            teacher_images=[item["teacher_image"]],
                            prompt=args.prompt,
                            response_texts=[item["caption"]],
                            max_model_len=args.max_model_len,
                            kl_chunk_size=args.kl_chunk_size,
                            topk=args.topk,
                        )[0]
                    except Exception as exc:
                        print(
                            f"WARNING: failed scoring image_id={item['record'].get('image_id')}: {exc}",
                            file=sys.stderr,
                        )
                        failures += 1
                        scored = None
                    scored_batch.append(scored)

            for item, scored in zip(batch_items, scored_batch):
                if scored is None:
                    continue
                out = build_trace_record(args, item["record"], item["caption"], scored)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                done.add(record_done_key(args, item["record"]))
                written += 1
            f.flush()
            if written and written % 10 == 0:
                elapsed = time.time() - started
                print(f"Scored {written} records ({elapsed:.0f}s, skipped={skipped}, failures={failures})")
    print(f"Scoring complete: written={written} skipped={skipped} failures={failures}")


def summarize_rows(rows: list[dict[str, Any]], metric_fields: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows)}
    image_ids = {row.get("image_id") for row in rows if row.get("image_id") is not None}
    if image_ids:
        out["unique_images"] = len(image_ids)
    for field in metric_fields:
        values = [row.get(field) for row in rows if bool_finite(row.get(field))]
        if values:
            out[field] = safe_mean(values)
            out[f"{field}_p50"] = percentile(values, 0.50)
            out[f"{field}_p75"] = percentile(values, 0.75)
            out[f"{field}_p90"] = percentile(values, 0.90)
    for field in BOOLEAN_FIELDS:
        values = []
        for row in rows:
            value = row.get(field)
            if value is True:
                values.append(1.0)
            elif value is False:
                values.append(0.0)
        if values:
            out[f"{field}_rate"] = safe_mean(values)
    if rows:
        mechanisms = Counter(row.get("mechanism", "unknown") for row in rows)
        for label, count in mechanisms.items():
            out[f"mechanism_{label}_rate"] = safe_div(count, len(rows)) or 0.0
    return out


def group_summary(rows: list[dict[str, Any]], key: str, metric_fields: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return {
        group: summarize_rows(group_rows, metric_fields)
        for group, group_rows in sorted(grouped.items(), key=lambda item: item[0])
    }


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


def assign_risk_scores(tokens: list[dict[str, Any]], top_p: float, random_seed: int) -> dict[str, Any]:
    rng = random.Random(random_seed)
    for in_field, out_field in (
        ("rkl_base_to_teacher", "_rank_rkl_base_teacher"),
        ("base_nll", "_rank_base_nll"),
        ("base_entropy", "_rank_base_entropy"),
    ):
        valid = [(idx, float(token[in_field])) for idx, token in enumerate(tokens) if bool_finite(token.get(in_field))]
        ranks = percentile_ranks([value for _, value in valid])
        for (idx, _), rank in zip(valid, ranks):
            tokens[idx][out_field] = rank
    for token in tokens:
        rkl_rank = token.get("_rank_rkl_base_teacher")
        nll_rank = token.get("_rank_base_nll")
        entropy_rank = token.get("_rank_base_entropy")
        if bool_finite(rkl_rank) and bool_finite(nll_rank):
            token["riskmask_nll_score"] = float(rkl_rank) * float(nll_rank)
        if bool_finite(rkl_rank) and bool_finite(entropy_rank):
            token["riskmask_entropy_score"] = float(rkl_rank) * float(entropy_rank)
        token["random_score"] = rng.random()

    out: dict[str, Any] = {}
    for score_field, selected_field in (
        ("riskmask_nll_score", "riskmask_nll_selected"),
        ("riskmask_entropy_score", "riskmask_entropy_selected"),
        ("random_score", "random_selected"),
    ):
        valid = [token for token in tokens if bool_finite(token.get(score_field))]
        for token in tokens:
            token[selected_field] = False
        if not valid:
            out[score_field] = {"valid_tokens": 0, "selected_tokens": 0, "threshold": None}
            continue
        k = max(1, int(math.ceil(len(valid) * top_p)))
        ordered = sorted(valid, key=lambda token: float(token[score_field]), reverse=True)
        threshold = float(ordered[k - 1][score_field])
        selected = [token for token in valid if float(token[score_field]) >= threshold]
        for token in selected:
            token[selected_field] = True
        out[score_field] = {
            "valid_tokens": len(valid),
            "selected_tokens": len(selected),
            "selected_fraction": safe_div(len(selected), len(valid)) or 0.0,
            "threshold": threshold,
        }
    return out


def classify_mechanisms(tokens: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    jsd_values = [abs(float(token["jsd_after_base"])) for token in tokens if bool_finite(token.get("jsd_after_base"))]
    abs_logp_values = [
        abs(float(token["delta_logp_after_base"]))
        for token in tokens
        if bool_finite(token.get("delta_logp_after_base"))
    ]
    jsd_p90 = percentile(jsd_values, 0.90) or 0.0
    abs_logp_p90 = percentile(abs_logp_values, 0.90) or 0.0
    logp_eps = max(args.logp_eps, abs_logp_p90 * 0.10)
    for token in tokens:
        same_top1 = token.get("after_base_top1_match")
        topk_jaccard = token.get("after_base_topk_jaccard")
        delta_entropy = token.get("delta_entropy_after_base")
        delta_top1_prob = token.get("delta_top1_prob_after_base")
        delta_margin = token.get("delta_top1_margin_after_base")
        delta_logp = token.get("delta_logp_after_base")
        jsd = token.get("jsd_after_base")

        reshaped = False
        if same_top1 is False:
            reshaped = True
        if bool_finite(topk_jaccard) and float(topk_jaccard) < args.topk_jaccard_threshold:
            reshaped = True
        if bool_finite(jsd) and float(jsd) >= jsd_p90 and same_top1 is False:
            reshaped = True

        sharpened = (
            same_top1 is True
            and bool_finite(delta_entropy)
            and bool_finite(delta_top1_prob)
            and float(delta_entropy) <= -args.entropy_eps
            and float(delta_top1_prob) >= args.prob_eps
        )
        if not sharpened and same_top1 is True and bool_finite(delta_entropy) and bool_finite(delta_margin):
            sharpened = float(delta_entropy) <= -args.entropy_eps and float(delta_margin) >= args.logp_eps

        if reshaped:
            token["mechanism"] = "reshape_topk"
        elif sharpened:
            token["mechanism"] = "sharpen_same_top1"
        elif bool_finite(delta_logp) and float(delta_logp) <= -logp_eps:
            token["mechanism"] = "suppress_emitted"
        elif bool_finite(delta_logp) and float(delta_logp) >= logp_eps:
            token["mechanism"] = "boost_emitted"
        else:
            token["mechanism"] = "stable"
    return {
        "jsd_after_base_p90": jsd_p90,
        "abs_delta_logp_after_base_p90": abs_logp_p90,
        "effective_logp_eps": logp_eps,
        "entropy_eps": args.entropy_eps,
        "prob_eps": args.prob_eps,
        "topk_jaccard_threshold": args.topk_jaccard_threshold,
    }


def flatten_tokens(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata") or {}
        image_id = metadata.get("image_id")
        for idx, token in enumerate(record.get("token_records") or []):
            token["_record_index"] = record["_record_index"]
            token["_token_index"] = idx
            token["image_id"] = image_id
            token["caption_source"] = record.get("caption_source")
            rows.append(token)
    return rows


def extract_objects_from_text(text: str, canonical_synonym_map: dict[str, list[str]]) -> set[str]:
    import re

    lowered = (text or "").lower()
    objects = set()
    candidates = []
    for canonical, terms in canonical_synonym_map.items():
        for term in terms:
            candidates.append((canonical, term))
    candidates.sort(key=lambda item: (-len(item[1]), item[0], item[1]))
    for canonical, term in candidates:
        if not term:
            continue
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            objects.add(canonical)
    return objects


def load_after_object_index(path: str | None, test_json: str, caption_field: str, canonical_synonym_map: dict[str, list[str]]) -> dict[int, set[str]]:
    if not path:
        return {}
    test_index = load_test_index(test_json)
    out: dict[int, set[str]] = {}
    for raw in load_jsonl(path):
        record = normalize_generation_record(raw, test_index, caption_field)
        image_id = as_int(record.get("image_id"))
        if image_id is None:
            continue
        caption = record.get("generated_caption") or ""
        if caption and not caption.startswith("[ERROR]"):
            out[image_id] = extract_objects_from_text(caption, canonical_synonym_map)
    if not out:
        fail(f"--after-eval-results contained no usable caption objects: {path}")
    return out


def span_tokens(record: dict[str, Any], mention: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = record.get("token_records") or []
    start = int(mention.get("token_start", 0) or 0)
    end = int(mention.get("token_end", start) or start)
    start = max(0, min(start, len(tokens)))
    end = max(start, min(end, len(tokens)))
    return tokens[start:end]


def enrich_mentions(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, inverse_synonym_dict = parse_official_synonyms()
    canonical_synonym_map = build_canonical_synonym_map(inverse_synonym_dict)
    after_objects = load_after_object_index(
        args.after_eval_results,
        args.test_json,
        args.caption_field,
        canonical_synonym_map,
    )
    by_index = {record["_record_index"]: record for record in records}
    mentions: list[dict[str, Any]] = []
    for record in records:
        record_mentions = find_object_mentions(record, canonical_synonym_map)
        add_non_object_token_roles(record, record_mentions)
        image_id = as_int((record.get("metadata") or {}).get("image_id"))
        after_set = after_objects.get(image_id, set())
        for mention in record_mentions:
            row = dict(mention)
            row["_record_index"] = record["_record_index"]
            row["image_id"] = row.get("image_id", image_id)
            if after_objects:
                obj = row.get("canonical_object")
                object_type = row.get("object_type")
                kept = obj in after_set
                if object_type == "correct_object":
                    row["after_change_status"] = "kept_correct" if kept else "removed_correct"
                elif object_type == "hallucinated_object":
                    row["after_change_status"] = "kept_hallucinated" if kept else "removed_hallucinated"
                else:
                    row["after_change_status"] = "unknown_object_status"
            tokens = span_tokens(by_index[row["_record_index"]], row)
            row["_span_token_count"] = len(tokens)
            row.update(aggregate_token_span(tokens))
            mentions.append(row)
        record["object_mentions"] = record_mentions
    meta = {
        "after_eval_results": args.after_eval_results,
        "after_object_records": len(after_objects),
    }
    return mentions, meta


def aggregate_token_span(tokens: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in TOKEN_METRIC_FIELDS:
        values = [token.get(field) for token in tokens if bool_finite(token.get(field))]
        if values:
            out[f"{field}_mean"] = safe_mean(values)
            out[f"{field}_max"] = max(values)
            out[f"{field}_min"] = min(values)
    for field in BOOLEAN_FIELDS:
        values = []
        for token in tokens:
            value = token.get(field)
            if value is True:
                values.append(1.0)
            elif value is False:
                values.append(0.0)
        if values:
            out[f"{field}_frac"] = safe_mean(values)
    if tokens:
        out["riskmask_nll_selected_token_frac"] = safe_mean(
            [1.0 if token.get("riskmask_nll_selected") else 0.0 for token in tokens]
        )
        out["riskmask_entropy_selected_token_frac"] = safe_mean(
            [1.0 if token.get("riskmask_entropy_selected") else 0.0 for token in tokens]
        )
        mechanisms = Counter(token.get("mechanism", "unknown") for token in tokens)
        for label in MECHANISM_LABELS:
            out[f"mechanism_{label}_frac"] = safe_div(mechanisms.get(label, 0), len(tokens)) or 0.0
    return out


def labeled_mentions(mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in mentions if m.get("object_type") in {"correct_object", "hallucinated_object"}]


def risk_selection_summary(rows: list[dict[str, Any]], selected_field: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get(selected_field)]
    unselected = [row for row in rows if not row.get(selected_field)]
    return {
        "selected": summarize_rows(selected, TOKEN_METRIC_FIELDS),
        "unselected": summarize_rows(unselected, TOKEN_METRIC_FIELDS),
    }


def make_examples(mentions: list[dict[str, Any]], limit: int = 25) -> dict[str, list[dict[str, Any]]]:
    def clean(row: dict[str, Any]) -> dict[str, Any]:
        keep = [
            "image_id",
            "object_type",
            "after_change_status",
            "canonical_object",
            "mention_text",
            "context",
            "token_start",
            "token_end",
            "_span_token_count",
            "riskmask_nll_score_mean",
            "rkl_base_to_teacher_mean",
            "base_nll_mean",
            "base_entropy_mean",
            "rkl_after_to_base_mean",
            "jsd_after_base_mean",
            "delta_logp_after_base_mean",
            "delta_entropy_after_base_mean",
            "after_base_topk_jaccard_mean",
            "after_base_top1_match_frac",
            "riskmask_nll_selected_token_frac",
        ]
        return {key: row.get(key) for key in keep if key in row}

    labeled = labeled_mentions(mentions)
    examples = {
        "removed_hallucinated_suppressed": sorted(
            [
                m for m in labeled
                if m.get("after_change_status") == "removed_hallucinated"
                and bool_finite(m.get("delta_logp_after_base_mean"))
            ],
            key=lambda m: (float(m.get("delta_logp_after_base_mean", 0.0)), -float(m.get("riskmask_nll_score_mean") or 0.0)),
        )[:limit],
        "kept_correct_stable": sorted(
            [
                m for m in labeled
                if m.get("after_change_status") == "kept_correct"
                and bool_finite(m.get("delta_logp_after_base_mean"))
            ],
            key=lambda m: (abs(float(m.get("delta_logp_after_base_mean", 0.0))), float(m.get("jsd_after_base_mean") or 0.0)),
        )[:limit],
        "reshaped_hallucinated": sorted(
            [
                m for m in labeled
                if m.get("object_type") == "hallucinated_object"
                and bool_finite(m.get("jsd_after_base_mean"))
            ],
            key=lambda m: float(m.get("jsd_after_base_mean") or 0.0),
            reverse=True,
        )[:limit],
        "failure_removed_correct": sorted(
            [
                m for m in labeled
                if m.get("after_change_status") == "removed_correct"
                and bool_finite(m.get("delta_logp_after_base_mean"))
            ],
            key=lambda m: float(m.get("delta_logp_after_base_mean") or 0.0),
        )[:limit],
        "failure_kept_hallucinated": sorted(
            [
                m for m in labeled
                if m.get("after_change_status") == "kept_hallucinated"
                and bool_finite(m.get("riskmask_nll_score_mean"))
            ],
            key=lambda m: float(m.get("riskmask_nll_score_mean") or 0.0),
            reverse=True,
        )[:limit],
    }
    return {key: [clean(row) for row in rows] for key, rows in examples.items()}


def write_examples_jsonl(examples: dict[str, list[dict[str, Any]]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for group, rows in examples.items():
            for row in rows:
                out = dict(row)
                out["example_group"] = group
                f.write(json.dumps(out, ensure_ascii=False) + "\n")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("\n", " ").replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def compact_row(name: str, stats: dict[str, Any]) -> list[Any]:
    return [
        name,
        stats.get("count", 0),
        fmt(stats.get("rkl_base_to_teacher")),
        fmt(stats.get("base_nll")),
        fmt(stats.get("base_entropy")),
        fmt(stats.get("rkl_after_to_base")),
        fmt(stats.get("jsd_after_base")),
        fmt(stats.get("delta_logp_after_base")),
        fmt(stats.get("delta_entropy_after_base")),
        fmt(stats.get("after_base_top1_match_rate")),
        fmt(stats.get("after_base_topk_jaccard")),
        fmt(stats.get("mechanism_sharpen_same_top1_rate")),
        fmt(stats.get("mechanism_reshape_topk_rate")),
        fmt(stats.get("mechanism_suppress_emitted_rate")),
    ]


def write_markdown(summary: dict[str, Any], path: str) -> None:
    lines = [
        "# Model-Delta Logprob Probe",
        "",
        "## Purpose",
        "",
        "This probe explains P2/P3: given base-generated captions, it compares the trained model against the base model on the original image, while keeping the low-resolution base view as the RKL gate reference.",
        "",
        "Important convention: `RKL(after || base)`, `KL(base || after)`, and `JSD(after, base)` are all reported for the actual post-training distribution shift. `RKL(base || low-res teacher)` is the pre-training RiskMask gate signal.",
        "",
        "## Inputs",
        "",
        f"- trace_jsonl: `{summary.get('trace_jsonl')}`",
        f"- after_name: `{summary.get('config', {}).get('after_name')}`",
        f"- eval_results: `{summary.get('config', {}).get('eval_results')}`",
        f"- after_eval_results: `{summary.get('config', {}).get('after_eval_results')}`",
        f"- records: {summary.get('num_records')}",
        f"- tokens: {summary.get('num_tokens')}",
        f"- object_mentions: {summary.get('num_object_mentions')}",
        "",
        "## Token Groups",
        "",
    ]
    headers = [
        "group",
        "N",
        "gate RKL",
        "base NLL",
        "base Ent.",
        "RKL(after,base)",
        "JSD(after,base)",
        "Delta logp",
        "Delta Ent.",
        "Top1 same",
        "TopK Jac.",
        "Sharpen",
        "Reshape",
        "Suppress",
    ]
    token_rows = []
    for group in ("correct_object", "hallucinated_object", "non_object_token", "unknown_token"):
        stats = summary.get("token_role_summary", {}).get(group)
        if stats:
            token_rows.append(compact_row(group, stats))
    lines.append(markdown_table(headers, token_rows))

    lines.extend(["", "## RiskMask-NLL Top-p vs Rest", ""])
    risk_stats = summary.get("riskmask_nll_token_summary", {})
    risk_rows = []
    for group in ("selected", "unselected"):
        stats = risk_stats.get(group)
        if stats:
            risk_rows.append(compact_row(group, stats))
    lines.append(markdown_table(headers, risk_rows))

    lines.extend(["", "## Object Mention Groups", ""])
    mention_rows = []
    mention_headers = [
        "group",
        "N",
        "gate RKL",
        "base NLL",
        "RKL(after,base)",
        "JSD(after,base)",
        "Delta logp",
        "Delta Ent.",
        "selected frac",
        "Sharpen",
        "Reshape",
        "Suppress",
    ]
    for group in ("correct_object", "hallucinated_object"):
        stats = summary.get("mention_object_type_summary", {}).get(group)
        if not stats:
            continue
        mention_rows.append([
            group,
            stats.get("count", 0),
            fmt(stats.get("rkl_base_to_teacher_mean")),
            fmt(stats.get("base_nll_mean")),
            fmt(stats.get("rkl_after_to_base_mean")),
            fmt(stats.get("jsd_after_base_mean")),
            fmt(stats.get("delta_logp_after_base_mean")),
            fmt(stats.get("delta_entropy_after_base_mean")),
            fmt(stats.get("riskmask_nll_selected_token_frac")),
            fmt(stats.get("mechanism_sharpen_same_top1_frac")),
            fmt(stats.get("mechanism_reshape_topk_frac")),
            fmt(stats.get("mechanism_suppress_emitted_frac")),
        ])
    lines.append(markdown_table(mention_headers, mention_rows))

    if summary.get("mention_change_summary"):
        lines.extend(["", "## Base Mention Fate Under Trained Model", ""])
        change_rows = []
        for group, stats in summary["mention_change_summary"].items():
            change_rows.append([
                group,
                stats.get("count", 0),
                fmt(stats.get("rkl_base_to_teacher_mean")),
                fmt(stats.get("base_nll_mean")),
                fmt(stats.get("delta_logp_after_base_mean")),
                fmt(stats.get("jsd_after_base_mean")),
                fmt(stats.get("riskmask_nll_selected_token_frac")),
            ])
        lines.append(markdown_table(
            ["fate", "N", "gate RKL", "base NLL", "Delta logp", "JSD(after,base)", "selected frac"],
            change_rows,
        ))

    lines.extend(["", "## Mechanism Counts", ""])
    mechanism_rows = []
    for label, count in summary.get("mechanism_counts", {}).items():
        mechanism_rows.append([label, count, fmt(safe_div(count, summary.get("num_tokens") or 0))])
    lines.append(markdown_table(["mechanism", "tokens", "rate"], mechanism_rows))

    lines.extend(["", "## Example Files", ""])
    lines.append(f"- examples_jsonl: `{summary.get('examples_jsonl')}`")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def analyze_trace(args: argparse.Namespace, output_jsonl: str, examples_jsonl: str) -> dict[str, Any]:
    records = []
    for idx, record in enumerate(load_jsonl(output_jsonl)):
        record = dict(record)
        record["_record_index"] = idx
        records.append(record)
    if not records:
        fail(f"Trace JSONL is empty: {output_jsonl}")
    tokens = flatten_tokens(records)
    if not tokens:
        fail(f"Trace JSONL has no token_records: {output_jsonl}")
    selection_meta = assign_risk_scores(tokens, args.top_p, args.random_seed)
    mechanism_meta = classify_mechanisms(tokens, args)
    mentions, mention_meta = enrich_mentions(records, args)
    labeled = labeled_mentions(mentions)
    examples = make_examples(mentions)
    write_examples_jsonl(examples, examples_jsonl)

    mechanism_counts = Counter(token.get("mechanism", "unknown") for token in tokens)
    summary = {
        "trace_jsonl": output_jsonl,
        "examples_jsonl": examples_jsonl,
        "num_records": len(records),
        "num_tokens": len(tokens),
        "num_object_mentions": len(mentions),
        "num_labeled_object_mentions": len(labeled),
        "token_role_summary": group_summary(tokens, "token_role", TOKEN_METRIC_FIELDS),
        "riskmask_nll_token_summary": risk_selection_summary(tokens, "riskmask_nll_selected"),
        "riskmask_entropy_token_summary": risk_selection_summary(tokens, "riskmask_entropy_selected"),
        "mention_object_type_summary": group_summary(labeled, "object_type", [f"{field}_mean" for field in TOKEN_METRIC_FIELDS]),
        "mention_change_summary": group_summary(labeled, "after_change_status", [f"{field}_mean" for field in TOKEN_METRIC_FIELDS])
        if mention_meta.get("after_object_records")
        else {},
        "mechanism_counts": dict(mechanism_counts),
        "risk_selection": selection_meta,
        "mechanism_thresholds": mechanism_meta,
        "mention_meta": mention_meta,
        "example_groups": {key: len(value) for key, value in examples.items()},
    }
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_jsonl, summary_json, summary_md, examples_jsonl = infer_output_paths(args)
    downloaded = prepare_oss_checkpoint(args)
    try:
        if not args.analyze_only:
            run_scoring(args, output_jsonl)
        if args.score_only:
            return
        summary = analyze_trace(args, output_jsonl, examples_jsonl)
        summary["config"] = {
            "base_model_path": args.base_model_path,
            "after_model_path": args.after_model_path,
            "after_name": args.after_name,
            "after_oss_checkpoint": args.after_oss_checkpoint,
            "eval_results": args.eval_results,
            "after_eval_results": args.after_eval_results,
            "student_ratio": args.student_ratio,
            "teacher_ratio": args.teacher_ratio,
            "degradation_mode": args.degradation_mode,
            "target_px": args.target_px,
            "top_p": args.top_p,
            "topk": args.topk,
            "kl_chunk_size": args.kl_chunk_size,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        }
        os.makedirs(os.path.dirname(os.path.abspath(summary_json)), exist_ok=True)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        write_markdown(summary, summary_md)
        print(f"Saved summary JSON: {summary_json}")
        print(f"Saved summary Markdown: {summary_md}")
        print(f"Saved examples JSONL: {examples_jsonl}")
    finally:
        if args.cleanup_after:
            for path in downloaded:
                cleanup_local_checkpoint(path)


if __name__ == "__main__":
    main()
