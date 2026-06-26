#!/usr/bin/env python3
"""
Probe whether token-level KL/RKL/JSD is enriched on hallucinated object mentions.

This script implements the diagnostics described in:

    res-opd/plan/same_image_rkl_signal.md

It forced-scores fixed captions with two distributions and computes exact
full-vocabulary per-token divergences:

    RKL_current = KL(p_student || p_teacher)
    FKL         = KL(p_teacher || p_student)
    JSD         = 0.5 KL(p_student || m) + 0.5 KL(p_teacher || m)

Supported pair modes:

    duplicate_base:
        same model, same original image, two separate forward passes.

    dual_view_same_model:
        same model, student image view vs teacher/degraded image view.

    dual_model_same_image:
        trained student checkpoint vs frozen/base teacher checkpoint,
        usually with both image ratios set to 1.0.

The output JSONL stores token records and object mention spans. The summary
tests whether high-divergence object mentions have higher hallucination rate.
"""

import argparse
import glob
import json
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
    parse_official_synonyms,
    try_singularize,
)
from score_opd_eval_trace import (  # noqa: E402
    DEFAULT_COCO_VAL_ROOT,
    PROMPT_TEXT,
    as_int,
    first_nonempty,
    load_jsonl,
    load_test_index,
    load_view_image,
    normalize_generation_record,
    record_value,
    resolve_image_path,
)


DEFAULT_SERVER_BASE_ORIGINAL_RESULTS = (
    "/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/"
    "res-opd/eval_results/latest/full/Qwen3VL-2B-Instruct/"
    "train5000_test1000_original_sr1p0/eval_results.jsonl"
)
DEFAULT_SERVER_BASE_LOWRES_DIR = (
    "/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/"
    "res-opd/eval_results/latest/full/Qwen3VL-2B-Instruct/"
    "train5000_test1000_original_sr0p75"
)
DEFAULT_TEST_JSON = os.path.join(RES_OPD_ROOT, "data", "test_1000.json")
DEFAULT_OUTPUT_ROOT = os.path.join(PROBE_DIR, "results", "same_image_rkl_signal")

DIVERGENCE_FIELDS = [
    "rkl_student_to_teacher",
    "fkl_teacher_to_student",
    "jsd",
]
TOKEN_METRIC_FIELDS = DIVERGENCE_FIELDS + [
    "student_nll",
    "teacher_nll",
    "student_selected_logprob",
    "teacher_selected_logprob",
    "student_minus_teacher_selected_logprob",
    "student_entropy",
    "teacher_entropy",
    "student_top1_top2_margin",
    "teacher_top1_top2_margin",
    "topk_overlap_ratio",
    "topk_jaccard",
    "top1_match",
]
SPAN_AGG_FIELDS = [
    "rkl_student_to_teacher_mean",
    "rkl_student_to_teacher_max",
    "fkl_teacher_to_student_mean",
    "fkl_teacher_to_student_max",
    "jsd_mean",
    "jsd_max",
    "student_nll_mean",
    "student_entropy_mean",
    "teacher_entropy_mean",
    "student_minus_teacher_selected_logprob_mean",
    "topk_overlap_ratio_mean",
    "top1_match_frac",
]
HIGH_BUCKET_METRICS = [
    "rkl_student_to_teacher_mean",
    "rkl_student_to_teacher_max",
    "fkl_teacher_to_student_mean",
    "fkl_teacher_to_student_max",
    "jsd_mean",
    "jsd_max",
    "student_nll_mean",
    "student_entropy_mean",
]
HIGH_BUCKET_FRACTIONS = [0.10, 0.20, 0.25]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute same-image / low-res KL signal and object-level hallucination enrichment."
    )
    parser.add_argument(
        "--pair-mode",
        choices=["duplicate_base", "dual_view_same_model", "dual_model_same_image"],
        required=True,
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Alias for --student-model-path; useful for duplicate_base and dual_view_same_model.",
    )
    parser.add_argument("--student-model-path", default=None)
    parser.add_argument("--teacher-model-path", default=None)
    parser.add_argument(
        "--student-oss-checkpoint",
        default=None,
        help=(
            "Optional OSS path for the student checkpoint. If --student-model-path "
            "does not already contain weights, ossutil cp -r will download it there."
        ),
    )
    parser.add_argument(
        "--teacher-oss-checkpoint",
        default=None,
        help=(
            "Optional OSS path for the teacher checkpoint. Usually not needed for "
            "the local frozen base model."
        ),
    )
    parser.add_argument(
        "--cleanup-after",
        action="store_true",
        help="Remove checkpoint directories downloaded from OSS after scoring and summary are complete.",
    )
    parser.add_argument(
        "--eval-results",
        default=DEFAULT_SERVER_BASE_ORIGINAL_RESULTS,
        help=(
            "Caption source eval_results.jsonl. Defaults to the server full5k/test1000 "
            "base original-results path."
        ),
    )
    parser.add_argument("--test-json", default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", default=DEFAULT_COCO_VAL_ROOT)
    parser.add_argument("--prompt", default=PROMPT_TEXT)
    parser.add_argument("--caption-field", default="auto")
    parser.add_argument("--caption-source-label", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip model scoring and summarize an existing --output-jsonl.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=9728)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--kl-chunk-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=20, help="Store top-k diagnostics; 0 disables top-k fields.")
    parser.add_argument("--degradation-mode", choices=["original", "square"], default="original")
    parser.add_argument("--student-ratio", type=float, default=1.0)
    parser.add_argument("--teacher-ratio", type=float, default=1.0)
    parser.add_argument("--student-px", type=int, default=0)
    parser.add_argument("--teacher-px", type=int, default=0)
    parser.add_argument("--target-px", type=int, default=448)
    parser.add_argument(
        "--print-server-paths",
        action="store_true",
        help="Print the default server original/lowres eval result locations and exit.",
    )
    return parser.parse_args()


def bool_finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def safe_div(num, den):
    return num / den if den else None


def safe_rate(num, den):
    return num / den if den else 0.0


def safe_mean(values):
    vals = [float(value) for value in values if bool_finite(value)]
    return sum(vals) / len(vals) if vals else None


def percentile(values, q):
    vals = sorted(float(value) for value in values if bool_finite(value))
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
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def safe_path_part(value):
    value = str(value or "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def ratio_tag(value):
    return str(value).replace(".", "p")


def infer_output_paths(args):
    eval_dir = os.path.basename(os.path.dirname(os.path.abspath(args.eval_results or "eval_results.jsonl")))
    student_name = os.path.basename(os.path.normpath(args.student_model_path or args.model_path or "model"))
    teacher_name = os.path.basename(os.path.normpath(args.teacher_model_path or args.student_model_path or args.model_path or "teacher"))
    run_name = safe_path_part(
        f"{args.pair_mode}__{student_name}__teacher_{teacher_name}__{eval_dir}"
        f"__sr{ratio_tag(args.student_ratio)}_tr{ratio_tag(args.teacher_ratio)}"
    )
    output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, run_name)
    output_jsonl = args.output_jsonl or os.path.join(output_dir, "pair_kl_trace.jsonl")
    summary_json = args.summary_json or os.path.join(output_dir, "pair_kl_object_summary.json")
    summary_md = args.summary_md or os.path.join(output_dir, "pair_kl_object_summary.md")
    return output_jsonl, summary_json, summary_md


def validate_args(args):
    if args.print_server_paths:
        return
    if not args.student_model_path:
        args.student_model_path = args.model_path
    if not args.student_model_path and not args.analyze_only:
        raise SystemExit("--student-model-path or --model-path is required unless --analyze-only is set.")
    if args.pair_mode == "dual_model_same_image" and not args.teacher_model_path and not args.analyze_only:
        raise SystemExit("--teacher-model-path is required for --pair-mode dual_model_same_image.")
    if args.student_oss_checkpoint and not args.student_model_path and not args.analyze_only:
        raise SystemExit("--student-model-path is required when --student-oss-checkpoint is set.")
    if args.teacher_oss_checkpoint and not args.teacher_model_path and not args.analyze_only:
        raise SystemExit("--teacher-model-path is required when --teacher-oss-checkpoint is set.")
    if args.pair_mode in {"duplicate_base", "dual_view_same_model"} and not args.teacher_model_path:
        args.teacher_model_path = args.student_model_path
    if args.kl_chunk_size <= 0:
        raise SystemExit("--kl-chunk-size must be positive.")


def checkpoint_has_weights(local_path):
    if not local_path or not os.path.isdir(local_path):
        return False
    weight_suffixes = (".safetensors", ".bin", ".pt", ".pth")
    for root, _, files in os.walk(local_path):
        for name in files:
            if name.endswith(weight_suffixes):
                return True
    return False


def fetch_checkpoint_from_oss(oss_path, local_path):
    import subprocess

    if checkpoint_has_weights(local_path):
        print(f"[OSS] Checkpoint already exists at {local_path}, skipping download.")
        return False

    print(f"[OSS] Downloading checkpoint from {oss_path} to {local_path} ...")
    os.makedirs(local_path, exist_ok=True)
    result = subprocess.run(
        ["ossutil", "cp", "-r", f"{oss_path.rstrip('/')}/", f"{local_path.rstrip('/')}/", "-f"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ossutil cp failed for {oss_path} -> {local_path} "
            f"(exit {result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    print(f"[OSS] Download complete: {local_path}")
    return True


def cleanup_local_checkpoint(local_path):
    import shutil

    if os.path.isdir(local_path):
        print(f"[Cleanup] Removing downloaded checkpoint: {local_path}")
        shutil.rmtree(local_path)
    else:
        print(f"[Cleanup] Downloaded checkpoint path not found, skipping: {local_path}")


def prepare_oss_checkpoints(args):
    downloaded = []
    if args.analyze_only:
        return downloaded
    if args.student_oss_checkpoint:
        did_download = fetch_checkpoint_from_oss(args.student_oss_checkpoint, args.student_model_path)
        if did_download:
            downloaded.append(os.path.abspath(args.student_model_path))
    if args.teacher_oss_checkpoint:
        did_download = fetch_checkpoint_from_oss(args.teacher_oss_checkpoint, args.teacher_model_path)
        if did_download:
            downloaded.append(os.path.abspath(args.teacher_model_path))
    return downloaded


def load_model_and_processor(model_path, torch_dtype_name):
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForVision2Seq
        model_cls = AutoModelForVision2Seq

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[torch_dtype_name]
    print(f"Loading model: {model_path}")
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    device = next(model.parameters()).device
    print(f"Model first device: {device}")
    return model, processor, device


def tensorize_inputs(processor, image, prompt, response_text, device):
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    full_inputs = processor(
        text=[prompt_text + response_text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    full_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in full_inputs.items()
    }
    response_ids = full_inputs["input_ids"][0, prompt_len:].detach().cpu()
    if response_ids.numel() == 0:
        raise ValueError("response has no tokens after prompt")
    return full_inputs, prompt_len, response_ids, prompt_text


def find_rank(token_ids, selected_id):
    for idx, token_id in enumerate(token_ids):
        if int(token_id) == int(selected_id):
            return idx + 1
    return None


def topk_fields(log_probs_row, selected_id, k):
    import torch

    if k <= 0:
        return {}
    k = min(k, int(log_probs_row.shape[-1]))
    vals, ids = torch.topk(log_probs_row, k=k, dim=-1)
    ids_list = [int(value) for value in ids.detach().cpu().tolist()]
    vals_list = [float(value) for value in vals.detach().cpu().tolist()]
    return {
        "topk_token_ids": ids_list,
        "topk_logprobs": vals_list,
        "top1_token_id": ids_list[0] if ids_list else None,
        "topk_mass": float(torch.exp(vals).sum().detach().cpu().item()),
        "top1_top2_margin": vals_list[0] - vals_list[1] if len(vals_list) > 1 else None,
        "selected_token_rank_in_topk": find_rank(ids_list, selected_id),
    }


def forward_target_logits(model, inputs, prompt_len, max_model_len):
    import torch

    input_ids = inputs["input_ids"][0]
    if int(input_ids.shape[0]) > max_model_len:
        raise ValueError(f"input length {int(input_ids.shape[0])} exceeds max_model_len={max_model_len}")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    target_logits = logits[prompt_len - 1:-1]
    target_ids = input_ids[prompt_len:]
    if target_logits.shape[0] != target_ids.shape[0]:
        raise ValueError(
            f"target/logit length mismatch: logits={target_logits.shape[0]} target={target_ids.shape[0]}"
        )
    return target_logits, target_ids


def score_pair(
    *,
    student_model,
    student_processor,
    student_device,
    teacher_model,
    teacher_processor,
    teacher_device,
    student_image,
    teacher_image,
    prompt,
    response_text,
    max_model_len,
    kl_chunk_size,
    topk,
):
    import torch

    student_inputs, student_prompt_len, student_response_ids, student_prompt_text = tensorize_inputs(
        student_processor, student_image, prompt, response_text, student_device
    )
    teacher_inputs, teacher_prompt_len, teacher_response_ids, teacher_prompt_text = tensorize_inputs(
        teacher_processor, teacher_image, prompt, response_text, teacher_device
    )
    if student_response_ids.tolist() != teacher_response_ids.tolist():
        raise ValueError(
            "student/teacher response tokenization differs; use matching model/tokenizer pair for this probe"
        )

    student_target_logits, student_target_ids = forward_target_logits(
        student_model, student_inputs, student_prompt_len, max_model_len
    )
    teacher_target_logits, teacher_target_ids = forward_target_logits(
        teacher_model, teacher_inputs, teacher_prompt_len, max_model_len
    )
    if int(student_target_logits.shape[0]) != int(teacher_target_logits.shape[0]):
        raise ValueError(
            f"student/teacher target length mismatch: "
            f"{student_target_logits.shape[0]} vs {teacher_target_logits.shape[0]}"
        )
    if student_target_ids.detach().cpu().tolist() != teacher_target_ids.detach().cpu().tolist():
        raise ValueError("student/teacher target ids differ after tokenization")

    # Compute exact full-vocabulary divergences in response-token chunks.
    n_tokens = int(student_target_ids.shape[0])
    token_records = []
    topk_overlap_ratios = []
    topk_jaccards = []
    top1_matches = []
    selected_token_ids = [int(value) for value in student_response_ids.tolist()]

    for start in range(0, n_tokens, kl_chunk_size):
        end = min(start + kl_chunk_size, n_tokens)
        selected = student_target_ids[start:end].to(student_target_logits.device)
        s_log_probs = torch.log_softmax(student_target_logits[start:end].float(), dim=-1)
        t_log_probs = torch.log_softmax(
            teacher_target_logits[start:end].to(student_target_logits.device).float(),
            dim=-1,
        )
        s_probs = s_log_probs.exp()
        t_probs = t_log_probs.exp()
        log_m = torch.logaddexp(s_log_probs, t_log_probs) - math.log(2.0)

        rkl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)
        fkl = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
        jsd = 0.5 * (s_probs * (s_log_probs - log_m)).sum(dim=-1) + 0.5 * (
            t_probs * (t_log_probs - log_m)
        ).sum(dim=-1)
        s_entropy = -(s_probs * s_log_probs).sum(dim=-1)
        t_entropy = -(t_probs * t_log_probs).sum(dim=-1)
        s_selected = s_log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)
        t_selected = t_log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)

        for offset in range(end - start):
            pos = start + offset
            token_id = int(selected[offset].detach().cpu().item())
            token_record = {
                "step": pos,
                "response_position": int(student_prompt_len + pos),
                "token_id": token_id,
                "token_text": student_processor.decode([token_id], skip_special_tokens=False),
                "rkl_student_to_teacher": float(rkl[offset].detach().cpu().item()),
                "fkl_teacher_to_student": float(fkl[offset].detach().cpu().item()),
                "jsd": float(jsd[offset].detach().cpu().item()),
                "student_selected_logprob": float(s_selected[offset].detach().cpu().item()),
                "teacher_selected_logprob": float(t_selected[offset].detach().cpu().item()),
                "student_minus_teacher_selected_logprob": float(
                    (s_selected[offset] - t_selected[offset]).detach().cpu().item()
                ),
                "student_nll": float((-s_selected[offset]).detach().cpu().item()),
                "teacher_nll": float((-t_selected[offset]).detach().cpu().item()),
                "student_entropy": float(s_entropy[offset].detach().cpu().item()),
                "teacher_entropy": float(t_entropy[offset].detach().cpu().item()),
            }
            if topk > 0:
                s_topk = topk_fields(s_log_probs[offset], token_id, topk)
                t_topk = topk_fields(t_log_probs[offset], token_id, topk)
                for key, value in s_topk.items():
                    token_record[f"student_{key}"] = value
                for key, value in t_topk.items():
                    token_record[f"teacher_{key}"] = value
                s_ids = token_record.get("student_topk_token_ids") or []
                t_ids = token_record.get("teacher_topk_token_ids") or []
                if s_ids and t_ids:
                    s_set = set(s_ids)
                    t_set = set(t_ids)
                    overlap_count = len(s_set & t_set)
                    union_count = len(s_set | t_set)
                    overlap_ratio = overlap_count / min(len(s_set), len(t_set))
                    top1_match = token_record.get("student_top1_token_id") == token_record.get("teacher_top1_token_id")
                    token_record["topk_overlap_count"] = overlap_count
                    token_record["topk_overlap_ratio"] = overlap_ratio
                    token_record["topk_jaccard"] = overlap_count / union_count if union_count else None
                    token_record["top1_match"] = top1_match
                    token_record["student_top1_top2_margin"] = token_record.get("student_top1_top2_margin")
                    token_record["teacher_top1_top2_margin"] = token_record.get("teacher_top1_top2_margin")
                    topk_overlap_ratios.append(overlap_ratio)
                    topk_jaccards.append(token_record["topk_jaccard"])
                    top1_matches.append(1.0 if top1_match else 0.0)

            token_records.append(token_record)

    summary = {
        "num_tokens": len(token_records),
        "rkl_student_to_teacher_mean": safe_mean(
            record.get("rkl_student_to_teacher") for record in token_records
        ),
        "fkl_teacher_to_student_mean": safe_mean(
            record.get("fkl_teacher_to_student") for record in token_records
        ),
        "jsd_mean": safe_mean(record.get("jsd") for record in token_records),
        "student_nll_mean": safe_mean(record.get("student_nll") for record in token_records),
        "student_entropy_mean": safe_mean(record.get("student_entropy") for record in token_records),
        "teacher_entropy_mean": safe_mean(record.get("teacher_entropy") for record in token_records),
        "student_selected_logprob_mean": safe_mean(
            record.get("student_selected_logprob") for record in token_records
        ),
        "teacher_selected_logprob_mean": safe_mean(
            record.get("teacher_selected_logprob") for record in token_records
        ),
        "topk_overlap_ratio_mean": safe_mean(topk_overlap_ratios),
        "topk_jaccard_mean": safe_mean(topk_jaccards),
        "top1_match_frac": safe_mean(top1_matches),
    }
    return {
        "prompt_text": student_prompt_text,
        "teacher_prompt_text": teacher_prompt_text,
        "response_token_ids": selected_token_ids,
        "token_records": token_records,
        "summary": summary,
    }


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
    irregular = {
        "person": "people",
        "man": "men",
        "woman": "women",
        "child": "children",
        "mouse": "mice",
        "knife": "knives",
    }
    if last in irregular:
        variants.add(" ".join(words[:-1] + [irregular[last]]))
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


def build_token_char_spans(token_records):
    parts = []
    spans = []
    cursor = 0
    for token in token_records:
        text = str(token.get("token_text") or "")
        text = text.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\n")
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append((start, cursor))
    return "".join(parts), spans


def spans_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def token_indices_for_char_span(token_spans, start, end):
    return [
        idx
        for idx, (token_start, token_end) in enumerate(token_spans)
        if token_start < end and start < token_end
    ]


def bounded_context(text, start, end, window=80):
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].replace("\n", " ")
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet += "..."
    return snippet


def get_gt_objects(record):
    metadata = record.get("metadata") or {}
    extra_info = metadata.get("extra_info") if isinstance(metadata.get("extra_info"), dict) else {}
    for source in (record, metadata, extra_info):
        if not isinstance(source, dict):
            continue
        for key in ("gt_objects", "objects", "mscoco_objects"):
            value = source.get(key)
            if value:
                return set(value)
    return set()


def aggregate_tokens(tokens):
    out = {}
    for field in TOKEN_METRIC_FIELDS:
        values = [token.get(field) for token in tokens if bool_finite(token.get(field))]
        if not values:
            continue
        out[f"{field}_mean"] = safe_mean(values)
        out[f"{field}_max"] = max(values)
        out[f"{field}_sum"] = sum(values)
    top1_values = []
    for token in tokens:
        raw = token.get("top1_match")
        if raw is True:
            top1_values.append(1.0)
        elif raw is False:
            top1_values.append(0.0)
    if top1_values:
        out["top1_match_frac"] = safe_mean(top1_values)
    return out


def find_object_mentions(record, canonical_synonym_map):
    token_records = record.get("token_records", []) or []
    token_text, token_spans = build_token_char_spans(token_records)
    search_text = token_text or record.get("response_text", "")
    if not search_text:
        return []
    lowered = search_text.lower()
    gt_objects = get_gt_objects(record)

    candidates = []
    for canonical, terms in canonical_synonym_map.items():
        for term in terms:
            if term:
                candidates.append((canonical, term))
    candidates.sort(key=lambda item: (-len(item[1]), item[0], item[1]))

    selected_spans_by_object = defaultdict(list)
    mentions = []
    for canonical, term in candidates:
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
        for match in re.finditer(pattern, lowered):
            char_span = (match.start(), match.end())
            if any(spans_overlap(char_span, existing) for existing in selected_spans_by_object[canonical]):
                continue
            token_indices = token_indices_for_char_span(token_spans, match.start(), match.end())
            if not token_indices:
                continue
            tokens = [token_records[idx] for idx in token_indices if idx < len(token_records)]
            if not tokens:
                continue
            selected_spans_by_object[canonical].append(char_span)
            object_type = "unknown_object"
            if gt_objects:
                object_type = "correct_object" if canonical in gt_objects else "hallucinated_object"
            metrics = aggregate_tokens(tokens)
            mention = {
                "image_id": record.get("metadata", {}).get("image_id"),
                "caption_source": record.get("caption_source"),
                "canonical_object": canonical,
                "source_term": term,
                "mention_text": search_text[match.start():match.end()],
                "context": bounded_context(search_text, match.start(), match.end()),
                "object_type": object_type,
                "char_start": match.start(),
                "char_end": match.end(),
                "token_start": min(token_indices),
                "token_end": max(token_indices) + 1,
                "num_tokens": len(tokens),
                "token_text": "".join(str(token.get("token_text", "")) for token in tokens),
            }
            mention.update(metrics)
            mentions.append(mention)

    mentions.sort(key=lambda item: (item.get("char_start", 0), item.get("canonical_object", "")))
    return mentions


def add_non_object_token_roles(record, object_mentions):
    covered = set()
    for mention in object_mentions:
        for idx in range(int(mention.get("token_start", 0)), int(mention.get("token_end", 0))):
            covered.add(idx)
    gt_objects = get_gt_objects(record)
    by_index_type = {}
    for mention in object_mentions:
        object_type = mention.get("object_type")
        for idx in range(int(mention.get("token_start", 0)), int(mention.get("token_end", 0))):
            by_index_type[idx] = object_type
    for idx, token in enumerate(record.get("token_records", []) or []):
        if idx in by_index_type:
            token["token_role"] = by_index_type[idx]
        elif gt_objects:
            token["token_role"] = "non_object_token"
        else:
            token["token_role"] = "unknown_token"


def build_trace_record(args, record, caption, scored):
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
    out = {
        "probe_type": "same_image_rkl_signal",
        "pair_mode": args.pair_mode,
        "caption_source": args.caption_source_label or record.get("caption_source") or "model_caption",
        "student_model_path": args.student_model_path,
        "teacher_model_path": args.teacher_model_path,
        "degradation_mode": args.degradation_mode,
        "student_ratio": args.student_ratio,
        "teacher_ratio": args.teacher_ratio,
        "student_px": args.student_px,
        "teacher_px": args.teacher_px,
        "target_px": args.target_px,
        "metadata": metadata,
        "response_text": caption,
        "response_token_ids": scored["response_token_ids"],
        "summary": scored["summary"],
        "token_records": scored["token_records"],
    }
    return out


def existing_done_keys(path):
    if not os.path.exists(path):
        return set()
    keys = set()
    for record in load_jsonl(path):
        metadata = record.get("metadata") or {}
        image_id = metadata.get("image_id")
        caption_source = record.get("caption_source")
        keys.add((as_int(image_id), caption_source))
    return keys


def load_models_for_mode(args):
    student_model, student_processor, student_device = load_model_and_processor(
        args.student_model_path, args.torch_dtype
    )
    if args.pair_mode in {"duplicate_base", "dual_view_same_model"} and args.teacher_model_path == args.student_model_path:
        return student_model, student_processor, student_device, student_model, student_processor, student_device
    teacher_model, teacher_processor, teacher_device = load_model_and_processor(
        args.teacher_model_path, args.torch_dtype
    )
    return student_model, student_processor, student_device, teacher_model, teacher_processor, teacher_device


def run_scoring(args, output_jsonl):
    test_index = load_test_index(args.test_json)
    records = [
        normalize_generation_record(record, test_index, args.caption_field)
        for record in load_jsonl(args.eval_results)
    ]
    if args.overwrite and os.path.exists(output_jsonl):
        os.remove(output_jsonl)
    done_keys = existing_done_keys(output_jsonl)

    (
        student_model,
        student_processor,
        student_device,
        teacher_model,
        teacher_processor,
        teacher_device,
    ) = load_models_for_mode(args)

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    started = time.time()
    written = 0
    skipped = 0
    failures = 0
    with open(output_jsonl, "a", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            if args.max_samples and idx >= args.max_samples:
                break
            caption = record.get("generated_caption") or ""
            if not caption or str(caption).startswith("[ERROR]"):
                skipped += 1
                continue
            caption_source = args.caption_source_label or record.get("caption_source") or "model_caption"
            key = (as_int(record.get("image_id")), caption_source)
            if key in done_keys:
                skipped += 1
                continue
            image_path = resolve_image_path(record, test_index, args.image_root)
            if not image_path:
                print(f"WARNING: missing image path for image_id={record.get('image_id')}", file=sys.stderr)
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
                scored = score_pair(
                    student_model=student_model,
                    student_processor=student_processor,
                    student_device=student_device,
                    teacher_model=teacher_model,
                    teacher_processor=teacher_processor,
                    teacher_device=teacher_device,
                    student_image=student_image,
                    teacher_image=teacher_image,
                    prompt=args.prompt,
                    response_text=caption,
                    max_model_len=args.max_model_len,
                    kl_chunk_size=args.kl_chunk_size,
                    topk=args.topk,
                )
                out = build_trace_record(args, record, caption, scored)
            except Exception as exc:
                print(
                    f"WARNING: failed scoring image_id={record.get('image_id')} "
                    f"caption_source={caption_source}: {exc}",
                    file=sys.stderr,
                )
                failures += 1
                continue
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            done_keys.add(key)
            written += 1
            if written % 10 == 0:
                elapsed = time.time() - started
                print(f"Scored {written} records ({elapsed:.0f}s, skipped={skipped}, failures={failures})")
    print(f"Scoring complete: written={written} skipped={skipped} failures={failures}")
    print(f"Saved trace JSONL: {output_jsonl}")


def group_summary(rows, group_key, metric_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get(group_key, "unknown")].append(row)
    out = {}
    for group, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        out[group] = summarize_rows(group_rows, metric_fields)
    return out


def summarize_rows(rows, metric_fields):
    out = {
        "count": len(rows),
    }
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
    return out


def flatten_token_rows(records):
    rows = []
    for record in records:
        metadata = record.get("metadata") or {}
        image_id = metadata.get("image_id")
        for token in record.get("token_records", []) or []:
            row = dict(token)
            row["image_id"] = image_id
            row["caption_source"] = record.get("caption_source")
            rows.append(row)
    return rows


def high_divergence_buckets(mentions):
    labeled = [
        row for row in mentions
        if row.get("object_type") in {"correct_object", "hallucinated_object"}
    ]
    total_hallucinated = len([row for row in labeled if row.get("object_type") == "hallucinated_object"])
    total_correct = len([row for row in labeled if row.get("object_type") == "correct_object"])
    total_labeled = total_hallucinated + total_correct
    base_rate = safe_rate(total_hallucinated, total_labeled)
    out = {
        "base_hallucination_rate": base_rate,
        "total_labeled": total_labeled,
        "total_hallucinated": total_hallucinated,
        "total_correct": total_correct,
        "buckets": [],
    }
    for metric in HIGH_BUCKET_METRICS:
        metric_rows = [row for row in labeled if bool_finite(row.get(metric))]
        metric_rows.sort(key=lambda row: row.get(metric), reverse=True)
        for frac in HIGH_BUCKET_FRACTIONS:
            if not metric_rows:
                selected = []
            else:
                k = max(1, int(math.ceil(len(metric_rows) * frac)))
                selected = metric_rows[:k]
            hallucinated = len([row for row in selected if row.get("object_type") == "hallucinated_object"])
            correct = len([row for row in selected if row.get("object_type") == "correct_object"])
            selected_total = hallucinated + correct
            precision = safe_rate(hallucinated, selected_total)
            out["buckets"].append({
                "metric": metric,
                "top_fraction": frac,
                "selected": selected_total,
                "hallucinated": hallucinated,
                "correct": correct,
                "hallucination_precision": precision,
                "hallucination_recall": safe_rate(hallucinated, total_hallucinated),
                "correct_fpr": safe_rate(correct, total_correct),
                "precision_lift": precision / base_rate if base_rate > 0 else None,
                "threshold_min": selected[-1].get(metric) if selected else None,
            })
    return out


def clean_mention(row):
    keep = [
        "image_id",
        "caption_source",
        "canonical_object",
        "source_term",
        "mention_text",
        "context",
        "object_type",
        "char_start",
        "char_end",
        "token_start",
        "token_end",
        "num_tokens",
        "token_text",
    ] + SPAN_AGG_FIELDS
    return {key: row.get(key) for key in keep if key in row}


def analyze_trace(output_jsonl):
    _, inverse_synonym_dict = parse_official_synonyms()
    canonical_synonym_map = build_canonical_synonym_map(inverse_synonym_dict)
    records = load_jsonl(output_jsonl)
    object_mentions = []
    for record in records:
        mentions = find_object_mentions(record, canonical_synonym_map)
        add_non_object_token_roles(record, mentions)
        record["object_mentions"] = mentions
        object_mentions.extend(mentions)

    token_rows = flatten_token_rows(records)
    labeled_mentions = [
        row for row in object_mentions
        if row.get("object_type") in {"correct_object", "hallucinated_object"}
    ]
    correct_mentions = [row for row in labeled_mentions if row.get("object_type") == "correct_object"]
    hallucinated_mentions = [
        row for row in labeled_mentions if row.get("object_type") == "hallucinated_object"
    ]
    high_buckets = high_divergence_buckets(object_mentions)

    summary = {
        "trace_jsonl": output_jsonl,
        "num_records": len(records),
        "num_tokens": len(token_rows),
        "num_object_mentions": len(object_mentions),
        "num_labeled_object_mentions": len(labeled_mentions),
        "num_correct_object_mentions": len(correct_mentions),
        "num_hallucinated_object_mentions": len(hallucinated_mentions),
        "object_type_summary": group_summary(object_mentions, "object_type", SPAN_AGG_FIELDS),
        "token_role_summary": group_summary(token_rows, "token_role", TOKEN_METRIC_FIELDS),
        "high_divergence_buckets": high_buckets,
        "top_high_rkl_mentions": [
            clean_mention(row)
            for row in sorted(
                [row for row in labeled_mentions if bool_finite(row.get("rkl_student_to_teacher_mean"))],
                key=lambda row: row.get("rkl_student_to_teacher_mean"),
                reverse=True,
            )[:30]
        ],
        "top_high_jsd_mentions": [
            clean_mention(row)
            for row in sorted(
                [row for row in labeled_mentions if bool_finite(row.get("jsd_mean"))],
                key=lambda row: row.get("jsd_mean"),
                reverse=True,
            )[:30]
        ],
    }
    return summary


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
    object_summary = summary.get("object_type_summary", {})
    bucket_summary = summary.get("high_divergence_buckets", {})
    lines = [
        "# Same-Image RKL Signal Probe",
        "",
        "## Inputs",
        "",
        f"- trace_jsonl: `{summary.get('trace_jsonl')}`",
        f"- records: {summary.get('num_records')}",
        f"- tokens: {summary.get('num_tokens')}",
        f"- object_mentions: {summary.get('num_object_mentions')}",
        f"- labeled_object_mentions: {summary.get('num_labeled_object_mentions')}",
        f"- base_hallucination_rate: {fmt(bucket_summary.get('base_hallucination_rate'))}",
        "",
        "## Object Mention Summary",
        "",
    ]
    object_rows = []
    for group in ("correct_object", "hallucinated_object", "unknown_object"):
        stats = object_summary.get(group)
        if not stats:
            continue
        object_rows.append([
            group,
            stats.get("count", 0),
            fmt(stats.get("rkl_student_to_teacher_mean")),
            fmt(stats.get("rkl_student_to_teacher_mean_p75")),
            fmt(stats.get("fkl_teacher_to_student_mean")),
            fmt(stats.get("jsd_mean")),
            fmt(stats.get("student_nll_mean")),
            fmt(stats.get("student_entropy_mean")),
        ])
    lines.append(markdown_table(
        [
            "group",
            "count",
            "RKL mean",
            "RKL p75",
            "FKL mean",
            "JSD mean",
            "student NLL",
            "student entropy",
        ],
        object_rows,
    ))
    lines.extend(["", "## High-Divergence Buckets", ""])
    bucket_rows = []
    for bucket in bucket_summary.get("buckets", []):
        bucket_rows.append([
            bucket.get("metric"),
            fmt(bucket.get("top_fraction"), 2),
            bucket.get("selected"),
            bucket.get("hallucinated"),
            bucket.get("correct"),
            fmt(bucket.get("hallucination_precision")),
            fmt(bucket.get("hallucination_recall")),
            fmt(bucket.get("correct_fpr")),
            fmt(bucket.get("precision_lift")),
            fmt(bucket.get("threshold_min")),
        ])
    lines.append(markdown_table(
        [
            "metric",
            "top frac",
            "selected",
            "halluc",
            "correct",
            "precision",
            "recall",
            "correct FPR",
            "lift",
            "threshold",
        ],
        bucket_rows,
    ))
    lines.extend(["", "## Top High-RKL Mentions", ""])
    example_rows = []
    for mention in summary.get("top_high_rkl_mentions", [])[:20]:
        example_rows.append([
            mention.get("image_id"),
            mention.get("object_type"),
            mention.get("canonical_object"),
            mention.get("mention_text"),
            fmt(mention.get("rkl_student_to_teacher_mean")),
            fmt(mention.get("jsd_mean")),
            mention.get("context", "")[:160],
        ])
    lines.append(markdown_table(
        ["image", "type", "object", "mention", "RKL", "JSD", "context"],
        example_rows,
    ))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.print_server_paths:
        print("Default base original eval_results:")
        print(DEFAULT_SERVER_BASE_ORIGINAL_RESULTS)
        print("Suggested base lowres-0.75 eval_results directory:")
        print(DEFAULT_SERVER_BASE_LOWRES_DIR)
        return
    validate_args(args)
    output_jsonl, summary_json, summary_md = infer_output_paths(args)

    downloaded_paths = prepare_oss_checkpoints(args)
    try:
        if not args.analyze_only:
            run_scoring(args, output_jsonl)
        if not os.path.exists(output_jsonl):
            raise SystemExit(f"Trace JSONL does not exist: {output_jsonl}")

        summary = analyze_trace(output_jsonl)
        summary["config"] = {
            "pair_mode": args.pair_mode,
            "eval_results": args.eval_results,
            "student_model_path": args.student_model_path,
            "teacher_model_path": args.teacher_model_path,
            "student_oss_checkpoint": args.student_oss_checkpoint,
            "teacher_oss_checkpoint": args.teacher_oss_checkpoint,
            "degradation_mode": args.degradation_mode,
            "student_ratio": args.student_ratio,
            "teacher_ratio": args.teacher_ratio,
            "student_px": args.student_px,
            "teacher_px": args.teacher_px,
            "target_px": args.target_px,
            "topk": args.topk,
            "kl_chunk_size": args.kl_chunk_size,
        }
        os.makedirs(os.path.dirname(os.path.abspath(summary_json)), exist_ok=True)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        write_markdown(summary, summary_md)
        print(f"Saved summary JSON: {summary_json}")
        print(f"Saved summary Markdown: {summary_md}")
    finally:
        if args.cleanup_after:
            for path in sorted(set(downloaded_paths)):
                cleanup_local_checkpoint(path)


if __name__ == "__main__":
    main()
