"""
Offline OPD scorer for eval/test or mini-eval captions.

This script does not generate captions. It reads fixed captions from
eval_chair.py outputs or verl validation generation dumps, then runs two
forced forwards with the same checkpoint:

  - student view: eval-time student image degradation
  - teacher view: low-resolution teacher image degradation

The output JSONL is compatible with analyze_opd_trace.py and can be joined with
case_analysis by image_id when the scored split matches the case-analysis split.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter

from PIL import Image


EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
from eval_chair import PROMPT_TEXT  # noqa: E402


DEFAULT_COCO_VAL_ROOT = "/home/liuyanlin.lyl/notebook/data/COCO/coco2017val/val2017"


def parse_args():
    parser = argparse.ArgumentParser(description="Forced-score eval/mini-eval captions with OPD student/teacher views")
    parser.add_argument("--model-path", required=True, help="Merged HF checkpoint path")
    parser.add_argument(
        "--eval-results",
        required=True,
        help="eval_chair.py eval_results.jsonl or verl validation_data_dir/<step>.jsonl",
    )
    parser.add_argument("--output-jsonl", required=True, help="Where to write OPD eval trace JSONL")
    parser.add_argument("--test-json", help="Optional test.json for image_path/GT fallback")
    parser.add_argument("--case-analysis", help="Optional all_cases_sorted.json; used for baseline captions/groups")
    parser.add_argument("--image-root", default=DEFAULT_COCO_VAL_ROOT, help="Fallback COCO val image root")
    parser.add_argument("--prompt", default=PROMPT_TEXT, help="Prompt used for forced scoring")
    parser.add_argument(
        "--trace-scope",
        default="eval",
        help="Scope label written to output records, e.g. eval, test, or mini_eval",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Global/checkpoint step for output records. Defaults to model path, then eval-results file name.",
    )
    parser.add_argument(
        "--caption-field",
        default="auto",
        help="Caption field to score. auto tries generated_caption, output, response_text, caption.",
    )
    parser.add_argument("--degradation-mode", choices=["square", "original"], default="square")
    parser.add_argument("--student-px", type=int, default=0)
    parser.add_argument("--teacher-px", type=int, default=0)
    parser.add_argument("--target-px", type=int, default=448)
    parser.add_argument("--student-ratio", type=float, default=1.0)
    parser.add_argument("--teacher-ratio", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--entropy", action="store_true", help="Compute full-vocabulary entropy")
    parser.add_argument("--score-baseline-caption", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="Debug cap by eval sample; 0 = all")
    parser.add_argument("--max-model-len", type=int, default=9728)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


def bool_finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def safe_mean(values):
    values = [float(v) for v in values if bool_finite(v)]
    return sum(values) / len(values) if values else None


def parse_checkpoint_step(path):
    for part in reversed(path.split(os.sep)):
        match = re.match(r"global_step_(\d+)$", part)
        if match:
            return int(match.group(1))
    basename = os.path.basename(path)
    match = re.match(r"(\d+)(?:\.[^.]+)?$", basename)
    if match:
        return int(match.group(1))
    return None


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_metadata(record):
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def get_extra_info(record):
    metadata = get_metadata(record)
    for value in (record.get("extra_info"), metadata.get("extra_info")):
        if isinstance(value, dict):
            return value
    return {}


def first_nonempty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        if isinstance(value, (list, tuple, dict, set)) and not value:
            continue
        return value
    return None


def record_value(record, *names):
    metadata = get_metadata(record)
    extra_info = get_extra_info(record)
    for name in names:
        value = first_nonempty(record.get(name), metadata.get(name), extra_info.get(name))
        if value is not None:
            return value
    return None


def infer_caption(record, caption_field):
    if caption_field != "auto":
        return record.get(caption_field)
    for field in ("generated_caption", "output", "response_text", "caption"):
        value = record.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def normalize_generation_record(record, test_index, caption_field):
    metadata = get_metadata(record)
    extra_info = get_extra_info(record)
    image_id = as_int(record_value(record, "image_id"))
    indexed = test_index.get(image_id, {}) if image_id is not None else {}
    file_name = first_nonempty(
        record_value(record, "file_name"),
        indexed.get("file_name"),
    )
    image_path = first_nonempty(
        record_value(record, "image_path"),
        indexed.get("image_path"),
    )
    gt_objects = first_nonempty(
        record_value(record, "gt_objects", "objects", "present_objects", "coco_objects", "categories"),
        indexed.get("gt_objects"),
        indexed.get("objects"),
    ) or []
    gt_captions = first_nonempty(
        record_value(record, "gt_captions", "captions"),
        indexed.get("gt_captions"),
        indexed.get("captions"),
    ) or []
    data_source = first_nonempty(
        record_value(record, "data_source"),
        indexed.get("data_source"),
        "coco_res_opd_eval",
    )
    generation_step = first_nonempty(record.get("global_step"), record.get("step"), metadata.get("global_step"))
    generation_step = as_int(generation_step)
    uid = first_nonempty(record_value(record, "uid"), f"{data_source}:{image_id}:{generation_step}")
    caption_source = first_nonempty(record.get("caption_source"), metadata.get("caption_source"), "model_caption")

    normalized = dict(record)
    normalized.update(
        {
            "uid": uid,
            "data_source": data_source,
            "image_id": image_id,
            "file_name": file_name,
            "image_path": image_path,
            "gt_objects": gt_objects,
            "gt_captions": gt_captions,
            "generated_caption": infer_caption(record, caption_field),
            "caption_source": caption_source,
            "generation_step": generation_step,
            "metadata": metadata,
            "extra_info": extra_info,
        }
    )
    return normalized


def make_square_degraded_image(image_path, px, target_px):
    image = Image.open(image_path).convert("RGB")
    if px <= 0:
        return Image.new("RGB", (target_px, target_px), color=(128, 128, 128))
    small = image.resize((px, px), Image.LANCZOS)
    return small.resize((target_px, target_px), Image.LANCZOS)


def make_ratio_degraded_image(image_path, ratio):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if ratio <= 0:
        return Image.new("RGB", (width, height), color=(128, 128, 128))
    if ratio >= 1.0:
        return image
    small_size = (
        max(1, int(round(width * ratio))),
        max(1, int(round(height * ratio))),
    )
    small = image.resize(small_size, Image.LANCZOS)
    return small.resize((width, height), Image.LANCZOS)


def load_view_image(image_path, degradation_mode, px, target_px, ratio, *, blank_when_px_zero=False):
    if degradation_mode == "original":
        return make_ratio_degraded_image(image_path, ratio)
    if px <= 0:
        if blank_when_px_zero:
            return Image.new("RGB", (target_px, target_px), color=(128, 128, 128))
        return Image.open(image_path).convert("RGB")
    return make_square_degraded_image(image_path, px, target_px)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"WARNING: skip malformed JSON {path}:{line_no}: {exc}", file=sys.stderr)
    return records


def load_test_index(test_json):
    if not test_json:
        return {}
    with open(test_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if isinstance(samples, dict):
        samples = samples.get("samples") or samples.get("data") or list(samples.values())
    index = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        image_id = as_int(sample.get("image_id"))
        if image_id is not None:
            index[image_id] = sample
    return index


def load_case_index(case_analysis):
    if not case_analysis:
        return {}
    with open(case_analysis, "r", encoding="utf-8") as f:
        cases = json.load(f)
    return {int(case["image_id"]): case for case in cases if case.get("image_id") is not None}


def resolve_image_path(record, test_index, image_root):
    image_path = record_value(record, "image_path")
    if image_path and os.path.exists(image_path):
        return image_path
    image_id = as_int(record.get("image_id"))
    if image_id is not None:
        sample = test_index.get(image_id, {})
        image_path = sample.get("image_path")
        if image_path and os.path.exists(image_path):
            return image_path
    file_name = record_value(record, "file_name") or (
        test_index.get(image_id, {}) if image_id is not None else {}
    ).get("file_name")
    if file_name:
        candidate = os.path.join(image_root, os.path.basename(str(file_name)))
        if os.path.exists(candidate):
            return candidate
    return None


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
    print(f"Loading model for offline OPD trace: {model_path}")
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    device = next(model.parameters()).device
    print(f"Model loaded on first device: {device}")
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
    return full_inputs, prompt_len, prompt_text


def find_rank(token_ids, selected_id):
    for idx, token_id in enumerate(token_ids):
        if int(token_id) == int(selected_id):
            return idx + 1
    return None


def score_view(model, processor, device, image, prompt, response_text, topk, compute_entropy, max_model_len):
    import torch

    inputs, prompt_len, prompt_text = tensorize_inputs(processor, image, prompt, response_text, device)
    input_ids = inputs["input_ids"][0]
    if int(input_ids.shape[0]) > max_model_len:
        raise ValueError(f"input length {int(input_ids.shape[0])} exceeds max_model_len={max_model_len}")
    response_ids = input_ids[prompt_len:].detach().cpu()
    if response_ids.numel() == 0:
        raise ValueError("response has no tokens after chat-template prompt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    target_logits = logits[prompt_len - 1:-1].float()
    target_ids = input_ids[prompt_len:]
    if target_logits.shape[0] != target_ids.shape[0]:
        raise ValueError(
            f"target/logit length mismatch: logits={target_logits.shape[0]} target={target_ids.shape[0]}"
        )

    records = []
    chunk_size = 64
    for start in range(0, int(target_ids.shape[0]), chunk_size):
        end = min(start + chunk_size, int(target_ids.shape[0]))
        log_probs = torch.log_softmax(target_logits[start:end], dim=-1)
        selected = target_ids[start:end]
        selected_logprobs = log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)
        k = min(max(topk, 0), log_probs.shape[-1])
        if k > 0:
            top_vals, top_ids = torch.topk(log_probs, k=k, dim=-1)
        else:
            top_vals, top_ids = None, None
        if compute_entropy:
            entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
        else:
            entropies = None

        for offset in range(end - start):
            pos = start + offset
            token_id = int(selected[offset].detach().cpu().item())
            token_record = {
                "step": pos,
                "response_position": prompt_len + pos,
                "token_id": token_id,
                "token_text": processor.decode([token_id], skip_special_tokens=False),
                "selected_logprob": float(selected_logprobs[offset].detach().cpu().item()),
            }
            if entropies is not None:
                token_record["entropy"] = float(entropies[offset].detach().cpu().item())
            if top_vals is not None and top_ids is not None:
                ids = [int(value) for value in top_ids[offset].detach().cpu().tolist()]
                vals = [float(value) for value in top_vals[offset].detach().cpu().tolist()]
                token_record["topk_token_ids"] = ids
                token_record["topk_logprobs"] = vals
                token_record["top1_token_id"] = ids[0] if ids else None
                token_record["topk_mass"] = float(torch.exp(top_vals[offset]).sum().detach().cpu().item())
                token_record["top1_top2_margin"] = vals[0] - vals[1] if len(vals) > 1 else None
                token_record["selected_token_rank_in_topk"] = find_rank(ids, token_id)
            records.append(token_record)

    return {
        "prompt_text": prompt_text,
        "response_token_ids": [int(value) for value in response_ids.tolist()],
        "token_records": records,
    }


def summarize_side(records, prefix):
    selected = [record.get(f"{prefix}_selected_logprob") for record in records]
    entropies = [record.get(f"{prefix}_entropy") for record in records]
    masses = [record.get(f"{prefix}_topk_mass") for record in records]
    margins = [record.get(f"{prefix}_top1_top2_margin") for record in records]
    return {
        f"{prefix}_selected_logprob_mean": safe_mean(selected),
        f"{prefix}_entropy_mean": safe_mean(entropies),
        f"{prefix}_topk_mass_mean": safe_mean(masses),
        f"{prefix}_top1_top2_margin_mean": safe_mean(margins),
    }


def combine_student_teacher(student, teacher):
    student_ids = student["response_token_ids"]
    teacher_ids = teacher["response_token_ids"]
    n = min(len(student_ids), len(teacher_ids))
    alignment_warning = None
    if student_ids != teacher_ids:
        alignment_warning = (
            f"student/teacher response token ids differ; truncating to {n} tokens "
            f"(student={len(student_ids)}, teacher={len(teacher_ids)})"
        )

    token_records = []
    overlap_ratios = []
    topk_jaccards = []
    top1_matches = []
    for idx in range(n):
        s = student["token_records"][idx]
        t = teacher["token_records"][idx]
        token_record = {
            "step": idx,
            "response_position": int(s.get("response_position", idx)),
            "token_id": int(s["token_id"]),
            "token_text": s.get("token_text", ""),
            "student_selected_logprob": s.get("selected_logprob"),
            "teacher_selected_logprob": t.get("selected_logprob"),
        }
        if "entropy" in s:
            token_record["student_entropy"] = s["entropy"]
        if "entropy" in t:
            token_record["teacher_entropy"] = t["entropy"]

        for src, prefix in ((s, "student"), (t, "teacher")):
            if "topk_token_ids" not in src:
                continue
            token_record[f"{prefix}_topk_token_ids"] = src.get("topk_token_ids", [])
            token_record[f"{prefix}_topk_logprobs"] = src.get("topk_logprobs", [])
            token_record[f"{prefix}_top1_token_id"] = src.get("top1_token_id")
            token_record[f"{prefix}_topk_mass"] = src.get("topk_mass")
            token_record[f"{prefix}_top1_top2_margin"] = src.get("top1_top2_margin")
            token_record[f"selected_token_rank_in_{prefix}_topk"] = src.get("selected_token_rank_in_topk")

        student_ids_topk = token_record.get("student_topk_token_ids") or []
        teacher_ids_topk = token_record.get("teacher_topk_token_ids") or []
        if student_ids_topk and teacher_ids_topk:
            student_set = set(student_ids_topk)
            teacher_set = set(teacher_ids_topk)
            overlap_count = len(student_set & teacher_set)
            union_count = len(student_set | teacher_set)
            overlap_ratio = overlap_count / min(len(student_set), len(teacher_set))
            top1_match = token_record.get("student_top1_token_id") == token_record.get("teacher_top1_token_id")
            token_record["topk_overlap_count"] = overlap_count
            token_record["topk_overlap_ratio"] = overlap_ratio
            token_record["topk_jaccard"] = overlap_count / union_count if union_count else None
            token_record["top1_match"] = top1_match
            overlap_ratios.append(overlap_ratio)
            topk_jaccards.append(token_record["topk_jaccard"])
            top1_matches.append(1.0 if top1_match else 0.0)

        token_records.append(token_record)

    summary = {
        "num_tokens": len(token_records),
        "topk_overlap_ratio_mean": safe_mean(overlap_ratios),
        "topk_jaccard_mean": safe_mean(topk_jaccards),
        "top1_match_frac": safe_mean(top1_matches),
    }
    summary.update(summarize_side(token_records, "student"))
    summary.update(summarize_side(token_records, "teacher"))
    return {
        "response_token_ids": student_ids[:n],
        "token_records": token_records,
        "summary": summary,
        "alignment_warning": alignment_warning,
    }


def existing_trace_keys(path):
    keys = set()
    if not os.path.exists(path):
        return keys
    for record in load_jsonl(path):
        image_id = record.get("metadata", {}).get("image_id") or record.get("image_id")
        caption_source = record.get("caption_source") or record.get("metadata", {}).get("caption_source")
        step = record.get("checkpoint_step") or record.get("global_step")
        keys.add((as_int(step), as_int(image_id), caption_source))
    return keys


def iter_caption_jobs(eval_records, case_index, max_samples, score_baseline):
    count = 0
    for record in eval_records:
        if max_samples and count >= max_samples:
            break
        caption = record.get("generated_caption", "")
        if caption and not str(caption).startswith("[ERROR]"):
            yield record, record.get("caption_source") or "model_caption", caption
        image_id = record.get("image_id")
        image_id = as_int(image_id)
        case = case_index.get(image_id, {}) if image_id is not None else {}
        baseline_caption = case.get("baseline_caption")
        if score_baseline and baseline_caption:
            yield record, "baseline_caption", baseline_caption
        count += 1


def main():
    args = parse_args()
    if args.overwrite and os.path.exists(args.output_jsonl):
        os.remove(args.output_jsonl)

    test_index = load_test_index(args.test_json)
    eval_records = [
        normalize_generation_record(record, test_index, args.caption_field)
        for record in load_jsonl(args.eval_results)
    ]
    case_index = load_case_index(args.case_analysis)
    done_keys = existing_trace_keys(args.output_jsonl)
    checkpoint_step_default = (
        args.checkpoint_step
        if args.checkpoint_step is not None
        else parse_checkpoint_step(args.model_path) or parse_checkpoint_step(args.eval_results) or 0
    )
    model, processor, device = load_model_and_processor(args.model_path, args.torch_dtype)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    started = time.time()
    written = 0
    skipped = 0
    failures = 0
    source_counter = Counter()
    with open(args.output_jsonl, "a", encoding="utf-8") as f:
        for record, caption_source, caption in iter_caption_jobs(
            eval_records,
            case_index,
            args.max_samples,
            args.score_baseline_caption,
        ):
            image_id = record.get("image_id")
            record_step = record.get("generation_step")
            checkpoint_step = args.checkpoint_step if args.checkpoint_step is not None else checkpoint_step_default
            if checkpoint_step == 0 and record_step is not None:
                checkpoint_step = record_step
            key = (as_int(checkpoint_step), as_int(image_id), caption_source)
            if key in done_keys:
                skipped += 1
                continue
            image_path = resolve_image_path(record, test_index, args.image_root)
            if not image_path:
                print(f"WARNING: missing image path for image_id={image_id}", file=sys.stderr)
                failures += 1
                continue

            image_id_int = as_int(image_id)
            case = case_index.get(image_id_int, {}) if image_id_int is not None else {}
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
                    model,
                    processor,
                    device,
                    student_image,
                    args.prompt,
                    caption,
                    args.topk,
                    args.entropy,
                    args.max_model_len,
                )
                teacher = score_view(
                    model,
                    processor,
                    device,
                    teacher_image,
                    args.prompt,
                    caption,
                    args.topk,
                    args.entropy,
                    args.max_model_len,
                )
                combined = combine_student_teacher(student, teacher)
            except Exception as exc:
                print(f"WARNING: failed scoring image_id={image_id} source={caption_source}: {exc}", file=sys.stderr)
                failures += 1
                continue

            metadata = {
                "uid": record.get("uid") or f"{args.trace_scope}:{image_id}:{caption_source}:step{checkpoint_step}",
                "index": record.get("index"),
                "data_source": record.get("data_source") or "coco_res_opd_eval",
                "image_id": image_id,
                "file_name": record.get("file_name"),
                "caption_source": caption_source,
                "case_group": case.get("distill_recommendation") or case.get("behavior_pattern"),
                "behavior_pattern": case.get("behavior_pattern"),
                "distill_recommendation": case.get("distill_recommendation"),
                "extra_info": {
                    "image_id": image_id,
                    "file_name": record.get("file_name"),
                    "image_path": image_path,
                    "objects": record.get("gt_objects", []) or case.get("gt_objects", []),
                    "captions": record.get("gt_captions", []),
                    "caption_source": caption_source,
                    "generation_step": record.get("generation_step"),
                },
            }
            out = {
                "trace_scope": args.trace_scope,
                "global_step": checkpoint_step,
                "checkpoint_step": checkpoint_step,
                "rank": 0,
                "sample_index_in_rank_batch": 0,
                "caption_source": caption_source,
                "model_path": args.model_path,
                "degradation_mode": args.degradation_mode,
                "student_px": args.student_px,
                "teacher_px": args.teacher_px,
                "target_px": args.target_px,
                "student_ratio": args.student_ratio,
                "teacher_ratio": args.teacher_ratio,
                "metadata": metadata,
                "response_token_ids": combined["response_token_ids"],
                "response_text": caption,
                "summary": combined["summary"],
                "token_records": combined["token_records"],
            }
            if combined.get("alignment_warning"):
                out["alignment_warning"] = combined["alignment_warning"]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            done_keys.add(key)
            written += 1
            source_counter[caption_source] += 1
            if written % 10 == 0:
                elapsed = time.time() - started
                print(f"Scored {written} traces ({elapsed:.0f}s, failures={failures}, skipped={skipped})")

    print(
        "OPD eval trace complete: "
        f"written={written} skipped={skipped} failures={failures} sources={dict(source_counter)}"
    )
    print(f"Saved trace to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
