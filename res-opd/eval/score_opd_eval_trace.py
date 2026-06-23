"""
Offline OPD scorer for eval/test captions.

This script does not generate captions. It reads fixed captions from
eval_chair.py outputs and runs two forced forwards with the same checkpoint:

  - student view: eval-time student image degradation
  - teacher view: low-resolution teacher image degradation

The output JSONL is compatible with analyze_opd_trace.py and can be joined with
case_analysis by image_id because it is computed on the eval/test split.
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
    parser = argparse.ArgumentParser(description="Forced-score eval captions with OPD student/teacher views")
    parser.add_argument("--model-path", required=True, help="Merged HF checkpoint path")
    parser.add_argument("--eval-results", required=True, help="eval_chair.py eval_results.jsonl")
    parser.add_argument("--output-jsonl", required=True, help="Where to write OPD eval trace JSONL")
    parser.add_argument("--test-json", help="Optional test.json for image_path/GT fallback")
    parser.add_argument("--case-analysis", help="Optional all_cases_sorted.json; used for baseline captions/groups")
    parser.add_argument("--image-root", default=DEFAULT_COCO_VAL_ROOT, help="Fallback COCO val image root")
    parser.add_argument("--prompt", default=PROMPT_TEXT, help="Prompt used for forced scoring")
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
    return 0


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
    return {int(sample["image_id"]): sample for sample in samples if sample.get("image_id") is not None}


def load_case_index(case_analysis):
    if not case_analysis:
        return {}
    with open(case_analysis, "r", encoding="utf-8") as f:
        cases = json.load(f)
    return {int(case["image_id"]): case for case in cases if case.get("image_id") is not None}


def resolve_image_path(record, test_index, image_root):
    image_path = record.get("image_path")
    if image_path and os.path.exists(image_path):
        return image_path
    image_id = record.get("image_id")
    try:
        image_id = int(image_id)
    except (TypeError, ValueError):
        image_id = None
    if image_id is not None:
        sample = test_index.get(image_id, {})
        image_path = sample.get("image_path")
        if image_path and os.path.exists(image_path):
            return image_path
    file_name = record.get("file_name") or (test_index.get(image_id, {}) if image_id is not None else {}).get("file_name")
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
        keys.add((image_id, caption_source))
    return keys


def iter_caption_jobs(eval_records, case_index, max_samples, score_baseline):
    count = 0
    for record in eval_records:
        if max_samples and count >= max_samples:
            break
        caption = record.get("generated_caption", "")
        if caption and not str(caption).startswith("[ERROR]"):
            yield record, "model_caption", caption
        image_id = record.get("image_id")
        try:
            image_id = int(image_id)
        except (TypeError, ValueError):
            image_id = None
        case = case_index.get(image_id, {}) if image_id is not None else {}
        baseline_caption = case.get("baseline_caption")
        if score_baseline and baseline_caption:
            yield record, "baseline_caption", baseline_caption
        count += 1


def main():
    args = parse_args()
    if args.overwrite and os.path.exists(args.output_jsonl):
        os.remove(args.output_jsonl)

    eval_records = load_jsonl(args.eval_results)
    test_index = load_test_index(args.test_json)
    case_index = load_case_index(args.case_analysis)
    done_keys = existing_trace_keys(args.output_jsonl)
    checkpoint_step = parse_checkpoint_step(args.model_path)
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
            key = (image_id, caption_source)
            if key in done_keys:
                skipped += 1
                continue
            image_path = resolve_image_path(record, test_index, args.image_root)
            if not image_path:
                print(f"WARNING: missing image path for image_id={image_id}", file=sys.stderr)
                failures += 1
                continue

            case = case_index.get(int(image_id), {}) if image_id is not None and str(image_id).isdigit() else {}
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
                "uid": f"eval:{image_id}:{caption_source}",
                "index": record.get("index"),
                "data_source": "coco_res_opd_eval",
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
                },
            }
            out = {
                "trace_scope": "eval",
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
