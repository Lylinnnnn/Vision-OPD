#!/usr/bin/env python3
"""Build and score targeted POPE probes from base-vs-model caption diffs.

The targeted set is built from compare_eval_result_objects.py output:

  - removed_hallucinated: base mentioned a non-GT object, other removed it.
    Label is "no".
  - added_hallucinated: other added a non-GT object.
    Label is "no".
  - removed_correct: base mentioned a GT object, other removed it.
    Label is "yes".
  - added_correct: other added a GT object.
    Label is "yes".

This lets us test whether CHAIR improvements reflect actual yes/no visual
existence behavior, not just caption style.
"""

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import math
import os
import random
import re
import threading
import time

from PIL import Image


PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_OPD_ROOT = os.path.dirname(PROBE_DIR)
DEFAULT_TEST_JSON = os.path.join(RES_OPD_ROOT, "data", "test_1000.json")
DEFAULT_COMPARISON_JSON = os.path.join(
    PROBE_DIR,
    "results",
    "same_image_kl_probe_results",
    "6_eval_compare_base_vs_tr10_rkl.json",
)
DEFAULT_OUTPUT_DIR = os.path.join(PROBE_DIR, "results", "targeted_pope_base_vs_tr10_rkl")
DEFAULT_PROMPT_SUFFIX = "Please answer yes or no."
BUCKET_LABELS = {
    "removed_hallucinated": "no",
    "added_hallucinated": "no",
    "removed_correct": "yes",
    "added_correct": "yes",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Targeted POPE from caption object-diff cases.")
    parser.add_argument("--comparison-json", default=DEFAULT_COMPARISON_JSON)
    parser.add_argument("--test-json", default=DEFAULT_TEST_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-jsonl", default=None)
    parser.add_argument(
        "--buckets",
        default="removed_hallucinated,added_hallucinated,removed_correct,added_correct",
        help="Comma-separated targeted buckets to include.",
    )
    parser.add_argument("--max-per-bucket", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build-only", action="store_true")

    parser.add_argument("--api-base", default=None, help="OpenAI-compatible vLLM API base, e.g. http://localhost:8000/v1/")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--run-name", default=None, help="Name used for this answer file and summary.")
    parser.add_argument("--answers-jsonl", default=None)
    parser.add_argument("--parallel-workers", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--prompt-suffix", default=DEFAULT_PROMPT_SUFFIX)
    parser.add_argument("--save-logprobs", action="store_true")
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--degradation-mode", choices=["original", "square"], default="original")
    parser.add_argument("--student-ratio", type=float, default=1.0)
    parser.add_argument("--student-px", type=int, default=0)
    parser.add_argument("--target-px", type=int, default=448)

    parser.add_argument("--base-answers", default=None)
    parser.add_argument("--other-answers", default=None)
    parser.add_argument("--base-name", default="base")
    parser.add_argument("--other-name", default="other")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-md", default=None)
    return parser.parse_args()


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_object(obj):
    return str(obj or "").strip().lower().replace("_", " ")


def safe_part(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("_") or "unknown"


def load_test_index(test_json):
    if not os.path.exists(test_json):
        print(f"WARNING: test json not found; image_path/file_name fields may be empty: {test_json}")
        return {}
    samples = load_json(test_json)
    out = {}
    for idx, sample in enumerate(samples):
        image_id = sample.get("image_id", idx)
        out[int(image_id)] = sample
    return out


def object_question(obj, prompt_suffix):
    question = f"Is there a {obj} in the image?"
    suffix = str(prompt_suffix or "").strip()
    return f"{question}\n{suffix}" if suffix else question


def build_samples(args):
    comparison = load_json(args.comparison_json)
    cases = comparison.get("comparison", {}).get("cases", [])
    test_index = load_test_index(args.test_json)
    buckets = [bucket.strip() for bucket in args.buckets.split(",") if bucket.strip()]
    unknown = [bucket for bucket in buckets if bucket not in BUCKET_LABELS]
    if unknown:
        raise SystemExit(f"Unknown buckets: {unknown}. Supported: {sorted(BUCKET_LABELS)}")

    rows = []
    seen = set()
    for case in cases:
        image_id = int(case.get("image_id"))
        sample = test_index.get(image_id, {})
        image_path = sample.get("image_path") or case.get("image_path")
        for bucket in buckets:
            label = BUCKET_LABELS[bucket]
            for obj in case.get(bucket, []) or []:
                obj = normalize_object(obj)
                if not obj:
                    continue
                uid = f"{image_id}:{bucket}:{obj}"
                if uid in seen:
                    continue
                seen.add(uid)
                rows.append(
                    {
                        "sample_uid": uid,
                        "image_id": image_id,
                        "file_name": sample.get("file_name"),
                        "image_path": image_path,
                        "bucket": bucket,
                        "object": obj,
                        "response": label,
                        "query": object_question(obj, args.prompt_suffix),
                        "base_caption": case.get("base_caption", ""),
                        "other_caption": case.get("other_caption", ""),
                        "base_mentioned": case.get("base_mentioned", []),
                        "other_mentioned": case.get("other_mentioned", []),
                    }
                )

    if args.max_per_bucket > 0:
        rng = random.Random(args.seed)
        by_bucket = defaultdict(list)
        for row in rows:
            by_bucket[row["bucket"]].append(row)
        limited = []
        for bucket in buckets:
            bucket_rows = by_bucket.get(bucket, [])
            rng.shuffle(bucket_rows)
            limited.extend(bucket_rows[: args.max_per_bucket])
        rows = sorted(limited, key=lambda row: (row["bucket"], row["image_id"], row["object"]))
    else:
        rows = sorted(rows, key=lambda row: (row["bucket"], row["image_id"], row["object"]))
    return rows


def make_ratio_degraded_image(image_path, ratio):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if ratio <= 0:
        return Image.new("RGB", (width, height), color=(128, 128, 128))
    if ratio >= 1.0:
        return image
    small_size = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
    small = image.resize(small_size, Image.LANCZOS)
    return small.resize((width, height), Image.LANCZOS)


def make_square_degraded_image(image_path, student_px, target_px):
    image = Image.open(image_path).convert("RGB")
    if student_px <= 0:
        return image
    small = image.resize((student_px, student_px), Image.LANCZOS)
    return small.resize((target_px, target_px), Image.LANCZOS)


def image_to_data_uri(image_path, args):
    if args.degradation_mode == "original":
        image = make_ratio_degraded_image(image_path, args.student_ratio)
    else:
        image = make_square_degraded_image(image_path, args.student_px, args.target_px)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def official_pope_extract(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    first_sentence = text.strip().split(".")[0].replace(",", "")
    words = first_sentence.split()
    if "No" in words or "no" in words or "not" in words:
        return "no"
    return "yes"


def normalize_logprob_token(token):
    token = str(token or "")
    token = token.replace("Ġ", "").replace("▁", "").strip().lower()
    token = re.sub(r"^[^a-z]+|[^a-z]+$", "", token)
    return token


def extract_yes_no_logprobs(choice):
    logprob_yes = None
    logprob_no = None
    logprobs_obj = getattr(choice, "logprobs", None)
    content = getattr(logprobs_obj, "content", None)
    if not content:
        return logprob_yes, logprob_no
    first = content[0]
    top_logprobs = getattr(first, "top_logprobs", None) or []
    for item in top_logprobs:
        token = normalize_logprob_token(getattr(item, "token", ""))
        logprob = getattr(item, "logprob", None)
        if logprob is None:
            continue
        if token.startswith("yes"):
            logprob_yes = logprob if logprob_yes is None else max(logprob_yes, logprob)
        if token.startswith("no"):
            logprob_no = logprob if logprob_no is None else max(logprob_no, logprob)
    return logprob_yes, logprob_no


def run_api(args, samples, answers_path):
    from openai import OpenAI

    existing = {row.get("sample_uid"): row for row in load_jsonl(answers_path)}
    todo = [
        row for row in samples
        if row.get("sample_uid") not in existing
        or str(existing[row.get("sample_uid")].get("model_answer", "")).startswith("[ERROR]")
    ]
    print(f"Targeted POPE samples={len(samples)} completed={len(samples) - len(todo)} remaining={len(todo)}")
    if not todo:
        return list(existing.values())

    os.makedirs(os.path.dirname(os.path.abspath(answers_path)), exist_ok=True)
    thread_local = threading.local()
    write_lock = threading.Lock()

    def get_client():
        client = getattr(thread_local, "client", None)
        if client is None:
            client = OpenAI(base_url=args.api_base, api_key=args.api_key, timeout=300)
            thread_local.client = client
        return client

    def run_one(sample):
        image_path = sample.get("image_path")
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_uri = image_to_data_uri(image_path, args)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": sample["query"]},
                ],
            }
        ]
        request = {
            "model": args.model_name,
            "messages": messages,
            "max_tokens": args.max_new_tokens,
            "temperature": 0.0,
        }
        if args.save_logprobs:
            request["logprobs"] = True
            request["top_logprobs"] = args.top_logprobs

        answer = ""
        logprob_yes = None
        logprob_no = None
        for attempt in range(1, args.max_retries + 1):
            try:
                response = get_client().chat.completions.create(**request)
                choice = response.choices[0]
                answer = (choice.message.content or "").strip()
                if args.save_logprobs:
                    logprob_yes, logprob_no = extract_yes_no_logprobs(choice)
                break
            except Exception as exc:
                if attempt == args.max_retries:
                    answer = f"[ERROR] {exc}"
                else:
                    time.sleep(float(attempt))

        out = dict(sample)
        out["run_name"] = args.run_name or args.model_name
        out["model_name"] = args.model_name
        out["model_answer"] = answer
        out["pred_answer"] = official_pope_extract(answer)
        out["correct"] = out["pred_answer"] == str(out.get("response", "")).strip().lower()
        if args.save_logprobs:
            out["logprob_yes"] = logprob_yes
            out["logprob_no"] = logprob_no
            if logprob_yes is not None and logprob_no is not None:
                out["yes_minus_no_logprob"] = logprob_yes - logprob_no
                if str(out.get("response", "")).strip().lower() == "yes":
                    out["target_margin_logprob"] = logprob_yes - logprob_no
                elif str(out.get("response", "")).strip().lower() == "no":
                    out["target_margin_logprob"] = logprob_no - logprob_yes
        return out

    done = len(existing)
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor, open(
        answers_path, "a", encoding="utf-8"
    ) as f_out:
        future_to_sample = {executor.submit(run_one, sample): sample for sample in todo}
        for future in as_completed(future_to_sample):
            try:
                row = future.result()
            except Exception as exc:
                sample = future_to_sample[future]
                row = dict(sample)
                row["run_name"] = args.run_name or args.model_name
                row["model_name"] = args.model_name
                row["model_answer"] = f"[ERROR] {exc}"
                row["pred_answer"] = ""
                row["correct"] = False
            with write_lock:
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_out.flush()
            existing[row["sample_uid"]] = row
            done += 1
            if done % 50 == 0:
                elapsed = max(time.time() - start, 1e-6)
                print(f"Generated {done}/{len(samples)} ({done / elapsed:.2f} samples/s)")
    return list(existing.values())


def safe_div(num, den):
    return num / den if den else 0.0


def mean(values):
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def summarize_rows(rows):
    tp = fp = tn = fn = 0
    margins = []
    target_margins = []
    for row in rows:
        label = str(row.get("response", "")).strip().lower()
        pred = str(row.get("pred_answer", "")).strip().lower()
        if label == "yes" and pred == "yes":
            tp += 1
        elif label == "no" and pred == "yes":
            fp += 1
        elif label == "no" and pred == "no":
            tn += 1
        elif label == "yes" and pred == "no":
            fn += 1
        if isinstance(row.get("yes_minus_no_logprob"), (int, float)):
            margins.append(float(row["yes_minus_no_logprob"]))
        if isinstance(row.get("target_margin_logprob"), (int, float)):
            target_margins.append(float(row["target_margin_logprob"]))
    total = tp + fp + tn + fn
    return {
        "count": total,
        "accuracy": safe_div(tp + tn, total),
        "pred_yes_ratio": safe_div(tp + fp, total),
        "true_yes_ratio": safe_div(tp + fn, total),
        "false_yes_rate_on_no": safe_div(fp, fp + tn),
        "false_no_rate_on_yes": safe_div(fn, tp + fn),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "yes_minus_no_logprob_mean": mean(margins),
        "target_margin_logprob_mean": mean(target_margins),
    }


def summarize_answer_file(path):
    rows = list({row.get("sample_uid"): row for row in load_jsonl(path)}.values())
    by_bucket = defaultdict(list)
    for row in rows:
        by_bucket[row.get("bucket", "unknown")].append(row)
    return {
        "path": path,
        "overall": summarize_rows(rows),
        "by_bucket": {
            bucket: summarize_rows(bucket_rows)
            for bucket, bucket_rows in sorted(by_bucket.items())
        },
    }


def paired_summary(base_path, other_path):
    base_rows = {row.get("sample_uid"): row for row in load_jsonl(base_path)}
    other_rows = {row.get("sample_uid"): row for row in load_jsonl(other_path)}
    common = sorted(set(base_rows) & set(other_rows))
    by_bucket = defaultdict(list)
    for uid in common:
        base = base_rows[uid]
        other = other_rows[uid]
        by_bucket[base.get("bucket", "unknown")].append((base, other))

    def summarize_pairs(pairs):
        flips = Counter()
        margin_deltas = []
        target_margin_deltas = []
        for base, other in pairs:
            bp = base.get("pred_answer", "")
            op = other.get("pred_answer", "")
            flips[f"{bp}->{op}"] += 1
            bm = base.get("yes_minus_no_logprob")
            om = other.get("yes_minus_no_logprob")
            if isinstance(bm, (int, float)) and isinstance(om, (int, float)):
                margin_deltas.append(float(om) - float(bm))
            btm = base.get("target_margin_logprob")
            otm = other.get("target_margin_logprob")
            if isinstance(btm, (int, float)) and isinstance(otm, (int, float)):
                target_margin_deltas.append(float(otm) - float(btm))
        return {
            "count": len(pairs),
            "flips": dict(flips),
            "other_minus_base_yes_no_margin_mean": mean(margin_deltas),
            "other_minus_base_target_margin_mean": mean(target_margin_deltas),
        }

    return {
        "common": len(common),
        "by_bucket": {
            bucket: summarize_pairs(pairs)
            for bucket, pairs in sorted(by_bucket.items())
        },
    }


def fmt(value, digits=4):
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
        lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    return "\n".join(lines)


def write_summary_md(summary, path):
    lines = ["# Targeted POPE Summary", ""]
    for name in ("base", "other", "single"):
        if name not in summary:
            continue
        block = summary[name]
        lines.extend([f"## {name}", "", f"- path: `{block['path']}`", ""])
        rows = []
        for bucket, stats in block["by_bucket"].items():
            rows.append([
                bucket,
                stats["count"],
                fmt(stats["accuracy"]),
                fmt(stats["pred_yes_ratio"]),
                fmt(stats["false_yes_rate_on_no"]),
                fmt(stats["false_no_rate_on_yes"]),
                fmt(stats.get("yes_minus_no_logprob_mean")),
                fmt(stats.get("target_margin_logprob_mean")),
            ])
        lines.extend([
            markdown_table(
                [
                    "bucket",
                    "n",
                    "acc",
                    "pred yes",
                    "false yes on no",
                    "false no on yes",
                    "yes-no margin",
                    "target margin",
                ],
                rows,
            ),
            "",
        ])
    if "paired" in summary:
        lines.extend(["## paired", "", f"- common: {summary['paired']['common']}", ""])
        rows = []
        for bucket, stats in summary["paired"]["by_bucket"].items():
            flips = ", ".join(f"{k}:{v}" for k, v in sorted(stats["flips"].items()))
            rows.append([
                bucket,
                stats["count"],
                flips,
                fmt(stats.get("other_minus_base_yes_no_margin_mean")),
                fmt(stats.get("other_minus_base_target_margin_mean")),
            ])
        lines.extend([
            markdown_table(
                ["bucket", "n", "pred flips", "other-base yes-no margin", "other-base target margin"],
                rows,
            ),
            "",
        ])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    samples_path = args.samples_jsonl or os.path.join(args.output_dir, "targeted_pope_samples.jsonl")
    answers_path = args.answers_jsonl
    if answers_path is None and args.api_base:
        run_name = safe_part(args.run_name or args.model_name)
        answers_path = os.path.join(args.output_dir, f"{run_name}_answers.jsonl")
    summary_json = args.summary_json or os.path.join(args.output_dir, "targeted_pope_summary.json")
    summary_md = args.summary_md or os.path.join(args.output_dir, "targeted_pope_summary.md")

    samples = build_samples(args)
    write_jsonl(samples_path, samples)
    print(f"Saved targeted samples: {samples_path} ({len(samples)} rows)")

    if args.build_only:
        return

    if args.api_base:
        if not answers_path:
            raise SystemExit("--answers-jsonl could not be inferred")
        run_api(args, samples, answers_path)

    summary = {
        "samples_jsonl": samples_path,
        "num_samples": len(samples),
        "comparison_json": args.comparison_json,
    }
    if args.base_answers:
        summary["base"] = summarize_answer_file(args.base_answers)
    if args.other_answers:
        summary["other"] = summarize_answer_file(args.other_answers)
    if args.base_answers and args.other_answers:
        summary["paired"] = paired_summary(args.base_answers, args.other_answers)
    if answers_path and os.path.exists(answers_path) and not (args.base_answers or args.other_answers):
        summary["single"] = summarize_answer_file(answers_path)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_summary_md(summary, summary_md)
    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved summary Markdown: {summary_md}")


if __name__ == "__main__":
    main()
