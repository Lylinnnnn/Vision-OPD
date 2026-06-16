#!/usr/bin/env python3
"""
POPE evaluation for Res-OPD checkpoints.

This script keeps the Res-OPD evaluation flow self-contained while reusing
Vision-OPD benchmark data preparation. It follows the common POPE evaluation
protocol: ask object-existence yes/no questions, extract a binary answer with
the official-style rule, and report accuracy, precision, recall, F1, and the
predicted yes ratio.
"""

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


BENCHMARK_JSON_MAP = {
    "pope": "POPE.json",
    "pope_adv": "POPE_adv.json",
    "pope_pop": "POPE_pop.json",
    "pope_random": "POPE_random.json",
}

VISION_PREPARE_SUFFIX = "\nAnswer the question using a single word or phrase."
DEFAULT_PROMPT_SUFFIX = "Please answer yes or no."


def parse_benchmarks(value: str) -> list[str]:
    if value.strip().lower() in {"all", "pope_all"}:
        return ["pope_adv", "pope_pop", "pope_random"]
    benchmarks = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in benchmarks if item not in BENCHMARK_JSON_MAP]
    if unknown:
        raise ValueError(
            f"Unsupported POPE benchmark(s): {unknown}. "
            f"Supported: {sorted(BENCHMARK_JSON_MAP)}"
        )
    return benchmarks


def make_sample_uid(item: dict, benchmark: str) -> str:
    for key in ("sample_uid", "uid", "index", "question_id", "id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return f"{benchmark}:{key}:{value}"
    image = (item.get("images") or [""])[0]
    return f"{benchmark}:fallback:{image}:{item.get('query', '')}"


def prepare_benchmark_if_needed(vision_opd_root: Path, benchmark: str) -> Path:
    eval_dir = vision_opd_root / "eval"
    benchmark_json = eval_dir / BENCHMARK_JSON_MAP[benchmark]
    if benchmark_json.exists():
        return benchmark_json

    prepare_script = eval_dir / "prepare_data.py"
    if not prepare_script.exists():
        raise FileNotFoundError(f"Vision-OPD prepare_data.py not found: {prepare_script}")

    print(f"Preparing {benchmark} data via Vision-OPD eval/prepare_data.py ...")
    subprocess.run(
        [
            sys.executable,
            str(prepare_script),
            "--benchmark",
            benchmark,
            "--data_dir",
            str(eval_dir),
        ],
        check=True,
    )
    if not benchmark_json.exists():
        raise FileNotFoundError(f"Expected benchmark json was not generated: {benchmark_json}")
    return benchmark_json


def load_jsonl(path: Path) -> dict[str, dict]:
    records = {}
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = record.get("sample_uid")
            if uid and not str(record.get("model_answer", "")).startswith("[ERROR]"):
                records[uid] = record
    return records


def degrade_image(image_path: str, student_px: int, target_px: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if student_px <= 0:
        return image
    small = image.resize((student_px, student_px), Image.LANCZOS)
    return small.resize((target_px, target_px), Image.LANCZOS)


def image_to_data_uri(image_path: str, student_px: int, target_px: int) -> str:
    image = degrade_image(image_path, student_px, target_px)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_query(item: dict, prompt_suffix: str, use_prepared_query: bool) -> str:
    query = str(item.get("query", "")).strip()
    if use_prepared_query:
        return query
    if query.endswith(VISION_PREPARE_SUFFIX.strip()):
        query = query[: -len(VISION_PREPARE_SUFFIX.strip())].strip()
    suffix = prompt_suffix.strip()
    if suffix and suffix.lower() not in query.lower():
        query = f"{query}\n{suffix}"
    return query


def official_pope_extract(text: str) -> str:
    """Official-style POPE yes/no extraction.

    The original POPE evaluator inspects the first sentence, removes commas,
    and treats answers containing "No", "no", or "not" as "no"; all other
    non-empty answers are counted as "yes".
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    first_sentence = text.strip().split(".")[0].replace(",", "")
    words = first_sentence.split()
    if "No" in words or "no" in words or "not" in words:
        return "no"
    return "yes"


def compute_metrics(records: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    for record in records:
        label = str(record.get("response", "")).strip().lower()
        pred = str(record.get("pred_answer", "")).strip().lower()
        if label == "yes" and pred == "yes":
            tp += 1
        elif label == "no" and pred == "yes":
            fp += 1
        elif label == "no" and pred == "no":
            tn += 1
        elif label == "yes" and pred == "no":
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    predicted_yes_ratio = (tp + fp) / total if total else 0.0
    label_yes_ratio = (tp + fn) / total if total else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_yes_ratio": predicted_yes_ratio,
        "label_yes_ratio": label_yes_ratio,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "num_samples": total,
    }


def run_single_benchmark(args, benchmark: str) -> dict:
    benchmark_json = prepare_benchmark_if_needed(args.vision_opd_root, benchmark)
    with open(benchmark_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    bench_dir = args.output_dir / "pope" / benchmark
    bench_dir.mkdir(parents=True, exist_ok=True)
    results_path = bench_dir / "eval_results.jsonl"
    metrics_path = bench_dir / "pope_metrics.json"

    completed = load_jsonl(results_path)
    todo = []
    for item in samples:
        uid = make_sample_uid(item, benchmark)
        if uid not in completed:
            item = dict(item)
            item["sample_uid"] = uid
            todo.append(item)

    print(
        f"[{benchmark}] samples={len(samples)} completed={len(completed)} "
        f"remaining={len(todo)}"
    )

    thread_local = threading.local()
    write_lock = threading.Lock()

    def get_client():
        client = getattr(thread_local, "client", None)
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=args.api_base, api_key=args.api_key, timeout=300)
            thread_local.client = client
        return client

    def run_one(item: dict) -> dict:
        image_path = (item.get("images") or [""])[0]
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        query = build_query(item, args.prompt_suffix, args.use_prepared_query)
        image_uri = image_to_data_uri(image_path, args.student_px, args.target_px)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": query},
                ],
            }
        ]

        answer = ""
        for attempt in range(1, args.max_retries + 1):
            try:
                response = get_client().chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    max_tokens=args.max_new_tokens,
                    temperature=0.0,
                )
                answer = (response.choices[0].message.content or "").strip()
                break
            except Exception as exc:
                if attempt == args.max_retries:
                    answer = f"[ERROR] {exc}"
                else:
                    time.sleep(float(attempt))

        record = dict(item)
        record["query_used"] = query
        record["model_answer"] = answer
        record["pred_answer"] = official_pope_extract(answer)
        record["correct"] = record["pred_answer"] == str(record.get("response", "")).strip().lower()
        record["student_px"] = args.student_px
        record["target_px"] = args.target_px
        return record

    if todo:
        done = len(completed)
        start = time.time()
        with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor, open(
            results_path, "a", encoding="utf-8"
        ) as f_out:
            future_to_item = {executor.submit(run_one, item): item for item in todo}
            for future in as_completed(future_to_item):
                try:
                    record = future.result()
                except Exception as exc:
                    item = future_to_item[future]
                    record = dict(item)
                    record["model_answer"] = f"[ERROR] {exc}"
                    record["pred_answer"] = ""
                    record["correct"] = False
                with write_lock:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    done += 1
                if done % 100 == 0:
                    elapsed = max(time.time() - start, 1e-6)
                    print(f"[{benchmark}] generated {done}/{len(samples)} ({done / elapsed:.2f} samples/s)")

    records_by_uid = load_jsonl(results_path)
    records = list(records_by_uid.values())
    metrics = compute_metrics(records)
    metrics.update(
        {
            "benchmark": benchmark,
            "student_px": args.student_px,
            "target_px": args.target_px,
            "extractor": "official_pope_first_sentence_no_not_rule",
            "benchmark_json": str(benchmark_json),
        }
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(
        f"[{benchmark}] Acc={metrics['accuracy'] * 100:.2f}% "
        f"Prec={metrics['precision'] * 100:.2f}% "
        f"Rec={metrics['recall'] * 100:.2f}% "
        f"F1={metrics['f1'] * 100:.2f}% "
        f"Yes={metrics['predicted_yes_ratio'] * 100:.2f}% "
        f"(n={metrics['num_samples']})"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="POPE evaluation for Res-OPD models")
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="Res-OPD")
    parser.add_argument("--benchmark", default="pope_adv,pope_pop,pope_random")
    parser.add_argument("--vision-opd-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path("./eval_results"))
    parser.add_argument("--student-px", type=int, default=0)
    parser.add_argument("--target-px", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--parallel-workers", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--prompt-suffix", default=DEFAULT_PROMPT_SUFFIX)
    parser.add_argument(
        "--use-prepared-query",
        action="store_true",
        help="Use the prepared Vision-OPD query verbatim instead of forcing a yes/no suffix.",
    )
    args = parser.parse_args()

    args.vision_opd_root = args.vision_opd_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = parse_benchmarks(args.benchmark)
    all_metrics = {}
    for benchmark in benchmarks:
        all_metrics[benchmark] = run_single_benchmark(args, benchmark)

    summary_path = args.output_dir / "pope" / "pope_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    macro_f1 = sum(m["f1"] for m in all_metrics.values()) / len(all_metrics) if all_metrics else 0.0
    macro_acc = sum(m["accuracy"] for m in all_metrics.values()) / len(all_metrics) if all_metrics else 0.0
    summary = {
        "benchmarks": all_metrics,
        "macro_accuracy": macro_acc,
        "macro_f1": macro_f1,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved POPE summary to: {summary_path}")


if __name__ == "__main__":
    main()
