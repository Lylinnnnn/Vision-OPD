#!/usr/bin/env python3
"""
POPE evaluation for Res-OPD checkpoints.

By default this script builds POPE-style yes/no object-existence probes from
the same Res-OPD COCO test JSON used by CHAIR. This keeps POPE and CHAIR on the
same image split. An opt-in Vision-OPD/HF POPE source is also available for
benchmark compatibility.
"""

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
import random
import re
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

COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def extract_final_response_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    final = text.strip()
    think_end = final.rfind("</think>")
    if think_end != -1:
        final = final[think_end + len("</think>"):].strip()
    match = re.search(r"<answer>(.*?)</answer>", final, flags=re.IGNORECASE | re.DOTALL)
    if match:
        final = match.group(1).strip()
    for marker in ("Final Answer:", "Final answer:", "Answer:", "answer:"):
        if marker in final:
            final = final.split(marker, 1)[1].strip()
            break
    return final


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


def normalize_object_name(name: str) -> str:
    return str(name).strip().lower().replace("_", " ")


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def pope_split_type(benchmark: str) -> str:
    if benchmark in {"pope", "pope_random"}:
        return "random"
    if benchmark == "pope_pop":
        return "popular"
    if benchmark == "pope_adv":
        return "adversarial"
    raise ValueError(f"Unsupported POPE benchmark: {benchmark}")


def object_question(obj: str) -> str:
    return f"Is there a {obj} in the image?"


def load_res_opd_test_samples(test_json: Path) -> list[dict]:
    with open(test_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError(f"Expected a list in {test_json}")
    return samples


def build_object_statistics(samples: list[dict]) -> tuple[Counter, dict[str, Counter]]:
    freq = Counter()
    cooc = defaultdict(Counter)
    vocab = set(COCO_CATEGORIES)

    for sample in samples:
        objects = {
            normalize_object_name(obj)
            for obj in sample.get("objects", [])
            if normalize_object_name(obj) in vocab
        }
        freq.update(objects)
        for obj in objects:
            for other in objects:
                if other != obj:
                    cooc[obj][other] += 1
    return freq, cooc


def choose_positive_objects(objects: set[str], count: int, seed: int) -> list[str]:
    candidates = sorted(objects)
    rnd = random.Random(seed)
    rnd.shuffle(candidates)
    return candidates[:count]


def choose_negative_objects(
    present_objects: set[str],
    count: int,
    split_type: str,
    freq: Counter,
    cooc: dict[str, Counter],
    seed: int,
) -> list[str]:
    absent = [obj for obj in COCO_CATEGORIES if obj not in present_objects]
    if split_type == "random":
        rnd = random.Random(seed)
        rnd.shuffle(absent)
        return absent[:count]

    if split_type == "popular":
        return sorted(absent, key=lambda obj: (-freq[obj], obj))[:count]

    if split_type == "adversarial":
        def score(obj: str) -> tuple[int, int, str]:
            cooc_score = sum(cooc[present][obj] for present in present_objects)
            return (-cooc_score, -freq[obj], obj)

        return sorted(absent, key=score)[:count]

    raise ValueError(f"Unsupported POPE split type: {split_type}")


def build_res_opd_pope_samples(args, benchmark: str) -> tuple[list[dict], str]:
    """Build POPE-style probes from res-opd/data/test.json.

    This mirrors the common POPE setup: balanced positive/negative object
    existence questions over COCO categories, with random/popular/adversarial
    negative sampling. Positive and negative counts are controlled by
    --questions-per-label.
    """
    source_samples = load_res_opd_test_samples(args.test_json)
    freq, cooc = build_object_statistics(source_samples)
    split_type = pope_split_type(benchmark)
    vocab = set(COCO_CATEGORIES)
    samples = []

    for idx, sample in enumerate(source_samples):
        image_path = sample.get("image_path")
        if not image_path:
            continue
        image_id = sample.get("image_id", idx)
        present = {
            normalize_object_name(obj)
            for obj in sample.get("objects", [])
            if normalize_object_name(obj) in vocab
        }
        if not present:
            continue

        seed_base = stable_seed(args.seed, benchmark, image_id, idx)
        positives = choose_positive_objects(present, args.questions_per_label, seed_base)
        negatives = choose_negative_objects(
            present,
            args.questions_per_label,
            split_type,
            freq,
            cooc,
            seed_base + 17,
        )

        for label, objects in (("yes", positives), ("no", negatives)):
            for obj_idx, obj in enumerate(objects):
                qid = f"{benchmark}:{image_id}:{label}:{obj_idx}:{obj}"
                samples.append(
                    {
                        "index": len(samples),
                        "id": qid,
                        "question_id": qid,
                        "image_id": image_id,
                        "file_name": sample.get("file_name", os.path.basename(image_path)),
                        "images": [image_path],
                        "query": object_question(obj),
                        "response": label,
                        "category": split_type,
                        "object": obj,
                        "present_objects": sorted(present),
                        "image_source": "res_opd_test_json",
                    }
                )

    return samples, str(args.test_json)


def load_vision_opd_pope_samples(args, benchmark: str) -> tuple[list[dict], str]:
    benchmark_json = prepare_benchmark_if_needed(args.vision_opd_root, benchmark)
    with open(benchmark_json, "r", encoding="utf-8") as f:
        return json.load(f), str(benchmark_json)


def load_pope_samples(args, benchmark: str) -> tuple[list[dict], str]:
    if args.pope_source == "res-opd-test":
        return build_res_opd_pope_samples(args, benchmark)
    if args.pope_source == "vision-opd":
        return load_vision_opd_pope_samples(args, benchmark)
    raise ValueError(f"Unsupported --pope-source: {args.pope_source}")


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


def degrade_image_by_ratio(image_path: str, ratio: float) -> Image.Image:
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


def image_to_data_uri(
    image_path: str,
    student_px: int,
    target_px: int,
    degradation_mode: str,
    student_ratio: float,
) -> str:
    if degradation_mode == "original":
        image = degrade_image_by_ratio(image_path, student_ratio)
    else:
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
    text = extract_final_response_text(text)
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
    samples, source_path = load_pope_samples(args, benchmark)
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
        image_uri = image_to_data_uri(
            image_path,
            args.student_px,
            args.target_px,
            args.degradation_mode,
            args.student_ratio,
        )
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
                request_kwargs = {}
                if args.enable_thinking is not None:
                    request_kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": args.enable_thinking == "True"}
                    }
                response = get_client().chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    max_tokens=args.max_new_tokens,
                    temperature=0.0,
                    **request_kwargs,
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
        final_answer = extract_final_response_text(answer)
        if final_answer != answer:
            record["final_answer_text"] = final_answer
        record["pred_answer"] = official_pope_extract(answer)
        record["correct"] = record["pred_answer"] == str(record.get("response", "")).strip().lower()
        record["student_px"] = args.student_px
        record["target_px"] = args.target_px
        record["degradation_mode"] = args.degradation_mode
        record["student_ratio"] = args.student_ratio
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
            "pope_source": args.pope_source,
            "student_px": args.student_px,
            "target_px": args.target_px,
            "degradation_mode": args.degradation_mode,
            "student_ratio": args.student_ratio,
            "questions_per_label": args.questions_per_label,
            "extractor": "official_pope_first_sentence_no_not_rule",
            "source_path": source_path,
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
    parser.add_argument(
        "--pope-source",
        choices=["res-opd-test", "vision-opd"],
        default="res-opd-test",
        help="Use Res-OPD COCO test.json by default; vision-opd uses the HF/Vision-OPD POPE data.",
    )
    parser.add_argument(
        "--test-json",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "test.json",
        help="Res-OPD COCO test JSON used when --pope-source=res-opd-test.",
    )
    parser.add_argument("--questions-per-label", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vision-opd-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path("./eval_results"))
    parser.add_argument("--student-px", type=int, default=0)
    parser.add_argument("--target-px", type=int, default=448)
    parser.add_argument("--degradation-mode", choices=["square", "original"], default="square")
    parser.add_argument("--student-ratio", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--parallel-workers", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--prompt-suffix", default=DEFAULT_PROMPT_SUFFIX)
    parser.add_argument("--enable-thinking", choices=["True", "False"], default=None)
    parser.add_argument(
        "--use-prepared-query",
        action="store_true",
        help="Use the prepared Vision-OPD query verbatim instead of forcing a yes/no suffix.",
    )
    args = parser.parse_args()

    args.vision_opd_root = args.vision_opd_root.resolve()
    args.test_json = args.test_json.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.questions_per_label <= 0:
        raise ValueError("--questions-per-label must be positive")
    if args.pope_source == "res-opd-test" and not args.test_json.exists():
        raise FileNotFoundError(f"Res-OPD test JSON not found: {args.test_json}")

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
