#!/usr/bin/env python3
"""Classic MME yes/no evaluation for an OpenAI-compatible VLM endpoint."""

import argparse
import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import mimetypes
import os
from pathlib import Path
import threading
import time

from PIL import Image


PERCEPTION_TASKS = {
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "ocr",
}
COGNITION_TASKS = {"commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"}


def load_json_or_jsonl(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "records", "samples", "questions"):
            if isinstance(data.get(key), list):
                return data[key]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list, jsonl, or dict with data/records/samples/questions in {path}")
    return data


def load_official_mme_root(root: Path, categories: list[str]) -> list[dict]:
    rows = []
    for category in categories:
        category_dir = root / category
        if not category_dir.exists():
            raise FileNotFoundError(f"MME category directory not found: {category_dir}")
        txt_files = sorted(category_dir.rglob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No MME txt files found under: {category_dir}")
        for txt_path in txt_files:
            with open(txt_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        parts = line.split(",", 2)
                    if len(parts) < 3:
                        raise ValueError(f"Invalid MME line in {txt_path}:{line_no + 1}: {line}")
                    image_name, question, answer = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    image_path = None
                    candidates = [
                        category_dir / "images" / image_name,
                        category_dir / "image" / image_name,
                        category_dir / image_name,
                        root / "images" / category / image_name,
                    ]
                    for candidate in candidates:
                        if candidate.exists():
                            image_path = candidate
                            break
                    if image_path is None:
                        matches = list(category_dir.rglob(image_name))
                        if matches:
                            image_path = matches[0]
                    rows.append(
                        {
                            "image": str(image_path or candidates[0]),
                            "question": question,
                            "answer": answer,
                            "category": category,
                            "question_id": f"{category}:{txt_path.name}:{line_no}",
                        }
                    )
    return rows


def pick_first(item: dict, keys: tuple[str, ...], default=""):
    for key in keys:
        value = item.get(key)
        if value is not None and value != "":
            return value
    return default


def normalize_sample(item: dict, idx: int) -> dict:
    image = pick_first(item, ("image", "image_path", "img_path", "file_path", "images"), "")
    if isinstance(image, list):
        image = image[0] if image else ""
    question = pick_first(item, ("question", "query", "text", "prompt"), "")
    answer = str(pick_first(item, ("answer", "response", "label", "gt"), "")).strip().lower()
    category = str(pick_first(item, ("category", "task", "type"), "unknown")).strip()
    image_id = pick_first(item, ("image_id", "image", "image_path", "file_path", "id"), image)
    question_id = pick_first(item, ("question_id", "qid", "id", "index"), idx)
    if answer not in {"yes", "no"}:
        answer = extract_yes_no(answer)
    if not image or not question or answer not in {"yes", "no"}:
        raise ValueError(f"Invalid MME sample at index {idx}: need image, question, yes/no answer")
    return {
        "index": idx,
        "question_id": question_id,
        "image_id": str(image_id),
        "images": [str(image)],
        "query": str(question).strip(),
        "response": answer,
        "category": category,
    }


def sample_uid(item: dict) -> str:
    return f"{item.get('category')}:{item.get('image_id')}:{item.get('question_id')}:{item.get('index')}"


def load_existing(path: Path) -> dict[str, dict]:
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
            answer = str(record.get("model_answer", ""))
            if uid and answer and not answer.startswith("[ERROR]"):
                records[uid] = record
    return records


def load_eval_image(path: str, degradation_mode: str, student_px: int, target_px: int, student_ratio: float):
    image = Image.open(path).convert("RGB")
    if degradation_mode == "original":
        width, height = image.size
        if student_ratio <= 0:
            return Image.new("RGB", (width, height), color=(128, 128, 128))
        if student_ratio >= 1.0:
            return image
        small_size = (
            max(1, int(round(width * student_ratio))),
            max(1, int(round(height * student_ratio))),
        )
        return image.resize(small_size, Image.LANCZOS).resize((width, height), Image.LANCZOS)
    if student_px > 0:
        return image.resize((student_px, student_px), Image.LANCZOS).resize((target_px, target_px), Image.LANCZOS)
    return image


def image_to_data_uri(
    path: str,
    degradation_mode: str = "square",
    student_px: int = 0,
    target_px: int = 448,
    student_ratio: float = 1.0,
) -> str:
    p = Path(path)
    if degradation_mode == "original" or student_px > 0:
        image = load_eval_image(str(p), degradation_mode, student_px, target_px, student_ratio)
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        payload = buf.getvalue()
        mime = "image/jpeg"
    else:
        with open(p, "rb") as f:
            payload = f.read()
        mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(payload).decode('utf-8')}"


def extract_yes_no(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.strip().lower()
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1].strip()
    if "<answer>" in t and "</answer>" in t:
        t = t.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if t.startswith("answer"):
        t = t.split("answer", 1)[1].lstrip(":").strip()
    first_sentence = t.split(".", 1)[0].replace(",", "")
    words = first_sentence.split()
    if "no" in words or "not" in words or first_sentence.startswith("no"):
        return "no"
    if "yes" in words or first_sentence.startswith("yes"):
        return "yes"
    return ""


def build_prompt(query: str) -> str:
    lowered = query.lower()
    if "yes or no" in lowered:
        return query
    return f"{query}\nPlease answer yes or no."


def task_group(category: str) -> str:
    norm = category.strip().lower().replace(" ", "_").replace("-", "_")
    if norm in PERCEPTION_TASKS:
        return "perception"
    if norm in COGNITION_TASKS:
        return "cognition"
    return "unknown"


def compute_metrics(records: list[dict]) -> dict:
    by_category = defaultdict(list)
    for record in records:
        by_category[str(record.get("category", "unknown"))].append(record)

    categories = {}
    totals = {"perception": 0.0, "cognition": 0.0, "unknown": 0.0}
    for category, items in sorted(by_category.items()):
        total = len(items)
        correct = sum(1 for item in items if item.get("correct"))
        acc = 100.0 * correct / total if total else 0.0
        grouped = defaultdict(list)
        for item in items:
            grouped[str(item.get("image_id", ""))].append(item)
        plus_total = len(grouped)
        plus_correct = sum(1 for group in grouped.values() if group and all(x.get("correct") for x in group))
        acc_plus = 100.0 * plus_correct / plus_total if plus_total else 0.0
        score = acc + acc_plus
        group_name = task_group(category)
        totals[group_name] += score
        categories[category] = {
            "accuracy": acc,
            "accuracy_plus": acc_plus,
            "score": score,
            "num_questions": total,
            "num_images": plus_total,
            "group": group_name,
        }

    return {
        "categories": categories,
        "perception_score": totals["perception"],
        "cognition_score": totals["cognition"],
        "unknown_score": totals["unknown"],
        "total_score": sum(totals.values()),
        "num_samples": sum(len(items) for items in by_category.values()),
    }


def main():
    parser = argparse.ArgumentParser(description="Classic MME rule-based yes/no eval")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="Res-OPD")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mme-json", type=Path)
    source.add_argument("--mme-root", type=Path)
    parser.add_argument(
        "--categories",
        default="existence,count,position,color",
        help="Comma-separated MME categories.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-px", type=int, default=0)
    parser.add_argument("--target-px", type=int, default=448)
    parser.add_argument("--degradation-mode", choices=["square", "original"], default="square")
    parser.add_argument("--student-ratio", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--parallel-workers", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    out_dir = args.output_dir / "mme"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mme_json:
        rows = load_json_or_jsonl(args.mme_json)
        source_path = args.mme_json
    else:
        categories = [x.strip() for x in args.categories.split(",") if x.strip()]
        rows = load_official_mme_root(args.mme_root, categories)
        source_path = args.mme_root
    samples = [normalize_sample(row, i) for i, row in enumerate(rows)]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    results_path = out_dir / "eval_results.jsonl"
    metrics_path = out_dir / "mme_metrics.json"

    completed = load_existing(results_path)
    todo = []
    for sample in samples:
        uid = sample_uid(sample)
        if uid not in completed:
            sample = dict(sample)
            sample["sample_uid"] = uid
            todo.append(sample)

    print(f"MME classic samples={len(samples)} completed={len(completed)} remaining={len(todo)}")

    thread_local = threading.local()
    write_lock = threading.Lock()

    def get_client():
        client = getattr(thread_local, "client", None)
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=args.api_base, api_key=args.api_key, timeout=300)
            thread_local.client = client
        return client

    def run_one(sample: dict) -> dict:
        image_path = sample["images"][0]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_uri(
                                image_path,
                                degradation_mode=args.degradation_mode,
                                student_px=args.student_px,
                                target_px=args.target_px,
                                student_ratio=args.student_ratio,
                            )
                        },
                    },
                    {"type": "text", "text": build_prompt(sample["query"])},
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

        record = dict(sample)
        record["student_px"] = args.student_px
        record["target_px"] = args.target_px
        record["degradation_mode"] = args.degradation_mode
        record["student_ratio"] = args.student_ratio
        record["model_answer"] = answer
        record["pred_answer"] = extract_yes_no(answer)
        record["correct"] = record["pred_answer"] == record["response"]
        return record

    if todo:
        with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor, open(
            results_path, "a", encoding="utf-8"
        ) as f_out:
            future_to_sample = {executor.submit(run_one, sample): sample for sample in todo}
            done = len(completed)
            for future in as_completed(future_to_sample):
                try:
                    record = future.result()
                except Exception as exc:
                    sample = future_to_sample[future]
                    record = dict(sample)
                    record["model_answer"] = f"[ERROR] {exc}"
                    record["pred_answer"] = ""
                    record["correct"] = False
                with write_lock:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    done += 1
                if done % 100 == 0:
                    print(f"MME generated {done}/{len(samples)}")

    records = list(load_existing(results_path).values())
    metrics = compute_metrics(records)
    metrics.update(
        {
            "benchmark": "mme",
            "source_path": str(source_path),
            "extractor": "first_sentence_yes_no",
            "student_px": args.student_px,
            "target_px": args.target_px,
            "degradation_mode": args.degradation_mode,
            "student_ratio": args.student_ratio,
        }
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"MME total_score={metrics['total_score']:.2f} n={metrics['num_samples']}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
