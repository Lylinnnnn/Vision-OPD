#!/usr/bin/env python3
"""Generate AMBER responses and optionally run the official AMBER evaluator."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Optional

from PIL import Image


RES_OPD_ROOT = Path(__file__).resolve().parents[1]

QUERY_MAP = {
    "a": "query_all.json",
    "g": "query_generative.json",
    "d": "query_discriminative.json",
    "de": "query_discriminative-existence.json",
    "da": "query_discriminative-attribute.json",
    "dr": "query_discriminative-relation.json",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_eval_image(path: Path, degradation_mode: str, student_px: int, target_px: int, student_ratio: float):
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
    path: Path,
    degradation_mode: str = "square",
    student_px: int = 0,
    target_px: int = 448,
    student_ratio: float = 1.0,
) -> str:
    if degradation_mode == "original" or student_px > 0:
        image = load_eval_image(path, degradation_mode, student_px, target_px, student_ratio)
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        payload = buf.getvalue()
        mime = "image/jpeg"
    else:
        with open(path, "rb") as f:
            payload = f.read()
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(payload).decode('utf-8')}"


def resolve_image_path(amber_root: Path, image_root: Optional[Path], image_name: str) -> Path:
    candidates = []
    if image_root:
        candidates.append(image_root / image_name)
    candidates.extend(
        [
            amber_root / "images" / image_name,
            amber_root / "image" / image_name,
            amber_root / "data" / "images" / image_name,
            amber_root / image_name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"AMBER image not found: {image_name}. Tried: {candidates}")


def extract_yes_no(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "No"
    t = extract_final_response_text(text)
    lowered = t.lower()
    if lowered.startswith("yes") or re.search(r"\byes\b", lowered.split(".", 1)[0]):
        return "Yes"
    if lowered.startswith("no") or " not " in f" {lowered.split('.', 1)[0]} " or re.search(r"\bno\b", lowered.split(".", 1)[0]):
        return "No"
    return "No"


def extract_final_response_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1].strip()
    match = re.search(r"<answer>(.*?)</answer>", t, flags=re.IGNORECASE | re.DOTALL)
    if match:
        t = match.group(1).strip()
    for marker in ("Final Answer:", "Final answer:", "Answer:", "answer:"):
        if marker in t:
            t = t.split(marker, 1)[1].strip()
            break
    return t


def build_prompt(query: str, is_discriminative: bool) -> str:
    query = query.replace("<image>", "").strip()
    if is_discriminative and "yes or no" not in query.lower():
        return f"{query}\nPlease answer Yes or No."
    return query


def load_existing(path: Path) -> dict[int, dict]:
    records = {}
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return records
    for record in data:
        if isinstance(record, dict) and "id" in record and record.get("response"):
            records[int(record["id"])] = record
    return records


def save_official_responses(path: Path, records_by_id: dict[int, dict]) -> None:
    records = [records_by_id[key] for key in sorted(records_by_id)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def parse_official_stdout(text: str) -> dict:
    metrics = {}
    section = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith("Task:") or line.endswith(":"):
            section = line.rstrip(":").strip().replace(" ", "_").lower()
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().replace(" ", "_").lower()
            value = value.strip()
            try:
                value_obj = float(value)
            except ValueError:
                continue
            full_key = f"{section}_{key}" if section else key
            metrics[full_key] = value_obj
    return metrics


def amber_official_paths(args) -> list[tuple[str, str, Optional[Path]]]:
    return [
        ("--word_association", "data/relation.json", args.word_association),
        ("--safe_words", "data/safe_words.txt", args.safe_words),
        ("--annotation", "data/annotations.json", args.annotation),
        ("--metrics", "data/metrics.txt", args.metrics),
    ]


def run_official_eval(args, response_path: Path, out_dir: Path) -> None:
    parallel_evaluator = RES_OPD_ROOT / "scripts" / "run_amber_parallel.py"
    use_parallel = args.official_eval_workers > 1 and parallel_evaluator.exists()
    if use_parallel:
        cmd = [
            sys.executable,
            str(parallel_evaluator),
            "--inference_data",
            str(response_path),
            "--evaluation_type",
            args.evaluation_type,
            "--workers",
            str(args.official_eval_workers),
            "--amber_root",
            str(args.amber_root),
            "--output_metrics_json",
            str(out_dir / "amber_metrics_raw_counts.json"),
        ]
    else:
        evaluator = args.amber_root / "inference.py"
        if not evaluator.exists():
            raise FileNotFoundError(f"AMBER official inference.py not found: {evaluator}")
        cmd = [
            sys.executable,
            str(evaluator),
            "--inference_data",
            str(response_path),
            "--evaluation_type",
            args.evaluation_type,
        ]
    for opt, rel_default, value in amber_official_paths(args):
        cmd.extend([opt, str(value or (args.amber_root / rel_default))])
    proc = subprocess.run(
        cmd,
        cwd=str(args.amber_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    official_log = out_dir / "official_eval.log"
    official_log.write_text(proc.stdout, encoding="utf-8")
    summary = {
        "benchmark": "amber",
        "evaluation_type": args.evaluation_type,
        "official_eval_mode": "parallel" if use_parallel else "official",
        "official_eval_workers": args.official_eval_workers if use_parallel else 1,
        "official_returncode": proc.returncode,
        "official_log": str(official_log),
        "response_path": str(response_path),
        "student_px": args.student_px,
        "target_px": args.target_px,
        "degradation_mode": args.degradation_mode,
        "student_ratio": args.student_ratio,
        "metrics": parse_official_stdout(proc.stdout),
    }
    with open(out_dir / "amber_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if proc.returncode != 0:
        raise RuntimeError(f"AMBER evaluator failed. See {official_log}")


def main():
    parser = argparse.ArgumentParser(description="AMBER OpenAI-compatible inference wrapper")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="Res-OPD")
    parser.add_argument("--amber-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-px", type=int, default=0)
    parser.add_argument("--target-px", type=int, default=448)
    parser.add_argument("--degradation-mode", choices=["square", "original"], default="square")
    parser.add_argument("--student-ratio", type=float, default=1.0)
    parser.add_argument("--evaluation-type", choices=sorted(QUERY_MAP), default="a")
    parser.add_argument("--max-new-tokens-generative", type=int, default=4096)
    parser.add_argument("--max-new-tokens-discriminative", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--parallel-workers", type=int, default=64)
    parser.add_argument("--official-eval-workers", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--enable-thinking", choices=["True", "False"], default=None)
    parser.add_argument("--skip-official-eval", action="store_true")
    parser.add_argument("--word-association", type=Path, default=None)
    parser.add_argument("--safe-words", type=Path, default=None)
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=None)
    args = parser.parse_args()
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count")
    if args.official_eval_workers <= 0:
        raise ValueError("--official-eval-workers must be positive")

    query_path = args.amber_root / "data" / "query" / QUERY_MAP[args.evaluation_type]
    queries = load_json(query_path)
    if args.max_samples > 0:
        queries = queries[: args.max_samples]
    if args.shard_count > 1:
        queries = [
            item for idx, item in enumerate(queries)
            if idx % args.shard_count == args.shard_index
        ]
        print(
            f"AMBER shard {args.shard_index}/{args.shard_count}: "
            f"{len(queries)} samples after deterministic split"
        )

    out_dir = args.output_dir / "amber"
    out_dir.mkdir(parents=True, exist_ok=True)
    response_path = out_dir / f"amber_{args.evaluation_type}_responses.json"
    raw_path = out_dir / "raw_results.jsonl"

    completed = load_existing(response_path)
    todo = [item for item in queries if int(item["id"]) not in completed]
    print(f"AMBER samples={len(queries)} completed={len(completed)} remaining={len(todo)}")

    thread_local = threading.local()
    write_lock = threading.Lock()

    def get_client():
        client = getattr(thread_local, "client", None)
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=args.api_base, api_key=args.api_key, timeout=300)
            thread_local.client = client
        return client

    def run_one(item: dict) -> tuple[dict, dict]:
        item_id = int(item["id"])
        is_discriminative = item_id >= 1005
        image_path = resolve_image_path(args.amber_root, args.image_root, item["image"])
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
                    {"type": "text", "text": build_prompt(item["query"], is_discriminative)},
                ],
            }
        ]
        max_tokens = args.max_new_tokens_discriminative if is_discriminative else args.max_new_tokens_generative
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
                    max_tokens=max_tokens,
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
        official_answer = extract_yes_no(answer) if is_discriminative else extract_final_response_text(answer)
        official_record = {"id": item_id, "response": official_answer}
        raw_record = dict(item)
        raw_record.update(
            {
                "image_path": str(image_path),
                "student_px": args.student_px,
                "target_px": args.target_px,
                "degradation_mode": args.degradation_mode,
                "student_ratio": args.student_ratio,
                "model_answer": answer,
                "official_response": official_answer,
                "is_discriminative": is_discriminative,
            }
        )
        return official_record, raw_record

    records_by_id = dict(completed)
    if todo:
        with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor, open(
            raw_path, "a", encoding="utf-8"
        ) as f_raw:
            futures = {executor.submit(run_one, item): item for item in todo}
            done = len(completed)
            for future in as_completed(futures):
                try:
                    official_record, raw_record = future.result()
                except Exception as exc:
                    item = futures[future]
                    item_id = int(item["id"])
                    official_record = {"id": item_id, "response": "No" if item_id >= 1005 else f"[ERROR] {exc}"}
                    raw_record = dict(item)
                    raw_record["model_answer"] = f"[ERROR] {exc}"
                with write_lock:
                    records_by_id[int(official_record["id"])] = official_record
                    f_raw.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
                    f_raw.flush()
                    done += 1
                if done % 100 == 0:
                    print(f"AMBER generated {done}/{len(queries)}")

    save_official_responses(response_path, records_by_id)
    if args.skip_official_eval:
        print(f"Saved AMBER official-format responses to: {response_path}")
        return
    run_official_eval(args, response_path, out_dir)
    print(f"Saved AMBER metrics to: {out_dir / 'amber_metrics.json'}")


if __name__ == "__main__":
    main()
