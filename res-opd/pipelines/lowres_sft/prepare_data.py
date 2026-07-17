#!/usr/bin/env python3
"""Generate low-resolution captions and build a Res-OPD SFT parquet.

The resulting SFT examples use original images as model input and low-res
generated captions as assistant supervision.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


PROMPT_FALLBACK = (
    "Please describe this image in detail. Include all visible objects, "
    "their spatial relationships, colors, sizes, and any text or fine details."
)

THINK_CLOSE = "</think>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--generation-jsonl", required=True, type=Path)
    parser.add_argument("--api-base", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--lowres-ratio", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--parallel-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--enable-thinking", choices=["True", "False"], default=None)
    parser.add_argument("--final-answer-only", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--no-generate", action="store_true",
                        help="Do not call the API; fail if cached generations are incomplete.")
    parser.add_argument("--check-generations-only", action="store_true",
                        help="Only validate cached generations and the completion marker.")
    parser.add_argument("--strict-model-cache", action="store_true",
                        help="Require completion marker model_name to match --model-name.")
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    return value


def data_roots() -> list[Path]:
    home = Path.home()
    candidates = [
        os.environ.get("RES_OPD_DATA_ROOT", ""),
        os.environ.get("BENCHMARK_DATA_DIR", ""),
        os.environ.get("VISION_BENCHMARK_DATA_DIR", ""),
        home / "notebook" / "data",
        home / "notebook" / "yanlin" / "data",
        Path("/home/zhengyanzhao.zyz/notebook/yanlin/data"),
        Path("/home/liuyanlin.lyl/notebook/data"),
    ]
    roots: list[Path] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and path not in roots:
            roots.append(path)
    return roots


def resolve_path_text(path_text: str) -> str:
    path = Path(path_text).expanduser()
    if path.exists():
        return str(path)
    text = str(path)
    for marker in (
        "/COCO/",
        "/AMBER/",
        "/MMStar_images/",
        "/CVBench_2d_images/",
        "/CVBench_3d_images/",
    ):
        if marker not in text:
            continue
        suffix = text.split(marker, 1)[1]
        for root in data_roots():
            candidate = root / marker.strip("/") / suffix
            if candidate.exists():
                return str(candidate)
    return str(path)


def resolve_image_entry(image: Any) -> Any:
    image = to_builtin(image)
    if isinstance(image, str):
        return resolve_path_text(image)
    if isinstance(image, dict):
        resolved = dict(image)
        if isinstance(resolved.get("path"), str):
            resolved["path"] = resolve_path_text(resolved["path"])
        elif isinstance(resolved.get("image"), str):
            resolved["image"] = resolve_path_text(resolved["image"])
        return resolved
    return image


def image_path_for_generation(images: list[Any]) -> str:
    if not images:
        raise ValueError("sample has no images")
    image = resolve_image_entry(images[0])
    if isinstance(image, str):
        return image
    if isinstance(image, dict):
        if isinstance(image.get("path"), str):
            return image["path"]
        if isinstance(image.get("image"), str):
            return image["image"]
    raise TypeError(f"Unsupported image entry for generation: {type(image)}")


def make_ratio_degraded_image(image_path: str, ratio: float) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if ratio <= 0:
        return Image.new("RGB", image.size, (127, 127, 127))
    if ratio >= 1.0:
        return image
    width, height = image.size
    small_size = (
        max(1, int(round(width * ratio))),
        max(1, int(round(height * ratio))),
    )
    return image.resize(small_size, Image.LANCZOS).resize((width, height), Image.LANCZOS)


def extract_prompt_text(prompt_messages: Any) -> str:
    messages = to_builtin(prompt_messages)
    if not isinstance(messages, list):
        return PROMPT_FALLBACK
    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
    prompt = "\n".join(chunks)
    prompt = prompt.replace("<image>", "").strip()
    return prompt or PROMPT_FALLBACK


def extract_final_response_text(text: str) -> str:
    final = (text or "").strip()
    think_end = final.rfind(THINK_CLOSE)
    if think_end != -1:
        final = final[think_end + len(THINK_CLOSE):].strip()
    match = re.search(r"<answer>(.*?)</answer>", final, flags=re.IGNORECASE | re.DOTALL)
    if match:
        final = match.group(1).strip()
    for marker in ("Final Answer:", "Final answer:", "Answer:", "answer:"):
        if marker in final:
            final = final.split(marker, 1)[1].strip()
            break
    return final


def finalize_generation(raw_text: str, final_answer_only: bool) -> dict[str, Any]:
    raw_text = raw_text or ""
    if not final_answer_only:
        return {
            "lowres_caption": raw_text,
            "raw_lowres_caption": raw_text,
            "thinking_status": "not_requested",
            "final_answer_available": True,
        }
    if THINK_CLOSE not in raw_text:
        return {
            "lowres_caption": "",
            "raw_lowres_caption": raw_text,
            "thinking_status": "unclosed",
            "final_answer_available": False,
        }
    final_text = extract_final_response_text(raw_text)
    return {
        "lowres_caption": final_text,
        "raw_lowres_caption": raw_text,
        "thinking_status": "closed",
        "final_answer_available": bool(final_text.strip()),
    }


def sample_uid(row: dict[str, Any], index: int) -> str:
    extra = to_builtin(row.get("extra_info", {}))
    if isinstance(extra, dict) and extra.get("image_id") is not None:
        return f"image:{extra['image_id']}"
    return f"index:{index}"


def load_existing_generations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row.get("sample_uid")
            caption = str(row.get("lowres_caption", "") or "")
            if uid and caption and not caption.startswith("[ERROR]"):
                out[uid] = row
    return out


def generation_marker_path(generation_jsonl: Path) -> Path:
    if generation_jsonl.suffix:
        return generation_jsonl.with_name(f"{generation_jsonl.stem}.complete.json")
    return generation_jsonl.with_name(f"{generation_jsonl.name}.complete.json")


def uid_hash(uids: list[str]) -> str:
    digest = hashlib.sha256()
    for uid in uids:
        digest.update(uid.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def expected_uids(rows: list[dict[str, Any]]) -> list[str]:
    return [sample_uid(row, idx) for idx, row in enumerate(rows)]


def valid_generation(record: dict[str, Any] | None, lowres_ratio: float) -> bool:
    if not record:
        return False
    caption = str(record.get("lowres_caption", "") or "")
    if not caption or caption.startswith("[ERROR]"):
        return False
    if record.get("final_answer_available") is False:
        return False
    record_ratio = record.get("lowres_ratio")
    if record_ratio is not None:
        try:
            if abs(float(record_ratio) - lowres_ratio) > 1e-6:
                return False
        except Exception:
            return False
    return True


def verify_generation_cache(
    rows: list[dict[str, Any]],
    existing: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    *,
    require_marker: bool,
) -> tuple[bool, str, dict[str, Any]]:
    uids = expected_uids(rows)
    missing = [uid for uid in uids if not valid_generation(existing.get(uid), args.lowres_ratio)]
    metadata = {
        "complete": not missing,
        "num_input_rows": len(rows),
        "num_valid_generations": len(uids) - len(missing),
        "num_missing": len(missing),
        "lowres_ratio": args.lowres_ratio,
        "model_name": args.model_name,
        "final_answer_only": args.final_answer_only,
        "uid_hash": uid_hash(uids),
        "generation_jsonl": str(args.generation_jsonl),
    }
    if missing:
        return False, f"missing/invalid generations: {len(missing)} (first={missing[:5]})", metadata

    marker_path = generation_marker_path(args.generation_jsonl)
    if require_marker:
        if not marker_path.exists():
            return False, f"missing completion marker: {marker_path}", metadata
        try:
            marker = json.loads(marker_path.read_text())
        except Exception as exc:
            return False, f"invalid completion marker {marker_path}: {exc}", metadata
        checks = [
            marker.get("complete") is True,
            int(marker.get("num_input_rows", -1)) == metadata["num_input_rows"],
            int(marker.get("num_valid_generations", -1)) == metadata["num_valid_generations"],
            abs(float(marker.get("lowres_ratio", -1.0)) - args.lowres_ratio) <= 1e-6,
            bool(marker.get("final_answer_only", False)) == args.final_answer_only,
            marker.get("uid_hash") == metadata["uid_hash"],
        ]
        if args.strict_model_cache:
            checks.append(marker.get("model_name") == args.model_name)
        if not all(checks):
            return False, f"completion marker does not match current request: {marker_path}", metadata

    return True, "complete", metadata


def write_generation_marker(path: Path, metadata: dict[str, Any]) -> None:
    marker = dict(metadata)
    marker["complete"] = True
    marker["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(marker, indent=2, ensure_ascii=False))


def generate_one(
    thread_local: threading.local,
    api_base: str,
    model_name: str,
    image_path: str,
    prompt: str,
    lowres_ratio: float,
    max_new_tokens: int,
    max_retries: int,
    enable_thinking: str | None,
    final_answer_only: bool,
) -> dict[str, Any]:
    client = getattr(thread_local, "client", None)
    if client is None:
        from openai import OpenAI

        client = OpenAI(base_url=api_base, api_key="EMPTY", timeout=300)
        thread_local.client = client

    image = make_ratio_degraded_image(image_path, lowres_ratio)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    image_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    for attempt in range(1, max_retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            }
            if enable_thinking is not None:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": enable_thinking == "True"}
                }
            response = client.chat.completions.create(**kwargs)
            raw_text = response.choices[0].message.content or ""
            return finalize_generation(raw_text, final_answer_only)
        except Exception as exc:
            if attempt == max_retries:
                return {
                    "lowres_caption": f"[ERROR] {exc}",
                    "raw_lowres_caption": "",
                    "thinking_status": "error",
                    "final_answer_available": False,
                }
            time.sleep(min(2 ** attempt, 8))
    return {
        "lowres_caption": "[ERROR] unknown",
        "raw_lowres_caption": "",
        "thinking_status": "error",
        "final_answer_available": False,
    }


def build_sft_record(row: dict[str, Any], lowres_caption: str) -> dict[str, Any]:
    prompt_messages = copy.deepcopy(to_builtin(row.get("prompt", [])))
    if not isinstance(prompt_messages, list):
        raise TypeError("prompt column must be a list of chat messages")
    messages = prompt_messages + [{"role": "assistant", "content": lowres_caption}]
    images = [resolve_image_entry(image) for image in to_builtin(row.get("images", []))]
    extra_info = to_builtin(row.get("extra_info", {}))
    if not isinstance(extra_info, dict):
        extra_info = {"raw_extra_info": extra_info}
    extra_info = dict(extra_info)
    extra_info["sft_label_source"] = "lowres_generation"
    return {
        "data_source": "coco_res_opd_lowres_sft",
        "messages": messages,
        "images": images,
        "ability": to_builtin(row.get("ability", "image_captioning")),
        "reward_model": to_builtin(row.get("reward_model", {})),
        "extra_info": extra_info,
    }


def main() -> int:
    args = parse_args()
    if not args.train_file.exists():
        raise FileNotFoundError(f"Missing train parquet: {args.train_file}")

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.generation_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.force_regenerate and args.generation_jsonl.exists():
        args.generation_jsonl.unlink()
    if args.force_regenerate:
        generation_marker_path(args.generation_jsonl).unlink(missing_ok=True)

    dataframe = pd.read_parquet(args.train_file)
    if args.max_samples > 0:
        dataframe = dataframe.iloc[: args.max_samples]
    rows = [row.to_dict() for _, row in dataframe.iterrows()]
    existing = load_existing_generations(args.generation_jsonl)

    if args.check_generations_only:
        ok, reason, metadata = verify_generation_cache(rows, existing, args, require_marker=True)
        print(json.dumps({"ok": ok, "reason": reason, **metadata}, indent=2, ensure_ascii=False))
        return 0 if ok else 2

    todo: list[tuple[int, str, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        uid = sample_uid(row, idx)
        if uid not in existing:
            todo.append((idx, uid, row))

    print(
        f"Low-res SFT data: total={len(rows)} done={len(existing)} "
        f"todo={len(todo)} ratio={args.lowres_ratio}"
    )

    if todo:
        if args.no_generate:
            preview = ", ".join(uid for _, uid, _ in todo[:10])
            raise RuntimeError(f"Cached generations are incomplete and --no-generate was set: {preview}")
        if not args.api_base or not args.model_name:
            raise RuntimeError("--api-base and --model-name are required when generation is needed")
        thread_local = threading.local()
        write_lock = threading.Lock()
        done = len(existing)

        def run_item(item: tuple[int, str, dict[str, Any]]) -> dict[str, Any]:
            idx, uid, row = item
            images = to_builtin(row.get("images", []))
            image_path = image_path_for_generation(images)
            prompt = extract_prompt_text(row.get("prompt", []))
            result = generate_one(
                thread_local,
                args.api_base,
                args.model_name,
                image_path,
                prompt,
                args.lowres_ratio,
                args.max_new_tokens,
                args.max_retries,
                args.enable_thinking,
                args.final_answer_only,
            )
            extra = to_builtin(row.get("extra_info", {}))
            out = {
                "sample_uid": uid,
                "row_index": idx,
                "image_path": image_path,
                "lowres_ratio": args.lowres_ratio,
                "prompt_text": prompt,
                "image_id": extra.get("image_id") if isinstance(extra, dict) else None,
            }
            out.update(result)
            return out

        with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
            futures = [executor.submit(run_item, item) for item in todo]
            with args.generation_jsonl.open("a") as f:
                for future in as_completed(futures):
                    record = future.result()
                    with write_lock:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                    existing[record["sample_uid"]] = record
                    done += 1
                    if done % 50 == 0 or done == len(rows):
                        print(f"Generated {done}/{len(rows)}")

    sft_records: list[dict[str, Any]] = []
    missing: list[str] = []
    for idx, row in enumerate(rows):
        uid = sample_uid(row, idx)
        generation = existing.get(uid)
        caption = str(generation.get("lowres_caption", "") if generation else "")
        if not generation or not caption or caption.startswith("[ERROR]"):
            missing.append(uid)
            continue
        sft_records.append(build_sft_record(row, caption))

    if missing and not args.allow_missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Missing/invalid generations for {len(missing)} samples: {preview}")
    if not sft_records:
        raise RuntimeError("No valid SFT records were produced")

    import datasets

    dataset = datasets.Dataset.from_list(sft_records)
    dataset.to_parquet(str(args.output_parquet))
    manifest = {
        "train_file": str(args.train_file),
        "output_parquet": str(args.output_parquet),
        "generation_jsonl": str(args.generation_jsonl),
        "num_input_rows": len(rows),
        "num_sft_records": len(sft_records),
        "num_missing": len(missing),
        "lowres_ratio": args.lowres_ratio,
        "model_name": args.model_name,
        "final_answer_only": args.final_answer_only,
    }
    with (args.output_parquet.parent / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    if not missing:
        ok, reason, cache_metadata = verify_generation_cache(rows, existing, args, require_marker=False)
        if not ok:
            raise RuntimeError(f"Generation cache unexpectedly incomplete after SFT build: {reason}")
        write_generation_marker(generation_marker_path(args.generation_jsonl), cache_metadata)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
