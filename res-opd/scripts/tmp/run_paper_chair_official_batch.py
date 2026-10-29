#!/usr/bin/env python3

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="res-opd/eval/manifests/paper_chair_describe45_sources.txt",
    )
    parser.add_argument("--indices", default="1-45")
    parser.add_argument("--num-jobs", type=int, default=1)
    parser.add_argument("--job-index", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=8600)
    parser.add_argument("--gpu-list", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--eval-shard-count", type=int, default=8)
    parser.add_argument("--model-root", default=os.environ.get("MODEL_ROOT", ""))
    parser.add_argument("--prompt", default="Describe the image.")
    parser.add_argument("--output-suffix", default="chair_describe_image_official")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-oss-preflight", action="store_true")
    return parser.parse_args()


def root_dir():
    return Path(__file__).resolve().parents[3]


def parse_indices(spec):
    values = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(value) for value in item.split("-", 1))
            if start < 1 or end < start:
                raise ValueError(f"Invalid index range: {item}")
            values.update(range(start, end + 1))
        else:
            value = int(item)
            if value < 1:
                raise ValueError(f"Invalid index: {item}")
            values.add(value)
    return values


def load_manifest(path):
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("|")
        if len(fields) != 7:
            raise ValueError(f"Malformed manifest row: {raw}")
        index, label, source_type, primary, oss_name, step, student_ratio = fields
        entries.append(
            {
                "index": int(index),
                "label": label,
                "source_type": source_type,
                "primary": primary,
                "oss_name": oss_name,
                "step": step,
                "student_ratio": student_ratio,
            }
        )
    if len(entries) != 45 or [row["index"] for row in entries] != list(range(1, 46)):
        raise ValueError("The paper manifest must contain exactly the ordered indices 1-45")
    labels = [row["label"] for row in entries]
    if len(labels) != len(set(labels)):
        raise ValueError("The paper manifest contains duplicate labels")
    return entries


def model_path(entry, model_root):
    path = Path(entry["primary"])
    return Path(model_root) / path.name if model_root else path


def ratio_tag(value):
    return str(value).replace(".", "p")


def output_dir(entry, root, suffix):
    if entry["source_type"] == "local_base":
        experiment = Path(entry["primary"]).name
    else:
        experiment = f"{entry['primary']}_{entry['step']}"
    dataset = f"train5000_test1000_original_sr{ratio_tag(entry['student_ratio'])}"
    return root / "res-opd" / "eval_results" / "instruct" / "full" / experiment / dataset / suffix


def output_complete(path):
    metrics_path = path / "chair_metrics.json"
    results_path = path / "eval_results.jsonl"
    if not metrics_path.is_file() or not results_path.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        count = int(metrics.get("total_results", metrics.get("total_captions", 0)))
        schema = metrics.get("metric_schema")
        with results_path.open("r", encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return count == 1000 and rows == 1000 and schema == "official_chair_caption_and_mention_v1"


def run_capture(command):
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def preflight(entries, args):
    failures = []
    ossutil = shutil.which("ossutil")
    for entry in entries:
        if entry["source_type"] == "local_base":
            path = model_path(entry, args.model_root)
            if not path.is_dir() or not (path / "config.json").is_file():
                failures.append(f"{entry['index']:02d} {entry['label']}: incomplete model {path}")
            continue
        if entry["source_type"] != "oss_checkpoint":
            failures.append(
                f"{entry['index']:02d} {entry['label']}: unknown source {entry['source_type']}"
            )
            continue
        if args.skip_oss_preflight:
            continue
        if not ossutil:
            failures.append(f"{entry['index']:02d} {entry['label']}: ossutil not found")
            continue
        base = os.environ.get("OSS_BASE", "oss://industry-algo/yanlin/ckpt/OPD/v4").rstrip("/")
        for filename in ("model.safetensors", "config.json"):
            uri = f"{base}/{entry['oss_name']}/{entry['step']}/{filename}"
            result = run_capture([ossutil, "stat", uri])
            if result.returncode != 0:
                failures.append(f"{entry['index']:02d} {entry['label']}: missing {uri}")
    if failures:
        raise RuntimeError("Preflight failed; no evaluation was started:\n  " + "\n  ".join(failures))


def main():
    args = parse_args()
    if args.num_jobs < 1 or not 0 <= args.job_index < args.num_jobs:
        raise ValueError("--job-index must be in [0, --num-jobs)")
    gpu_ids = [value.strip() for value in args.gpu_list.split(",") if value.strip()]
    if len(gpu_ids) != args.eval_shard_count or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpu-list must contain exactly --eval-shard-count unique GPU IDs")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.output_suffix):
        raise ValueError("--output-suffix must be one safe path component")

    root = root_dir()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = root / manifest
    entries = load_manifest(manifest)
    requested = parse_indices(args.indices)
    selected = [
        row
        for row in entries
        if row["index"] in requested and (row["index"] - 1) % args.num_jobs == args.job_index
    ]
    if not selected:
        raise ValueError("No manifest entries matched this partition")

    print(
        f"Resolved {len(selected)}/45 entries; job={args.job_index}/{args.num_jobs}; "
        f"suffix={args.output_suffix}",
        flush=True,
    )
    preflight(selected, args)
    print("All selected model/checkpoint sources passed preflight.", flush=True)

    launcher = root / "res-opd" / "scripts" / "tmp" / "run_paper_chair_describe45_entry.sh"
    completed = 0
    skipped = 0
    for position, entry in enumerate(selected, 1):
        target = output_dir(entry, root, args.output_suffix)
        print(
            f"\n[{position}/{len(selected)}] {entry['index']:02d} {entry['label']}\nOutput: {target}",
            flush=True,
        )
        if output_complete(target) and not args.force:
            print("Complete official output exists; skipping.", flush=True)
            skipped += 1
            continue
        command = ["bash", str(launcher), str(entry["index"]), str(args.base_port)]
        env = os.environ.copy()
        env["CHAIR_RERUN_PROMPT"] = args.prompt
        env["CHAIR_RERUN_OUTPUT_SUFFIX"] = args.output_suffix
        env["GPU_LIST"] = args.gpu_list
        env["EVAL_SHARD_COUNT"] = str(args.eval_shard_count)
        if args.model_root:
            env["MODEL_ROOT"] = args.model_root
        print(" ".join(shlex.quote(value) for value in command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, env=env, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Evaluation failed for {entry['index']:02d} {entry['label']}: "
                f"exit={result.returncode}"
            )
        if not output_complete(target):
            raise RuntimeError(f"Evaluation succeeded but official output is incomplete: {target}")
        completed += 1
        print("Verified 1000 generations and official CHAIR metrics.", flush=True)

    print(
        f"\nFinished selected={len(selected)} completed={completed} skipped={skipped} "
        f"dry_run={args.dry_run}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
