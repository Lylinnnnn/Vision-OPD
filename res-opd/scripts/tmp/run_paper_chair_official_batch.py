#!/usr/bin/env python3

import argparse
import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
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
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--transition-wait-seconds", type=int, default=180)
    parser.add_argument("--require-idle-gpus", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-oss-preflight", action="store_true")
    parser.add_argument(
        "--oss-checkpoint-access",
        choices=("mounted", "download"),
        default=os.environ.get("OSS_CHECKPOINT_ACCESS_MODE", "mounted"),
    )
    parser.add_argument(
        "--oss-mount-root",
        default=os.environ.get("OSS_MOUNT_ROOT", "/data/oss_bucket_0"),
    )
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


def local_checkpoint_dir(entry, root):
    if entry["source_type"] != "oss_checkpoint":
        return None
    return root / "res-opd" / "checkpoints" / entry["primary"] / entry["step"]


@contextlib.contextmanager
def entry_lock(entry, root, suffix):
    lock_root = root / "res-opd" / "eval_results" / ".paper_chair_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_dir = lock_root / f"{suffix}.{entry['index']:02d}.lock"
    try:
        lock_dir.mkdir()
    except FileExistsError as error:
        owner_path = lock_dir / "owner.json"
        try:
            owner = owner_path.read_text(encoding="utf-8")
        except OSError:
            owner = "<owner metadata unavailable>"
        raise RuntimeError(
            f"Entry {entry['index']:02d} is already running on the shared filesystem; "
            f"lock={lock_dir}, owner={owner}"
        ) from error

    owner_path = lock_dir / "owner.json"
    owner_path.write_text(
        json.dumps(
            {
                "host": os.uname().nodename,
                "pid": os.getpid(),
                "index": entry["index"],
                "label": entry["label"],
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        yield
    finally:
        try:
            owner_path.unlink()
        except FileNotFoundError:
            pass
        try:
            lock_dir.rmdir()
        except FileNotFoundError:
            pass


def output_complete(path, expected_count):
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
    return (
        count == expected_count
        and rows == expected_count
        and schema == "official_chair_caption_and_mention_v1"
    )


def run_capture(command):
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def oss_uri_to_mount_path(uri, mount_root):
    match = re.fullmatch(r"oss://[^/]+/(.+)", uri.rstrip("/"))
    if not match:
        raise ValueError(f"Invalid OSS URI: {uri}")
    return Path(mount_root) / match.group(1)


def listening_pids(port):
    if shutil.which("lsof"):
        result = run_capture(["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"])
    elif shutil.which("fuser"):
        result = run_capture(["fuser", "-n", "tcp", str(port)])
    elif shutil.which("ss"):
        result = run_capture(["ss", "-ltnp", f"sport = :{port}"])
        return set(re.findall(r"pid=(\d+)", result.stdout))
    else:
        raise RuntimeError("Cannot inspect ports: install lsof, fuser, or ss")
    return set(re.findall(r"\b\d+\b", result.stdout))


def gpu_processes(gpu_ids):
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi is required for --require-idle-gpus")
    uuid_result = run_capture(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    )
    if uuid_result.returncode != 0:
        raise RuntimeError(f"Failed to query GPU UUIDs:\n{uuid_result.stdout}")
    uuid_to_index = {}
    for raw in uuid_result.stdout.splitlines():
        index, uuid = (value.strip() for value in raw.split(",", 1))
        uuid_to_index[uuid] = index

    process_result = run_capture(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    if process_result.returncode != 0:
        raise RuntimeError(f"Failed to query GPU processes:\n{process_result.stdout}")
    selected = set(gpu_ids)
    processes = {}
    for raw in process_result.stdout.splitlines():
        fields = [value.strip() for value in raw.split(",", 2)]
        if len(fields) != 3 or uuid_to_index.get(fields[0]) not in selected:
            continue
        processes[fields[1]] = {
            "gpu": uuid_to_index[fields[0]],
            "name": fields[2],
        }
    return processes


def transition_state(base_port, shard_count, gpu_ids, check_gpus):
    ports = {}
    for shard_index in range(shard_count):
        port = base_port + shard_index * 10
        pids = listening_pids(port)
        if pids:
            ports[port] = sorted(pids)
    gpus = gpu_processes(gpu_ids) if check_gpus else {}
    return ports, gpus


def wait_for_clean_transition(args, gpu_ids, allowed_gpu_pids):
    deadline = time.monotonic() + args.transition_wait_seconds
    while True:
        ports, gpu_process_map = transition_state(
            args.base_port,
            args.eval_shard_count,
            gpu_ids,
            args.require_idle_gpus,
        )
        new_gpu_processes = {
            pid: details
            for pid, details in gpu_process_map.items()
            if pid not in allowed_gpu_pids
        }
        if not ports and not new_gpu_processes:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Previous task did not release resources; refusing to start the next task: "
                f"ports={ports}, new_gpu_processes={new_gpu_processes}"
            )
        time.sleep(2)


def preflight(entries, args):
    failures = []
    ossutil = shutil.which("ossutil") if args.oss_checkpoint_access == "download" else None
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
        base = os.environ.get("OSS_BASE", "oss://industry-algo/yanlin/ckpt/OPD/v4").rstrip("/")
        checkpoint_uri = f"{base}/{entry['oss_name']}/{entry['step']}"
        if args.oss_checkpoint_access == "mounted":
            checkpoint_dir = oss_uri_to_mount_path(checkpoint_uri, args.oss_mount_root)
            for filename in ("model.safetensors", "config.json"):
                path = checkpoint_dir / filename
                try:
                    available = path.is_file() and path.stat().st_size > 0
                except OSError:
                    available = False
                if not available:
                    failures.append(
                        f"{entry['index']:02d} {entry['label']}: missing mounted file {path}"
                    )
        else:
            if not ossutil:
                failures.append(f"{entry['index']:02d} {entry['label']}: ossutil not found")
                continue
            for filename in ("model.safetensors", "config.json"):
                uri = f"{checkpoint_uri}/{filename}"
                result = run_capture([ossutil, "stat", uri])
                if result.returncode != 0:
                    failures.append(f"{entry['index']:02d} {entry['label']}: missing {uri}")
    if failures:
        raise RuntimeError("Preflight failed; no evaluation was started:\n  " + "\n  ".join(failures))


def main():
    args = parse_args()
    if args.num_jobs < 1 or not 0 <= args.job_index < args.num_jobs:
        raise ValueError("--job-index must be in [0, --num-jobs)")
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.transition_wait_seconds < 1:
        raise ValueError("--transition-wait-seconds must be positive")
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
        f"suffix={args.output_suffix}; checkpoint_access={args.oss_checkpoint_access}",
        flush=True,
    )
    preflight(selected, args)
    print("All selected model/checkpoint sources passed preflight.", flush=True)
    initial_ports, initial_gpu_processes = transition_state(
        args.base_port,
        args.eval_shard_count,
        gpu_ids,
        args.require_idle_gpus,
    )
    if initial_ports:
        raise RuntimeError(f"Selected ports are busy before evaluation: {initial_ports}")
    if args.require_idle_gpus and initial_gpu_processes:
        raise RuntimeError(
            f"Selected GPUs are not idle before evaluation: {initial_gpu_processes}"
        )
    allowed_gpu_pids = set(initial_gpu_processes)

    launcher = root / "res-opd" / "scripts" / "tmp" / "run_paper_chair_describe45_entry.sh"
    completed = 0
    skipped = 0
    for position, entry in enumerate(selected, 1):
        target = output_dir(entry, root, args.output_suffix)
        print(
            f"\n[{position}/{len(selected)}] {entry['index']:02d} {entry['label']}\nOutput: {target}",
            flush=True,
        )
        if output_complete(target, args.max_samples) and not args.force:
            print("Complete official output exists; skipping.", flush=True)
            skipped += 1
            continue
        if args.dry_run:
            command = ["bash", str(launcher), str(entry["index"]), str(args.base_port)]
            print(" ".join(shlex.quote(value) for value in command), flush=True)
            continue
        with entry_lock(entry, root, args.output_suffix):
            if output_complete(target, args.max_samples) and not args.force:
                print("Output completed while waiting for the shared lock; skipping.", flush=True)
                skipped += 1
                continue
            command = ["bash", str(launcher), str(entry["index"]), str(args.base_port)]
            env = os.environ.copy()
            env["CHAIR_RERUN_PROMPT"] = args.prompt
            env["CHAIR_RERUN_OUTPUT_SUFFIX"] = args.output_suffix
            env["GPU_LIST"] = args.gpu_list
            env["EVAL_SHARD_COUNT"] = str(args.eval_shard_count)
            env["CHAIR_MAX_SAMPLES"] = str(args.max_samples)
            env["OSS_CHECKPOINT_ACCESS_MODE"] = args.oss_checkpoint_access
            env["OSS_MOUNT_ROOT"] = args.oss_mount_root
            if args.model_root:
                env["MODEL_ROOT"] = args.model_root
            print(" ".join(shlex.quote(value) for value in command), flush=True)
            result = subprocess.run(command, env=env, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Evaluation failed for {entry['index']:02d} {entry['label']}: "
                    f"exit={result.returncode}"
                )
            if not output_complete(target, args.max_samples):
                raise RuntimeError(
                    f"Evaluation succeeded but official output is incomplete: {target}"
                )
            checkpoint_dir = local_checkpoint_dir(entry, root)
            if (
                args.oss_checkpoint_access == "download"
                and checkpoint_dir is not None
                and checkpoint_dir.exists()
            ):
                raise RuntimeError(
                    "Evaluation finished but the downloaded checkpoint was not removed; "
                    f"refusing to start the next task: {checkpoint_dir}"
                )
            wait_for_clean_transition(args, gpu_ids, allowed_gpu_pids)
            completed += 1
            print(
                f"Verified {args.max_samples} generations, official CHAIR metrics, "
                "released ports, no residual GPU process, and checkpoint source lifecycle.",
                flush=True,
            )

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
