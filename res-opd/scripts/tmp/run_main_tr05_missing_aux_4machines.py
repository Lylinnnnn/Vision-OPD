#!/usr/bin/env python3
"""Run missing main-table POPE/AMBER and InternVL Base vision jobs on four nodes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from check_main_tr05_pope_amber import (
    DATASET, EXPECTED_AMBER, EXPECTED_POPE, ROOT, TARGETS, VISION_EXPECTED,
    amber_metrics, build_specs, inspect_all, inspect_vision_all, internvl_base_dirs, target_dirs,
)
from run_three_tables_fill import gpu_memory_mib
from wait_eval_capacity import port_is_free


INTERNVL_VISION = "InternVL3-Base-vision"
QUEUES = {
    0: ("Base-2B", "M021", "Base-8B", "M032"),
    1: ("M005", "Base-4B", "M041", "M039"),
    2: ("M006", "M014", "M024", "M020", "M055"),
    3: ("M018", "M025", "M028", "M064", INTERNVL_VISION),
}
TARGET_BY_ID = {target.row_id: target for target in TARGETS
                if target.method not in ("OPERA", "HALC")}
SPECS = {(spec.size, spec.method): spec for spec in build_specs() if spec.family == "Qwen3VL"}
DEFAULT_MODEL_ROOT = Path("/home/liuyanlin.lyl/notebook/model")
DEFAULT_OSS_ROOT = Path("/data/oss_bucket_0/yanlin/ckpt/OPD/v4")
OSS_URI = "oss://industry-algo/yanlin/ckpt/OPD/v4"


def legacy_amber_errors(directories, expected=EXPECTED_AMBER):
    for directory in directories:
        raw_path = directory / "amber/raw_results.jsonl"
        response_path = directory / "amber/amber_a_responses.json"
        if not raw_path.is_file() or not response_path.is_file():
            continue
        try:
            if amber_metrics(directory) is None:
                continue
            answers = {}
            errors = set()
            duplicate_or_empty = False
            with raw_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    identity = str(row["id"])
                    answer = str(row.get("model_answer") or "").strip()
                    if identity in answers or not answer:
                        duplicate_or_empty = True
                        break
                    answers[identity] = answer
                    if answer.startswith(("[ERROR]", "[API_ERROR]", "[FUTURE_ERROR]")):
                        errors.add(identity)
            if duplicate_or_empty or len(answers) != expected or not errors:
                continue
            responses = json.loads(response_path.read_text(encoding="utf-8"))
            if not isinstance(responses, list) or len(responses) != expected:
                continue
            by_id = {str(row["id"]): str(row.get("response") or "").strip() for row in responses}
            if len(by_id) != expected or set(by_id) != set(answers):
                continue
            if any(not response or response.startswith(("[ERROR]", "[API_ERROR]", "[FUTURE_ERROR]"))
                   for identity, response in by_id.items() if identity not in errors):
                continue
            return len(errors), directory
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return None


def pending(target_id, eval_root, accept_amber_error_rows=False):
    if target_id == INTERNVL_VISION:
        directories = internvl_base_dirs(eval_root)
        return tuple(bench for bench, expected in VISION_EXPECTED.items()
                     if inspect_vision_all(directories, bench, expected).state != "已推理有结果")
    directories = target_dirs(TARGET_BY_ID[target_id], eval_root)
    missing = []
    for bench, expected in (("pope", EXPECTED_POPE), ("amber", EXPECTED_AMBER)):
        if inspect_all(directories, bench, expected).state == "已推理有结果":
            continue
        if bench == "amber" and accept_amber_error_rows and legacy_amber_errors(directories) is not None:
            continue
        missing.append(bench)
    return tuple(missing)


def provisional_amber(target_id, eval_root, enabled):
    if not enabled or target_id == INTERNVL_VISION:
        return None
    directories = target_dirs(TARGET_BY_ID[target_id], eval_root)
    if inspect_all(directories, "amber", EXPECTED_AMBER).state == "已推理有结果":
        return None
    return legacy_amber_errors(directories)


def model_path(target_id, model_root):
    if target_id == INTERNVL_VISION:
        return model_root / "internvl/InternVL3-2B-hf"
    target = TARGET_BY_ID[target_id]
    return model_root / "qwen" / f"Qwen3VL-{target.size}-Instruct"


def checkpoint_path(target_id, oss_root):
    target = TARGET_BY_ID[target_id]
    if target.step is None:
        return None
    spec = SPECS[(target.size, target.method)]
    return oss_root / spec.oss_name / f"global_step_{target.step}"


def output_dir(target_id, eval_root):
    if target_id == INTERNVL_VISION:
        return (eval_root / "internvl3/full/InternVL3-2B-hf" / DATASET /
                "main_tr05_base_vision_official_v2")
    target = TARGET_BY_ID[target_id]
    spec = SPECS[(target.size, target.method)]
    experiment = spec.experiment if target.step is None else f"{spec.experiment}_global_step_{target.step}"
    return (eval_root / "instruct/full" / experiment / DATASET /
            "main_tr05_aux_official_v2")


def preflight(args, jobs):
    if not args.python.is_file() or "vision-opd" not in str(args.python):
        raise RuntimeError(f"Use vision-opd Python: {args.python}")
    if not shutil.which("tmux"):
        raise RuntimeError("tmux is required for sharded evaluation")
    gpus = [gpu.strip() for gpu in args.gpu_list.split(",")]
    if not 1 <= len(gpus) <= 8 or len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus):
        raise ValueError("--gpu-list must contain 1-8 distinct GPU indices")
    if not 1024 <= args.base_port <= 65465:
        raise ValueError("--base-port leaves insufficient room for shard ports")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be between zero and one")
    if args.gpu_headroom_gib < 0:
        raise ValueError("--gpu-headroom-gib must be nonnegative")
    if args.kv_cache_gib <= 0:
        raise ValueError("--kv-cache-gib must be positive")
    for required in (ROOT / "res-opd/data/train_5k.parquet", ROOT / "res-opd/data/test_1000.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    for target_id in jobs:
        modes = pending(target_id, args.eval_root, args.accept_amber_error_rows)
        path = model_path(target_id, args.model_root)
        ckpt = checkpoint_path(target_id, args.oss_root) if target_id in TARGET_BY_ID else None
        print(f"{target_id:23} {'SKIP' if not modes else ','.join(modes):20} "
              f"model={path} ckpt={ckpt or '--'} output={output_dir(target_id, args.eval_root)}", flush=True)
        provisional = provisional_amber(target_id, args.eval_root, args.accept_amber_error_rows)
        if provisional:
            print(f"AMBER_PROVISIONAL {target_id}: {provisional[0]}/{EXPECTED_AMBER} error rows; "
                  f"official metrics include error-derived responses; source={provisional[1]}", flush=True)
        if not modes:
            continue
        if not (path / "config.json").is_file() or not any((path / name).is_file() for name in (
                "model.safetensors", "model.safetensors.index.json",
                "pytorch_model.bin", "pytorch_model.bin.index.json")):
            raise FileNotFoundError(f"Base model metadata or weights missing: {path}")
        if ckpt is not None:
            for name in ("model.safetensors", "config.json"):
                file = ckpt / name
                if not file.is_file() or file.stat().st_size == 0:
                    raise FileNotFoundError(file)
        if "amber" in modes and not (args.data_root / "AMBER/data/query/query_all.json").is_file():
            raise FileNotFoundError(args.data_root / "AMBER/data/query/query_all.json")
        if "pope" in modes and not (args.data_root / "COCO/coco2017val").is_dir():
            raise FileNotFoundError(args.data_root / "COCO/coco2017val")
        for bench in ("mmstar", "cv-bench"):
            if bench in modes:
                source = args.data_root / ("mmstar.json" if bench == "mmstar" else "cv_bench.json")
                if not source.is_file():
                    raise FileNotFoundError(source)
    if any("amber" in pending(target_id, args.eval_root, args.accept_amber_error_rows)
           for target_id in jobs):
        subprocess.run([str(args.python), "-c", "import nltk; from nltk.stem import WordNetLemmatizer; "
                        "nltk.word_tokenize('A dog runs.'); nltk.pos_tag(['A','dog','runs']); "
                        "WordNetLemmatizer().lemmatize('dogs')"], check=True)
    if any(checkpoint_path(target_id, args.oss_root) is not None and
           pending(target_id, args.eval_root, args.accept_amber_error_rows)
           for target_id in jobs if target_id in TARGET_BY_ID):
        if not shutil.which("ossutil") or not args.stage_dir.is_dir():
            raise RuntimeError("ossutil and a writable --stage-dir are required for checkpoint staging")


@contextmanager
def staged_checkpoint(target_id, args):
    source = checkpoint_path(target_id, args.oss_root)
    if source is None:
        yield model_path(target_id, args.model_root)
        return
    weight = source / "model.safetensors"
    needed = weight.stat().st_size + 8 * 1024**3
    if shutil.disk_usage(args.stage_dir).free < needed:
        raise RuntimeError(f"Insufficient local staging space for {target_id}: need {needed / 1024**3:.1f} GiB")
    with tempfile.TemporaryDirectory(prefix=f"resopd_aux_{target_id.lower()}_", dir=args.stage_dir) as temporary:
        staged = Path(temporary)
        for file in source.iterdir():
            if file.is_file() and file.name != weight.name:
                shutil.copy2(file, staged / file.name)
        spec = SPECS[(TARGET_BY_ID[target_id].size, TARGET_BY_ID[target_id].method)]
        uri = f"{args.oss_uri.rstrip('/')}/{spec.oss_name}/global_step_{TARGET_BY_ID[target_id].step}/model.safetensors"
        print(f"STAGE {target_id}: {uri} -> {staged}", flush=True)
        subprocess.run(["ossutil", "cp", uri, str(staged / weight.name), "-f"], check=True)
        if (staged / weight.name).stat().st_size != weight.stat().st_size:
            raise RuntimeError(f"Staged checkpoint size differs from OSS mount: {target_id}")
        sidecar = staged / "model.safetensors.sha256"
        if sidecar.is_file():
            parts = sidecar.read_text(encoding="utf-8").split()
            expected = parts[0].lower() if parts else ""
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise RuntimeError(f"Malformed SHA256 sidecar: {sidecar}")
            digest = hashlib.sha256()
            with (staged / weight.name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise RuntimeError(f"Staged checkpoint SHA256 mismatch: {target_id}")
        yield staged


def select_gpus(args, target_id, candidates):
    utilization = args.gpu_memory_utilization
    memory = gpu_memory_mib(candidates)
    required = {gpu: int(total * utilization + args.gpu_headroom_gib * 1024)
                for gpu, (_, total) in memory.items()}
    selected = [gpu for gpu in candidates if memory[gpu][0] >= required[gpu]]
    skipped = {gpu: {"free_mib": memory[gpu][0], "required_mib": required[gpu]}
               for gpu in candidates if gpu not in selected}
    if not selected:
        raise RuntimeError(f"No GPU has enough free memory for {target_id}; no waiting. "
                           f"GPU shortfall={skipped}")
    ports = [args.base_port + index * 10 for index in range(len(selected))]
    busy = [port for port in ports if not port_is_free(port)]
    if busy:
        raise RuntimeError(f"Evaluation ports busy for {target_id}; no waiting or process cleanup: {busy}")
    print(f"START {target_id}: GPUs={','.join(selected)} shards={len(selected)} "
          f"ports={ports[0]}..{ports[-1]} utilization={utilization:.2f} "
          f"skipped={skipped}", flush=True)
    return selected, utilization


def startup_memory_failure(output, started):
    shard_root = output / ".shards"
    if not shard_root.is_dir():
        return False
    for summary in shard_root.glob("*/failure_summary.txt"):
        if summary.stat().st_mtime < started:
            continue
        content = summary.read_text(encoding="utf-8", errors="replace")
        if "Free memory on device" in content and "less than desired GPU memory utilization" in content:
            return True
    return False


def run_one(args, target_id, candidates):
    modes = pending(target_id, args.eval_root, args.accept_amber_error_rows)
    if not modes:
        provisional = provisional_amber(target_id, args.eval_root, args.accept_amber_error_rows)
        label = "AMBER_PROVISIONAL" if provisional else "complete"
        print(f"SKIP {target_id}: all requested benchmarks {label}", flush=True)
        return
    output = output_dir(target_id, args.eval_root)
    target = TARGET_BY_ID.get(target_id)
    spec = SPECS[(target.size, target.method)] if target else None
    experiment = spec.experiment if spec else "InternVL3-2B-hf"
    if target and target.step is not None:
        experiment += f"_global_step_{target.step}"
    with staged_checkpoint(target_id, args) as inference_model:
        gpus, utilization = select_gpus(args, target_id, candidates)
        command = ["bash", str(ROOT / "res-opd/scripts/eval_after_merge_sharded_8gpu.sh"),
                   str(inference_model), "instruct" if target else "internvl3", ",".join(modes)]
        for attempt in range(2):
            env = os.environ.copy()
            env.update({
                "PYTHON_BIN": str(args.python), "RES_OPD_DATA_ROOT": str(args.data_root),
                "BENCHMARK_DATA_DIR": str(args.data_root), "VISION_BENCHMARK_DATA_DIR": str(args.data_root),
                "AMBER_ROOT": str(args.data_root / "AMBER"), "DATASET_VERSION": "full",
                "MODEL_PROFILE": "qwen3vl_instruct" if target else "default",
                "RESULT_VERSION_TAG": "instruct" if target else "internvl3",
                "EXPERIMENT_NAME_OVERRIDE": experiment, "EVAL_OUTPUT_DIR": str(output),
                "DEGRADATION_MODE": "original", "STUDENT_RATIO": "1.0",
                "CHAIR_PROMPT": "Describe the image.", "POPE_SOURCE": "res-opd-test",
                "POPE_BENCHMARK": "pope_adv,pope_pop,pope_random", "POPE_MAX_SAMPLES": "0",
                "AMBER_EVAL_TYPE": "a", "AMBER_MAX_SAMPLES": "0",
                "VISION_BENCHMARK": "mmstar,cv-bench", "VISION_BENCHMARK_AUTO_DOWNLOAD": "False",
                "VISION_BENCHMARK_REFRESH_PROMPTS": "False", "RULE_ONLY_JUDGE": "True",
                "MCQ_EXTRACT_MODE": "official", "EVAL_SHARD_COUNT": str(len(gpus)), "GPU_LIST": ",".join(gpus),
                "VLLM_BASE_PORT": str(args.base_port), "VLLM_PORT_CLEANUP": "False",
                "VLLM_STARTUP_TIMEOUT": str(args.startup_timeout), "VLLM_ENFORCE_EAGER": "True",
                "VLLM_MAX_NUM_SEQS": "8", "VLLM_GPU_MEMORY_UTILIZATION": str(utilization),
                "VLLM_KV_CACHE_MEMORY_BYTES": str(args.kv_cache_gib * 1024**3),
                "CLEANUP_LOCAL_CKPT": "False",
            })
            print(f"RUN {target_id}: {' '.join(command)} -> {output}", flush=True)
            started = time.time()
            try:
                subprocess.run(command, cwd=ROOT, env=env, check=True)
                break
            except subprocess.CalledProcessError:
                if attempt or not startup_memory_failure(output, started):
                    raise
                retry_gpus, utilization = select_gpus(args, target_id, candidates)
                if not set(retry_gpus) < set(gpus):
                    raise
                print(f"RETRY {target_id}: GPU memory changed; "
                      f"shards {len(gpus)} -> {len(retry_gpus)}", flush=True)
                gpus = retry_gpus
    still_missing = pending(target_id, args.eval_root, args.accept_amber_error_rows)
    if still_missing:
        raise RuntimeError(f"{target_id} returned success but outputs are still incomplete: {still_missing}")
    provisional = provisional_amber(target_id, args.eval_root, args.accept_amber_error_rows)
    if provisional:
        print(f"AMBER_PROVISIONAL {target_id}: {provisional[0]}/{EXPECTED_AMBER} error rows; "
              "official metrics include error-derived responses", flush=True)
    else:
        print(f"PASS {target_id}: {','.join(modes)}", flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--machine", type=int, choices=QUEUES, required=True)
    result.add_argument("--python", type=Path, default=Path(os.environ.get(
        "PYTHON_BIN") or str(Path(os.environ.get("CONDA_PREFIX", "")) / "bin/python3")))
    result.add_argument("--eval-root", type=Path, default=ROOT / "res-opd/eval_results")
    result.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    result.add_argument("--oss-root", type=Path, default=DEFAULT_OSS_ROOT)
    result.add_argument("--oss-uri", default=OSS_URI)
    result.add_argument("--data-root", type=Path, default=Path("/data/oss_bucket_0/yanlin"))
    result.add_argument("--stage-dir", type=Path, default=Path("/tmp"))
    result.add_argument("--gpu-list", default="0,1,2,3,4,5,6,7")
    result.add_argument("--base-port", type=int)
    result.add_argument("--startup-timeout", type=int, default=2400)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    result.add_argument("--gpu-headroom-gib", type=float, default=6.0)
    result.add_argument("--kv-cache-gib", type=int, default=8,
                        help="Per-shard vLLM KV cache in GiB; default preserves prior runs")
    result.add_argument("--accept-amber-error-rows", action="store_true",
                        help="Accept scored legacy AMBER outputs with logged error rows as provisional")
    result.add_argument("--preflight-only", action="store_true")
    return result


def main():
    args = parser().parse_args()
    if args.base_port is None:
        args.base_port = 28100 + args.machine * 1000
    jobs = QUEUES[args.machine]
    print(f"MACHINE {args.machine}: {len(jobs)} serial jobs: {', '.join(jobs)}", flush=True)
    preflight(args, jobs)
    print("PREFLIGHT=PASS", flush=True)
    if args.preflight_only:
        return
    args.eval_root.mkdir(parents=True, exist_ok=True)
    with (args.eval_root / f".main_tr05_aux_machine{args.machine}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Machine queue {args.machine} is already running") from error
        failures = []
        for target_id in jobs:
            try:
                run_one(args, target_id, args.gpu_list.split(","))
            except Exception as error:
                print(f"FAIL {target_id}: {type(error).__name__}: {error}", flush=True)
                failures.append(target_id)
        if failures:
            raise RuntimeError(f"Machine {args.machine} finished with failed jobs: {', '.join(failures)}")
    provisional = [target_id for target_id in jobs if provisional_amber(
        target_id, args.eval_root, args.accept_amber_error_rows)]
    print(f"MACHINE_{args.machine}={'PROVISIONAL_AMBER' if provisional else 'PASS'} "
          f"provisional={','.join(provisional) or '--'}", flush=True)


if __name__ == "__main__":
    main()
