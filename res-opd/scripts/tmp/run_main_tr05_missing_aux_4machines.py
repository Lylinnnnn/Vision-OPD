#!/usr/bin/env python3
"""Run missing main-table POPE/AMBER and InternVL Base vision jobs on four nodes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from check_main_tr05_pope_amber import (
    DATASET, EXPECTED_AMBER, EXPECTED_POPE, ROOT, TARGETS, VISION_EXPECTED,
    build_specs, inspect_all, inspect_vision_all, internvl_base_dirs, target_dirs,
)
from run_three_tables_fill import wait_for_capacity


INTERNVL_VISION = "InternVL3-Base-vision"
QUEUES = {
    0: ("M088", "Base-2B", "M021", "M032"),
    1: ("M005", "M006", "Base-4B", "M041", "M039"),
    2: ("M014", "M024", "M020", "Base-8B", "M055"),
    3: ("M018", "M025", "M028", "M064", INTERNVL_VISION),
}
TARGET_BY_ID = {target.row_id: target for target in TARGETS if target.method != "OPERA"}
SPECS = {(spec.size, spec.method): spec for spec in build_specs() if spec.family == "Qwen3VL"}
DEFAULT_MODEL_ROOT = Path("/home/liuyanlin.lyl/notebook/model")
DEFAULT_OSS_ROOT = Path("/data/oss_bucket_0/yanlin/ckpt/OPD/v4")
OSS_URI = "oss://industry-algo/yanlin/ckpt/OPD/v4"


def pending(target_id, eval_root):
    if target_id == INTERNVL_VISION:
        directories = internvl_base_dirs(eval_root)
        return tuple(bench for bench, expected in VISION_EXPECTED.items()
                     if inspect_vision_all(directories, bench, expected).state != "已推理有结果")
    directories = target_dirs(TARGET_BY_ID[target_id], eval_root)
    return tuple(bench for bench, expected in (("pope", EXPECTED_POPE), ("amber", EXPECTED_AMBER))
                 if inspect_all(directories, bench, expected).state != "已推理有结果")


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
    if target_id == "M088":
        return (halc_anchor(target_id, eval_root) or
                eval_root / "external_baselines/full/Qwen3VL-2B-Instruct" / DATASET /
                "chair_external_halc_qwen3vl_adapted_official")
    target = TARGET_BY_ID[target_id]
    spec = SPECS[(target.size, target.method)]
    experiment = spec.experiment if target.step is None else f"{spec.experiment}_global_step_{target.step}"
    return (eval_root / "instruct/full" / experiment / DATASET /
            "main_tr05_aux_official_v2")


def halc_anchor(target_id, eval_root):
    return next((directory for directory in target_dirs(TARGET_BY_ID[target_id], eval_root)
                 if (directory / "halc_config.json").is_file()), None)


def halc_command(args, modes, preflight_only=False):
    anchor = halc_anchor("M088", args.eval_root)
    if anchor is None:
        raise FileNotFoundError("HALC official CHAIR anchor/config not found; cannot reuse decoder settings")
    command = [str(args.python), str(ROOT / "res-opd/baselines/run_external_aux_benchmarks.py"),
               "--model-path", str(model_path("M088", args.model_root)), "--methods", "halc",
               "--benchmarks", ",".join(modes), "--halc-dir", str(anchor),
               "--gpu-list", args.gpu_list, "--base-port", str(args.base_port),
               "--data-root", str(args.data_root), "--min-free-mib", str(args.halc_min_free_mib),
               "--wait-seconds", str(args.wait_seconds)]
    return command + (["--preflight-only"] if preflight_only else [])


def preflight(args, jobs):
    if not args.python.is_file() or "vision-opd" not in str(args.python):
        raise RuntimeError(f"Use vision-opd Python: {args.python}")
    if not shutil.which("tmux"):
        raise RuntimeError("tmux is required for sharded evaluation")
    gpus = [gpu.strip() for gpu in args.gpu_list.split(",")]
    if len(gpus) != 8 or len(set(gpus)) != 8 or any(not gpu.isdigit() for gpu in gpus):
        raise ValueError("--gpu-list must contain 8 distinct GPU indices")
    if not 1024 <= args.base_port <= 65465:
        raise ValueError("--base-port leaves insufficient room for eight shard ports")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be between zero and one")
    for required in (ROOT / "res-opd/data/train_5k.parquet", ROOT / "res-opd/data/test_1000.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    for target_id in jobs:
        modes = pending(target_id, args.eval_root)
        path = model_path(target_id, args.model_root)
        ckpt = checkpoint_path(target_id, args.oss_root) if target_id in TARGET_BY_ID else None
        print(f"{target_id:23} {'SKIP' if not modes else ','.join(modes):20} "
              f"model={path} ckpt={ckpt or '--'} output={output_dir(target_id, args.eval_root)}", flush=True)
        if not modes:
            continue
        if not (path / "config.json").is_file() or not any((path / name).is_file() for name in (
                "model.safetensors", "model.safetensors.index.json",
                "pytorch_model.bin", "pytorch_model.bin.index.json")):
            raise FileNotFoundError(f"Base model metadata or weights missing: {path}")
        if target_id == "M088":
            subprocess.run(halc_command(args, modes, preflight_only=True), cwd=ROOT, check=True)
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
    if any("amber" in pending(target_id, args.eval_root) for target_id in jobs):
        subprocess.run([str(args.python), "-c", "import nltk; from nltk.stem import WordNetLemmatizer; "
                        "nltk.word_tokenize('A dog runs.'); nltk.pos_tag(['A','dog','runs']); "
                        "WordNetLemmatizer().lemmatize('dogs')"], check=True)
    if any(checkpoint_path(target_id, args.oss_root) is not None and pending(target_id, args.eval_root)
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


def run_one(args, target_id, gpus):
    modes = pending(target_id, args.eval_root)
    if not modes:
        print(f"SKIP {target_id}: all requested benchmarks complete", flush=True)
        return
    ports = [args.base_port + index * 10 for index in range(len(gpus))]
    print(f"WAIT {target_id}: GPUs={','.join(gpus)} ports={ports[0]}..{ports[-1]}", flush=True)
    wait_for_capacity(gpus, ports, args.wait_seconds, args.gpu_memory_utilization, 6.0)
    output = output_dir(target_id, args.eval_root)
    model = model_path(target_id, args.model_root)
    if target_id == "M088":
        command = halc_command(args, modes)
        print(f"RUN {target_id}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        target = TARGET_BY_ID.get(target_id)
        spec = SPECS[(target.size, target.method)] if target else None
        experiment = spec.experiment if spec else "InternVL3-2B-hf"
        if target and target.step is not None:
            experiment += f"_global_step_{target.step}"
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
            "MCQ_EXTRACT_MODE": "official", "EVAL_SHARD_COUNT": "8", "GPU_LIST": args.gpu_list,
            "VLLM_BASE_PORT": str(args.base_port), "VLLM_PORT_CLEANUP": "False",
            "VLLM_STARTUP_TIMEOUT": str(args.startup_timeout), "VLLM_ENFORCE_EAGER": "True",
            "VLLM_MAX_NUM_SEQS": "8", "VLLM_GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
            "VLLM_KV_CACHE_MEMORY_BYTES": str(8 * 1024**3), "CLEANUP_LOCAL_CKPT": "False",
        })
        with staged_checkpoint(target_id, args) as inference_model:
            command = ["bash", str(ROOT / "res-opd/scripts/eval_after_merge_sharded_8gpu.sh"),
                       str(inference_model), "instruct" if target else "internvl3", ",".join(modes)]
            print(f"RUN {target_id}: {' '.join(command)} -> {output}", flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    still_missing = pending(target_id, args.eval_root)
    if still_missing:
        raise RuntimeError(f"{target_id} returned success but outputs are still incomplete: {still_missing}")
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
    result.add_argument("--wait-seconds", type=int, default=43200)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    result.add_argument("--halc-min-free-mib", type=int, default=50000)
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
        for target_id in jobs:
            run_one(args, target_id, args.gpu_list.split(","))
    print(f"MACHINE_{args.machine}=PASS", flush=True)


if __name__ == "__main__":
    main()
