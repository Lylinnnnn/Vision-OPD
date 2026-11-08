#!/usr/bin/env python3
"""Fill missing evaluations for the three paper control tables, two serial queues."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from run_paper_chair_official_batch import gpu_processes, listening_pids


ROOT = Path(__file__).resolve().parents[3]
PROMPT = "Describe the image."
SUFFIX = "three_tables_fill_official_v2"
DATASET = "train5000_test1000_original_sr1p0"
POPE_SPLITS = ("pope_adv", "pope_pop", "pope_random")


@dataclass(frozen=True)
class Task:
    table_id: str
    scale: str
    ratio: str
    rho: int
    step: int
    modes: tuple[str, ...]

    @property
    def experiment(self):
        batch = "b16" if self.scale == "8B" else "b32"
        return (f"Res-OPD-Qwen3VL-{self.scale}-Instruct-orig-sr1.0-tr{self.ratio}"
                f"-a1.0-frozen-rkl-riskmask-nll-p{self.rho}-{batch}-rn4-full5k-e1")

    @property
    def oss_name(self):
        model = "Qwen3VL_8B_Instruct_" if self.scale == "8B" else ""
        batch = "b16" if self.scale == "8B" else "b32"
        return (f"ResOPD_{model}orig_sr1.0_tr{self.ratio}_a1.0_frozen_rkl_"
                f"riskmask_nll_p{self.rho}_{batch}_rn4_full5k-e1")

    @property
    def key(self):
        return f"{self.table_id}/s{self.step}"

    def checkpoint(self, oss_mount):
        return (oss_mount / "yanlin/ckpt/OPD/v4" / self.oss_name /
                f"global_step_{self.step}")

    def dataset_dir(self):
        return (ROOT / "res-opd/eval_results/instruct/full" /
                f"{self.experiment}_global_step_{self.step}" / DATASET)

    def output(self, suffix):
        return self.dataset_dir() / suffix


FULL = ("chair", "pope", "amber", "mmstar", "cv-bench")
CHAIR_AUX = ("chair", "amber", "mmstar", "cv-bench")
CHAIR_POPE_AMBER = ("chair", "pope", "amber")
POPE_AMBER = ("pope", "amber")


def queue(worker):
    if worker == 0:
        tasks = [Task("T014", "8B", "0.25", 30, step, CHAIR_POPE_AMBER)
                 for step in (50, 100)]
        tasks += [Task("T031", "2B", "0.0", 30, 156, CHAIR_AUX)]
        tasks += [Task("T046", "2B", "0.5", 10, step, FULL)
                  for step in (50, 100, 150, 156)]
        tasks += [Task("T006", "2B", "0.5", 30, 150, POPE_AMBER),
                  Task("T017", "8B", "0.5", 30, 100, POPE_AMBER)]
    else:
        tasks = [Task("T014", "8B", "0.25", 30, step, CHAIR_POPE_AMBER)
                 for step in (150, 200)]
        tasks += [Task("T041", "2B", "1.0", 30, 156, CHAIR_AUX)]
        tasks += [Task("T047", "2B", "0.5", 20, step, FULL)
                  for step in (50, 100, 150, 156)]
        tasks += [Task("T012", "2B", "0.75", 30, 156, POPE_AMBER),
                  Task("T028", "8B", "0.75", 30, 300, POPE_AMBER)]
    return tasks


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def count_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def complete(output, modes):
    try:
        if "chair" in modes:
            chair = read_json(output / "chair_metrics.json")
            if not (chair.get("metric_schema") == "official_chair_caption_and_mention_v1"
                    and chair.get("eval_prompt") == PROMPT
                    and chair.get("total_results") == 1000
                    and chair.get("total_captions") == 1000
                    and chair.get("invalid_final_answer", 0) == 0
                    and count_jsonl(output / "eval_results.jsonl") == 1000):
                return False
        if "pope" in modes:
            pope = read_json(output / "pope/pope_summary.json")
            if not (isinstance(pope.get("macro_f1"), (int, float))
                    and all(split in pope.get("benchmarks", {}) for split in POPE_SPLITS)
                    and all(count_jsonl(output / "pope" / split / "eval_results.jsonl") == 5223
                            for split in POPE_SPLITS)):
                return False
        if "amber" in modes:
            amber = read_json(output / "amber/amber_metrics.json")
            values = amber.get("metrics") or {}
            if not (amber.get("official_returncode") == 0
                    and amber.get("evaluation_type") == "a"
                    and all(isinstance(values.get(key), (int, float))
                            for key in ("generative_task_chair", "generative_task_hal"))):
                return False
        for bench, expected in (("mmstar", 1500), ("cv-bench", 2638)):
            if bench in modes:
                paths = list((output / bench / "judge").glob("*_answer.jsonl"))
                if len(paths) != 1 or count_jsonl(paths[0]) != expected:
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def existing_official_chair(task):
    dataset = task.dataset_dir()
    for path in dataset.glob("*/chair_metrics.json"):
        try:
            metrics = read_json(path)
            if (metrics.get("metric_schema") == "official_chair_caption_and_mention_v1"
                    and metrics.get("eval_prompt") == PROMPT
                    and metrics.get("total_results") == 1000
                    and (path.parent / "eval_results.jsonl").is_file()):
                return path
        except (OSError, ValueError, TypeError):
            continue
    return None


def resource_state(gpus, ports):
    busy_ports = {port: sorted(pids) for port in ports if (pids := listening_pids(port))}
    busy_gpus = gpu_processes(gpus)
    return busy_ports, busy_gpus


def wait_for_release(gpus, ports, seconds):
    deadline = time.monotonic() + seconds
    while True:
        busy_ports, busy_gpus = resource_state(gpus, ports)
        if not busy_ports and not busy_gpus:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Previous evaluation has not released resources: ports={busy_ports}, GPUs={busy_gpus}")
        time.sleep(5)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--worker", type=int, choices=(0, 1), required=True)
    result.add_argument("--oss-mount", type=Path, default=Path("/data/oss_bucket_0"))
    result.add_argument("--data-root", type=Path, default=Path("/data/oss_bucket_0/yanlin"))
    result.add_argument("--output-suffix", default=SUFFIX)
    result.add_argument("--gpu-list", default="0,1,2,3,4,5,6,7")
    result.add_argument("--base-port", type=int, default=8600)
    result.add_argument("--transition-wait-seconds", type=int, default=300)
    result.add_argument("--preflight-only", action="store_true")
    return result


def preflight(args, tasks, python):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.output_suffix):
        raise ValueError("Unsafe output suffix")
    gpus = [value.strip() for value in args.gpu_list.split(",")]
    if len(gpus) != 8 or len(set(gpus)) != 8:
        raise ValueError("Exactly eight distinct GPU IDs are required")
    if not python.is_file() or not shutil.which("tmux"):
        raise RuntimeError("Activate vision-opd and ensure tmux is installed")
    if args.transition_wait_seconds < 1:
        raise ValueError("--transition-wait-seconds must be positive")
    required = (ROOT / "res-opd/data/test_1000.json",
                ROOT / "res-opd/data/train_5k.parquet",
                args.data_root / "AMBER",
                args.data_root / "COCO/coco2017val",
                args.data_root / "mmstar.json",
                args.data_root / "cv_bench.json")
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    nltk_check = subprocess.run(
        [str(python), "-c",
         "import nltk; from nltk.stem import WordNetLemmatizer; "
         "nltk.word_tokenize('A dog runs.'); "
         "nltk.pos_tag(['A', 'dog', 'runs']); "
         "WordNetLemmatizer().lemmatize('dogs')"],
        capture_output=True, text=True, check=False,
    )
    if nltk_check.returncode != 0:
        raise RuntimeError(f"AMBER NLTK preflight failed:\n{nltk_check.stderr}")
    for task in tasks:
        checkpoint = task.checkpoint(args.oss_mount)
        for filename in ("model.safetensors", "config.json"):
            path = checkpoint / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"{task.key}: missing {path}")
        if "chair" not in task.modes and not existing_official_chair(task):
            raise RuntimeError(f"{task.key}: matching official CHAIR output not found")
        output = task.output(args.output_suffix)
        status = "SKIP complete" if complete(output, task.modes) else "NEW" if not output.exists() else "BLOCK partial"
        print(f"{task.key} {status}: {checkpoint} -> {output}", flush=True)
        if status == "BLOCK partial":
            raise RuntimeError(f"Inspect existing partial output before rerun: {output}")
    ports = [args.base_port + index * 10 for index in range(8)]
    busy_ports, busy_gpus = resource_state(gpus, ports)
    if busy_ports or busy_gpus:
        raise RuntimeError(f"Machine is not idle: ports={busy_ports}, GPUs={busy_gpus}")
    return gpus, ports


def run_one(args, task, python):
    output = task.output(args.output_suffix)
    if complete(output, task.modes):
        print(f"SKIP {task.key}: complete", flush=True)
        return
    if output.exists():
        raise RuntimeError(f"Partial output requires inspection: {output}")
    free_gib = shutil.disk_usage(ROOT).free / (1024 ** 3)
    if free_gib < 20:
        raise RuntimeError(f"Shared workspace has only {free_gib:.1f} GiB free; pausing before {task.key}")
    env = os.environ.copy()
    env.update({
        "PYTHON_BIN": str(python),
        "RES_OPD_DATA_ROOT": str(args.data_root),
        "BENCHMARK_DATA_DIR": str(args.data_root),
        "VISION_BENCHMARK_DATA_DIR": str(args.data_root),
        "AMBER_ROOT": str(args.data_root / "AMBER"),
        "DATASET_VERSION": "full",
        "MODEL_PROFILE": "qwen3vl_instruct",
        "RESULT_VERSION_TAG": "instruct",
        "EXPERIMENT_NAME_OVERRIDE": task.experiment,
        "EVAL_OUTPUT_DIR": str(output),
        "DEGRADATION_MODE": "original",
        "STUDENT_RATIO": "1.0",
        "CHAIR_PROMPT": PROMPT,
        "CHAIR_MAX_SAMPLES": "1000",
        "POPE_BENCHMARK": ",".join(POPE_SPLITS),
        "POPE_SOURCE": "res-opd-test",
        "POPE_MAX_SAMPLES": "0",
        "AMBER_EVAL_TYPE": "a",
        "AMBER_MAX_SAMPLES": "0",
        "VISION_BENCHMARK": "mmstar,cv-bench",
        "VISION_BENCHMARK_AUTO_DOWNLOAD": "False",
        "VISION_BENCHMARK_REFRESH_PROMPTS": "False",
        "VISION_BENCHMARK_OUTPUT_SUFFIX": "",
        "RULE_ONLY_JUDGE": "True",
        "MCQ_EXTRACT_MODE": "official",
        "EVAL_SHARD_COUNT": "8",
        "GPU_LIST": args.gpu_list,
        "VLLM_BASE_PORT": str(args.base_port),
        "VLLM_PORT_CLEANUP": "False",
        "VLLM_STARTUP_TIMEOUT": "1800",
        "CLEANUP_LOCAL_CKPT": "False",
    })
    command = ["bash", str(ROOT / "res-opd/scripts/eval_after_merge_sharded_8gpu.sh"),
               str(task.checkpoint(args.oss_mount)), "instruct", ",".join(task.modes)]
    print(f"RUN {task.key}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if not complete(output, task.modes):
        raise RuntimeError(f"Evaluation returned success but output is incomplete: {output}")
    print(f"PASS {task.key}", flush=True)


def main():
    args = parser().parse_args()
    tasks = queue(args.worker)
    python = Path(os.environ.get("PYTHON_BIN") or
                  str(Path(os.environ.get("CONDA_PREFIX", "")) / "bin/python3"))
    print(f"Worker {args.worker}: {len(tasks)} serial evaluations", flush=True)
    gpus, ports = preflight(args, tasks, python)
    print("PREFLIGHT=PASS", flush=True)
    if args.preflight_only:
        return
    lock = ROOT / "res-opd/eval_results" / f".{args.output_suffix}.worker{args.worker}.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"This shared-filesystem worker queue is already running: {lock}") from error
    try:
        for task in tasks:
            wait_for_release(gpus, ports, args.transition_wait_seconds)
            run_one(args, task, python)
        print(f"WORKER_{args.worker}=PASS", flush=True)
    finally:
        lock.rmdir()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
