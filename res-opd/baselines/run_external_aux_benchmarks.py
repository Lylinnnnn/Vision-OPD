#!/usr/bin/env python3
"""Evaluate external decoders on POPE, AMBER, MMStar, and CV-Bench."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
RES_OPD = ROOT / "res-opd"
sys.path.insert(0, str(RES_OPD / "baselines"))
sys.path.insert(0, str(RES_OPD / "scripts/tmp"))
from wait_eval_capacity import wait_for_capacity  # noqa: E402


METHOD_SUFFIX = {
    "vcd": "chair_external_vcd_greedy_official",
    "opera": "chair_external_opera_official",
    "halc": "chair_external_halc_qwen3vl_adapted_official",
}
POPE_SPLITS = ("pope_adv", "pope_pop", "pope_random")
VISION_COUNTS = {"mmstar": 1500, "cv-bench": 2638}
BENCHMARKS = ("pope", "amber", "mmstar", "cv-bench")
MODEL_FAMILIES = {
    **{f"Qwen3VL-{size}-Instruct": "qwen3vl" for size in ("2B", "4B", "8B")},
    "InternVL3-2B-hf": "internvl3",
}


def parse_benchmarks(value):
    names = tuple(item.strip() for item in value.split(","))
    if not names or len(names) != len(set(names)) or any(name not in BENCHMARKS for name in names):
        raise ValueError(f"--benchmarks must be distinct names from {', '.join(BENCHMARKS)}")
    return names


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def count_jsonl(path, answer_field=None):
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if answer_field and any(
        not str(row.get(answer_field, "")).strip() or
        str(row[answer_field]).startswith(("[ERROR]", "[API_ERROR]", "[FUTURE_ERROR]"))
        for row in rows
    ):
        raise RuntimeError(f"Invalid answer in {path}")
    return len(rows)


def result_dir(args, method):
    explicit = getattr(args, f"{method}_dir")
    if explicit:
        return explicit
    preferred = args.results_root / METHOD_SUFFIX[method]
    if (preferred / f"{method}_config.json").is_file():
        return preferred
    candidates = sorted(path for path in args.results_root.glob("*")
                        if (path / f"{method}_config.json").is_file())
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        f"Cannot uniquely locate {method} CHAIR result under {args.results_root}; "
        f"candidates={candidates}. Pass --{method}-dir explicitly."
    )


def stop_owned(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def validate_chair_anchor(directory, method, model_path):
    config = read_json(directory / f"{method}_config.json")
    chair = read_json(directory / "chair_metrics.json")
    if (chair.get("metric_schema") != "official_chair_caption_and_mention_v1"
            or chair.get("total_results") != 1000
            or chair.get("eval_prompt") != "Describe the image."
            or count_jsonl(directory / "eval_results.jsonl", "generated_caption") != 1000):
        raise RuntimeError(f"{method} CHAIR anchor is not a complete official 1000-image result: {directory}")
    if method in ("vcd", "opera") and config.get("model_name") != model_path.name:
        raise RuntimeError(f"{method} CHAIR anchor model mismatch: {directory}")
    if method == "halc" and Path(config.get("model_path", "")).resolve() != model_path.resolve():
        raise RuntimeError(f"HALC CHAIR anchor model mismatch: {directory}")
    max_tokens = config.get("max_new_tokens") if method != "halc" else config["parameters"]["max_new_tokens"]
    if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 8192:
        raise RuntimeError(f"Invalid {method} max_new_tokens in CHAIR config")
    return config, max_tokens


def standalone_config(args, method, existing, shard_count):
    model_family = getattr(args, "model_family", None) or MODEL_FAMILIES.get(args.model_path.name)
    if model_family is None:
        raise ValueError(f"Unsupported model: {args.model_path.name}")
    if method == "opera":
        from eval_opera_chair import opera_config

        return opera_config(SimpleNamespace(
            model_path=str(args.model_path), model_family=model_family,
            prompt="Describe the image.", max_new_tokens=args.max_new_tokens,
            num_beams=5, scale_factor=50.0, threshold=15,
            num_attn_candidates=5, penalty_weights=1.0, window_size=512,
            torch_dtype="bfloat16",
        ))
    if method != "halc":
        raise ValueError(f"Standalone mode is not available for {method}")

    from eval_halc_chair import ADAPTER_REVISION
    from halc_qwen3vl import HALCConfig

    detector_weights = args.detector_weights or (existing or {}).get("detector_weights")
    blip_model = args.blip_model_path or (existing or {}).get("blip_model_path")
    if not detector_weights or not blip_model:
        raise ValueError("Standalone HALC requires --detector-weights and --blip-model-path")
    parameters = HALCConfig(max_new_tokens=args.max_new_tokens)
    parameters.validate()
    registry = read_json(RES_OPD / "baselines/registry.json")
    config = {
        "method": f"halc_{model_family}_adaptation",
        "adapter_revision": ADAPTER_REVISION,
        "source_commit": registry["baselines"]["halc"]["commit"],
        "model_path": str(args.model_path.resolve()),
        "detector_weights": str(Path(detector_weights).resolve()),
        "blip_model_path": str(Path(blip_model).resolve()),
        "prompt": "Describe the image.",
        "seed": 42,
        "parameters": vars(parameters),
        "shard_count": shard_count,
        "shard_index": 0,
    }
    if model_family != "qwen3vl":
        config["model_family"] = model_family
    return config


def validate_inputs(args, methods, benchmarks=BENCHMARKS, shard_count=4):
    standalone = getattr(args, "standalone", False)
    if args.model_path.name not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported model: {args.model_path.name}")
    expected_family = MODEL_FAMILIES[args.model_path.name]
    model_family = getattr(args, "model_family", None) or expected_family
    if model_family != expected_family:
        raise ValueError(
            f"Model family mismatch: {args.model_path.name} requires {expected_family}, "
            f"got {args.model_family}"
        )
    args.model_family = model_family
    if not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(args.model_path / "config.json")
    if not any((args.model_path / name).is_file() for name in (
            "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json")):
        raise FileNotFoundError(f"Model weights not found under {args.model_path}")
    required = []
    if "pope" in benchmarks:
        required.append(RES_OPD / "data/test_1000.json")
    if "amber" in benchmarks:
        required.append(args.data_root / "AMBER/data/query/query_all.json")
    if "mmstar" in benchmarks:
        required.append(args.data_root / "mmstar.json")
    if "cv-bench" in benchmarks:
        required.append(args.data_root / "cv_bench.json")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if "pope" in benchmarks and not (args.data_root / "COCO/coco2017val").is_dir():
        raise FileNotFoundError(args.data_root / "COCO/coco2017val")
    if "amber" in benchmarks:
        import nltk
        from nltk.stem import WordNetLemmatizer
        nltk.word_tokenize("A dog runs.")
        nltk.pos_tag(["A", "dog", "runs"])
        WordNetLemmatizer().lemmatize("dogs")
    if "halc" in methods and not args.detector_config.is_file():
        raise FileNotFoundError(args.detector_config)
    configs = {}
    for method in methods:
        if standalone:
            directory = getattr(args, f"{method}_dir") or args.results_root / METHOD_SUFFIX[method]
            config_path = directory / f"{method}_config.json"
            existing = read_json(config_path) if config_path.is_file() else None
            config = standalone_config(args, method, existing, shard_count)
            if existing is not None and existing != config:
                raise RuntimeError(f"Refusing to overwrite different decoder settings: {config_path}")
            max_tokens = config.get("max_new_tokens") if method == "opera" else config["parameters"]["max_new_tokens"]
        else:
            directory = result_dir(args, method)
            config, max_tokens = validate_chair_anchor(directory, method, args.model_path)
        if method == "halc":
            for key in ("detector_weights", "blip_model_path"):
                if not Path(config[key]).exists():
                    raise FileNotFoundError(config[key])
        if standalone and not args.preflight_only and existing is None:
            directory.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        configs[method] = (directory, config, max_tokens)
        label = "DECODER_CONFIG" if standalone else "ANCHOR"
        print(f"{label}=PASS method={method} max_tokens={max_tokens} dir={directory}", flush=True)
    return configs


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def start_servers(args, method, directory, gpus, max_tokens, shard_root):
    processes = []
    model_id = f"external-{method}-{args.model_path.name}"
    try:
        for index, gpu in enumerate(gpus):
            port = args.base_port + index * 10
            shard = shard_root / f"shard_{index:02d}"
            (shard / "out").mkdir(parents=True, exist_ok=True)
            log_path = shard / "server.log"
            log = log_path.open("a", encoding="utf-8")
            command = [sys.executable, str(RES_OPD / "baselines/external_decode_api.py"),
                       "--method", method, "--model-path", str(args.model_path),
                       "--config", str(directory / f"{method}_config.json"),
                       "--model-id", model_id, "--port", str(port),
                       "--model-family", args.model_family]
            if method == "halc":
                command += ["--detector-config", str(args.detector_config)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            log.close()
            processes.append(process)
            print(f"SERVER method={method} shard={index}/{len(gpus)} gpu={gpu} port={port} log={log_path}", flush=True)
        deadline = time.monotonic() + args.startup_timeout
        while True:
            failed = [process.pid for process in processes if process.poll() is not None]
            if failed:
                raise RuntimeError(f"{method} decoder server exited during startup: {failed}; inspect server.log")
            if all(health(args.base_port + index * 10) for index in range(len(gpus))):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{method} decoder servers did not start; inspect server.log")
            time.sleep(5)
        return processes, model_id
    except BaseException:
        stop_servers(processes)
        raise


def stop_servers(processes):
    stop_owned(processes)


def evaluate_shard(args, method, shard_root, index, count, model_id, max_tokens,
                   active_processes, active_lock, benchmarks):
    shard = shard_root / f"shard_{index:02d}"
    output = shard / "out"
    port = args.base_port + index * 10
    api = f"http://127.0.0.1:{port}/v1/"
    log_path = shard / "eval.log"
    commands = []
    if "pope" in benchmarks:
        commands.append([sys.executable, str(RES_OPD / "eval/eval_pope.py"),
                         "--api-base", api, "--model-name", model_id,
                         "--test-json", str(RES_OPD / "data/test_1000.json"),
                         "--benchmark", ",".join(POPE_SPLITS), "--pope-source", "res-opd-test",
                         "--output-dir", str(output), "--parallel-workers", "1",
                         "--max-new-tokens", str(max_tokens), "--shard-count", str(count),
                         "--shard-index", str(index), "--max-retries", "1"])
    if "amber" in benchmarks:
        commands.append([sys.executable, str(RES_OPD / "eval/eval_amber.py"),
                         "--api-base", api, "--model-name", model_id,
                         "--amber-root", str(args.data_root / "AMBER"),
                         "--output-dir", str(output), "--evaluation-type", "a",
                         "--parallel-workers", "1", "--max-retries", "1",
                         "--max-new-tokens-generative", str(max_tokens),
                         "--max-new-tokens-discriminative", str(max_tokens),
                         "--shard-count", str(count), "--shard-index", str(index),
                         "--skip-official-eval"])
    vision_benches = [bench for bench in VISION_COUNTS if bench in benchmarks]
    if vision_benches:
        commands.append(["bash", str(ROOT / "eval/run_eval.sh")])
    env = os.environ.copy()
    env.update({
        "PYTHON_BIN": sys.executable, "RES_OPD_DATA_ROOT": str(args.data_root),
        "BENCHMARK_DATA_DIR": str(args.data_root),
        "BENCHMARK_PREPARE_DATA": "False", "BENCHMARK_AUTO_DOWNLOAD": "False",
        "BENCHMARK_REFRESH_PROMPTS": "False", "BENCHMARK_SKIP_JUDGE": "True",
        "BENCHMARK_OUTPUT_SUFFIX": "", "SEED": "42",
        "API_BASE": api, "OPENAI_MODEL_ID": model_id, "MODEL_NAME": model_id,
        "BENCHMARK": ",".join(vision_benches), "BENCHMARK_OUTPUT_LAYOUT": "per_benchmark",
        "OUT_DIR": str(output), "JUDGE_DIR": str(output),
        "RULE_ONLY_JUDGE": "True", "MCQ_EXTRACT_MODE": "official",
        "MAX_TOKENS": str(max_tokens), "MAX_RETRIES": "1", "PARALLEL_WORKERS": "1",
        "BENCHMARK_SHARD_COUNT": str(count), "BENCHMARK_SHARD_INDEX": str(index),
    })
    with log_path.open("a", encoding="utf-8") as log:
        for command in commands:
            print(f"RUN method={method} shard={index}: {' '.join(command)}", flush=True)
            log.write("RUN: " + " ".join(command) + "\n")
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            with active_lock:
                active_processes.add(process)
            try:
                code = process.wait()
            finally:
                with active_lock:
                    active_processes.discard(process)
            if code != 0:
                raise RuntimeError(f"{method} shard {index} failed with exit={code}; see {log_path}")
    return index


def validate_merged(directory, model_id, benchmarks=BENCHMARKS):
    if "pope" in benchmarks:
        pope = read_json(directory / "pope/pope_summary.json")
        for split in POPE_SPLITS:
            if split not in pope.get("benchmarks", {}) or count_jsonl(
                    directory / "pope" / split / "eval_results.jsonl", "model_answer") != 5223:
                raise RuntimeError(f"Incomplete POPE {split}: {directory}")
    if "amber" in benchmarks:
        amber = read_json(directory / "amber/amber_metrics.json")
        if amber.get("official_returncode") != 0 or amber.get("evaluation_type") != "a":
            raise RuntimeError(f"Incomplete official AMBER: {directory}")
        if count_jsonl(directory / "amber/raw_results.jsonl", "model_answer") != 15220:
            raise RuntimeError(f"Incomplete AMBER raw results: {directory}")
    for bench, expected in VISION_COUNTS.items():
        if bench not in benchmarks:
            continue
        answer = directory / bench / "model_answer" / f"{model_id}_seed42_answer.jsonl"
        judge = directory / bench / "judge" / f"{model_id}_seed42_answer.jsonl"
        judged = read_json(judge)
        if count_jsonl(answer, "model_answer") != expected or not isinstance(judged, list) or len(judged) != expected:
            raise RuntimeError(f"Incomplete {bench}: {directory}")
        if any(str(row.get("judge", "")).strip().lower() not in ("yes", "no") for row in judged):
            raise RuntimeError(f"Invalid judge output for {bench}: {judge}")


def run_method(args, method, directory, max_tokens, gpus, benchmarks):
    model_id = f"external-{method}-{args.model_path.name}"
    try:
        validate_merged(directory, model_id, benchmarks)
    except (OSError, ValueError, RuntimeError, KeyError):
        pass
    else:
        print(f"SKIP method={method}: {','.join(benchmarks)} complete", flush=True)
        return
    ports = [args.base_port + index * 10 for index in range(len(gpus))]
    wait_for_capacity(gpus, ports, args.min_free_mib, args.wait_seconds)
    shard_name = "external_aux_v1" if tuple(benchmarks) == BENCHMARKS else "external_aux_" + "_".join(benchmarks)
    if args.standalone:
        shard_name += "_standalone"
    shard_root = directory / ".aux_shards" / shard_name
    shard_root.mkdir(parents=True, exist_ok=True)
    lock_file = (shard_root / "run.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise RuntimeError(f"Auxiliary evaluation already running: {shard_root}") from error
    try:
        run_method_locked(args, method, directory, max_tokens, gpus, shard_root, benchmarks)
    finally:
        lock_file.close()


def run_method_locked(args, method, directory, max_tokens, gpus, shard_root, benchmarks):
    model_id = f"external-{method}-{args.model_path.name}"
    config_path = shard_root / "run_config.json"
    chair_config = directory / f"{method}_config.json"
    config_digest_key = "decoder_config_sha256" if args.standalone else "chair_config_sha256"
    run_config = {"method": method, "gpus": gpus, "model_id": model_id,
                  "max_tokens": max_tokens, "data_root": str(args.data_root),
                  "model_path": str(args.model_path.resolve()),
                  config_digest_key: hashlib.sha256(chair_config.read_bytes()).hexdigest()}
    if tuple(benchmarks) != BENCHMARKS:
        run_config["benchmarks"] = list(benchmarks)
    if config_path.is_file() and read_json(config_path) != run_config:
        raise RuntimeError(f"Existing shard run has a different configuration: {config_path}")
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    processes, model_id = start_servers(args, method, directory, gpus, max_tokens, shard_root)
    active_processes = set()
    active_lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=len(gpus))
    try:
        futures = [executor.submit(evaluate_shard, args, method, shard_root, index,
                                   len(gpus), model_id, max_tokens,
                                   active_processes, active_lock, benchmarks)
                   for index in range(len(gpus))]
        for future in futures:
            future.result()
    except BaseException:
        with active_lock:
            owned = list(active_processes)
        stop_owned(owned)
        raise
    finally:
        executor.shutdown(wait=True)
        stop_servers(processes)
    vision_benches = [bench for bench in VISION_COUNTS if bench in benchmarks]
    merge = [sys.executable, str(RES_OPD / "eval/merge_sharded_eval.py"),
             "--final-output-dir", str(directory), "--shard-root", str(shard_root),
             "--eval-mode", ",".join(benchmarks), "--pope-benchmark", ",".join(POPE_SPLITS),
             "--pope-source", "res-opd-test", "--vision-benchmark", ",".join(vision_benches),
             "--vision-benchmark-data-dir", str(args.data_root),
             "--experiment-name", model_id, "--seed", "42",
             "--amber-root", str(args.data_root / "AMBER"),
             "--amber-eval-type", "a", "--amber-official-eval-workers", "8",
             "--rule-only-judge", "--mcq-extract-mode", "official"]
    subprocess.run(merge, cwd=ROOT, check=True)
    validate_merged(directory, model_id, benchmarks)
    print(f"EXTERNAL_AUX=PASS method={method} dir={directory}", flush=True)


def main():
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("External auxiliary evaluation interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path,
                        default=Path("/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-2B-Instruct"))
    parser.add_argument("--model-family", choices=("qwen3vl", "internvl3"))
    parser.add_argument("--data-root", type=Path, default=Path("/data/oss_bucket_0/yanlin"))
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--vcd-dir", type=Path)
    parser.add_argument("--opera-dir", type=Path)
    parser.add_argument("--halc-dir", type=Path)
    parser.add_argument("--detector-config", type=Path,
                        default=RES_OPD / "third_party/baselines/HALC/decoder_zoo/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--methods", default="vcd,opera,halc")
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    parser.add_argument("--standalone", action="store_true",
                        help="Run without CHAIR output; pin decoder settings directly")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--detector-weights", type=Path)
    parser.add_argument("--blip-model-path", type=Path)
    parser.add_argument("--gpu-list", default="0,2,4,6")
    parser.add_argument("--base-port", type=int, default=24100)
    parser.add_argument("--min-free-mib", type=int, default=36000)
    parser.add_argument("--wait-seconds", type=int, default=7200)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.model_family = args.model_family or MODEL_FAMILIES.get(args.model_path.name)
    if args.results_root is None:
        args.results_root = (RES_OPD / "eval_results/external_baselines/full" /
                             args.model_path.name / "train5000_test1000_original_sr1p0")
    methods = [method.strip() for method in args.methods.split(",")]
    if not methods or len(methods) != len(set(methods)) or any(method not in METHOD_SUFFIX for method in methods):
        parser.error("--methods must be unique names from vcd,opera,halc")
    if args.standalone and any(method not in ("opera", "halc") for method in methods):
        parser.error("--standalone currently supports opera and halc")
    if not 1 <= args.max_new_tokens <= 8192:
        parser.error("--max-new-tokens must be between 1 and 8192")
    try:
        benchmarks = parse_benchmarks(args.benchmarks)
    except ValueError as error:
        parser.error(str(error))
    gpus = [gpu.strip() for gpu in args.gpu_list.split(",")]
    if not gpus or len(gpus) != len(set(gpus)) or any(not gpu.isdigit() for gpu in gpus):
        parser.error("--gpu-list must contain distinct GPU indices")
    if not 1024 <= args.base_port <= 65535 - 10 * (len(gpus) - 1):
        parser.error("Invalid --base-port")
    configs = validate_inputs(args, methods, benchmarks, len(gpus))
    if args.preflight_only:
        print("EXTERNAL_AUX_PREFLIGHT=PASS", flush=True)
        return
    for method in methods:
        directory, _, max_tokens = configs[method]
        run_method(args, method, directory, max_tokens, gpus, benchmarks)
    print("EXTERNAL_AUX_ALL=PASS", flush=True)


if __name__ == "__main__":
    main()
