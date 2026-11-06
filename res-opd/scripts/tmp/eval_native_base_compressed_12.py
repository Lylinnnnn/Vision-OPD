#!/usr/bin/env python3
"""Run the 12 native compressed Base evaluations (CHAIR, POPE, AMBER)."""

import argparse
import json
import os
import subprocess
from pathlib import Path


MODELS = {
    "qwen2b": ("Qwen3VL-2B-Instruct", "qwen", "instruct", "qwen3vl_instruct"),
    "qwen4b": ("Qwen3VL-4B-Instruct", "qwen", "instruct", "qwen3vl_instruct"),
    "qwen8b": ("Qwen3VL-8B-Instruct", "qwen", "instruct", "qwen3vl_instruct"),
    "internvl2b": ("InternVL3-2B-hf", "internvl", "internvl3", "default"),
}
RATIOS = ("0.25", "0.5", "0.75")
POPE_SPLITS = ("pope_adv", "pope_pop", "pope_random")
CHAIR_SCHEMA = "official_chair_caption_and_mention_v1"
CHAIR_PROMPT = "Describe the image."


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", choices=(*MODELS, "all"), required=True)
    result.add_argument("--ratios", default=",".join(RATIOS))
    result.add_argument("--model-root", type=Path, default=Path("/home/liuyanlin.lyl/notebook/model"))
    result.add_argument("--data-root", type=Path, default=Path("/data/oss_bucket_0/yanlin"))
    result.add_argument("--output-suffix", default="native_base_official_v2")
    result.add_argument("--gpu-list", default="0,1,2,3,4,5,6,7")
    result.add_argument("--shard-count", type=int, default=8)
    result.add_argument("--base-port", type=int, default=9700)
    result.add_argument("--preflight-only", action="store_true")
    return result


def jobs(model, ratios):
    selected = MODELS if model == "all" else {model: MODELS[model]}
    return [(key, ratio) for key in selected for ratio in ratios]


def result_dir(repo, model_key, ratio, suffix):
    name, _, tag, _ = MODELS[model_key]
    return (repo / "res-opd/eval_results" / tag / "full" / name /
            f"train5000_test1000_original_sr{ratio.replace('.', 'p')}" / suffix)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def complete(output, ratio):
    try:
        chair = read_json(output / "chair_metrics.json")
        pope = read_json(output / "pope/pope_summary.json")
        amber = read_json(output / "amber/amber_metrics.json")
        with (output / "eval_results.jsonl").open(encoding="utf-8") as handle:
            first_record = json.loads(handle.readline())
        return (
            chair.get("metric_schema") == CHAIR_SCHEMA
            and chair.get("eval_prompt") == CHAIR_PROMPT
            and chair.get("num_samples") == 1000
            and chair.get("total_results") == 1000
            and first_record.get("eval_degradation_mode") == "original"
            and float(first_record.get("eval_student_ratio", -1)) == float(ratio)
            and first_record.get("eval_prompt") == CHAIR_PROMPT
            and all(split in pope.get("benchmarks", {}) for split in POPE_SPLITS)
            and pope.get("macro_f1") is not None
            and amber.get("official_returncode") == 0
            and amber.get("evaluation_type") == "a"
            and bool(amber.get("metrics"))
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def preflight(repo, args, selected):
    if args.shard_count < 1 or len(args.gpu_list.split(",")) < args.shard_count:
        raise ValueError("GPU list must contain at least shard-count devices")
    if not args.output_suffix or not all(ch.isalnum() or ch in "._-" for ch in args.output_suffix):
        raise ValueError("Unsafe output suffix")
    for path in (repo / "res-opd/data/train_5k.parquet", repo / "res-opd/data/test_1000.json",
                 repo / "res-opd/scripts/eval_after_merge_sharded_8gpu.sh", args.data_root / "AMBER"):
        if not path.exists():
            raise FileNotFoundError(path)
    for key, ratio in selected:
        name, subdir, _, _ = MODELS[key]
        model = args.model_root / subdir / name
        if not (model / "config.json").is_file():
            raise FileNotFoundError(model / "config.json")
        output = result_dir(repo, key, ratio, args.output_suffix)
        status = "SKIP complete" if complete(output, ratio) else "NEW" if not output.exists() else "BLOCK partial"
        print(f"{key} sr={ratio} {status}: model={model} output={output}", flush=True)
        if status == "BLOCK partial":
            raise RuntimeError(f"Existing incomplete output requires inspection before rerun: {output}")


def run_one(repo, args, key, ratio):
    name, subdir, tag, profile = MODELS[key]
    output = result_dir(repo, key, ratio, args.output_suffix)
    if complete(output, ratio):
        print(f"SKIP {key} sr={ratio}: complete", flush=True)
        return
    env = os.environ.copy()
    env.update({
        "PYTHON_BIN": env.get("PYTHON_BIN") or str(Path(env["CONDA_PREFIX"]) / "bin/python3"),
        "RES_OPD_DATA_ROOT": str(args.data_root),
        "BENCHMARK_DATA_DIR": str(args.data_root),
        "VISION_BENCHMARK_DATA_DIR": str(args.data_root),
        "AMBER_ROOT": str(args.data_root / "AMBER"),
        "DATASET_VERSION": "full", "MODEL_PROFILE": profile,
        "RESULT_VERSION_TAG": tag, "EVAL_OUTPUT_DIR": str(output),
        "DEGRADATION_MODE": "original", "STUDENT_RATIO": ratio,
        "CHAIR_PROMPT": CHAIR_PROMPT,
        "POPE_BENCHMARK": ",".join(POPE_SPLITS), "POPE_SOURCE": "res-opd-test",
        "AMBER_EVAL_TYPE": "a", "POPE_MAX_SAMPLES": "0", "AMBER_MAX_SAMPLES": "0",
        "EVAL_SHARD_COUNT": str(args.shard_count), "GPU_LIST": args.gpu_list,
        "VLLM_BASE_PORT": str(args.base_port), "VLLM_PORT_CLEANUP": "False",
        "VLLM_STARTUP_TIMEOUT": "1200", "CLEANUP_LOCAL_CKPT": "False",
    })
    model = args.model_root / subdir / name
    print(f"RUN {key} sr={ratio} -> {output}", flush=True)
    subprocess.run(["bash", str(repo / "res-opd/scripts/eval_after_merge_sharded_8gpu.sh"),
                    str(model), tag, "chair,pope,amber"], cwd=repo, env=env, check=True)
    if not complete(output, ratio):
        raise RuntimeError(f"Evaluation finished but official metrics are incomplete: {output}")
    print(f"PASS {key} sr={ratio}", flush=True)


def main():
    args = parser().parse_args()
    ratios = tuple(part.strip() for part in args.ratios.split(","))
    if not ratios or len(set(ratios)) != len(ratios) or any(ratio not in RATIOS for ratio in ratios):
        raise ValueError(f"Ratios must be distinct values from {RATIOS}")
    repo = Path(__file__).resolve().parents[3]
    selected = jobs(args.model, ratios)
    preflight(repo, args, selected)
    if args.preflight_only:
        return
    if "CONDA_PREFIX" not in os.environ and "PYTHON_BIN" not in os.environ:
        raise EnvironmentError("Activate vision-opd or set PYTHON_BIN")
    for key, ratio in selected:
        run_one(repo, args, key, ratio)


if __name__ == "__main__":
    main()
