#!/usr/bin/env python3
"""Finish HALC CHAIR, then run POPE/AMBER in a separate presentation subdirectory."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


RES_OPD = Path(__file__).resolve().parents[1]
ROOT = RES_OPD.parent
DEFAULT_TEST_JSON = RES_OPD / "data/test_1000.json"
DEFAULT_DINO_CONFIG = (
    RES_OPD / "third_party/baselines/HALC/decoder_zoo/GroundingDINO"
    / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
PARAMETER_FLAGS = {
    "beam_size": "--beam-size",
    "context_window": "--context-window",
    "candidate_count": "--candidate-count",
    "expand_ratio": "--expand-ratio",
    "box_threshold": "--box-threshold",
    "contrast_weight": "--contrast-weight",
    "skip_rate": "--skip-rate",
}


def read_jsonl(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_original(old_root, samples, model_path, prompt):
    expected_ids = [int(sample["image_id"]) for sample in samples]
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("Test JSON has duplicate image IDs")
    expected_shards = {image_id: index % 8 for index, image_id in enumerate(expected_ids)}
    old_rows = {}
    source_files = {}
    reference_config = None
    for index in range(8):
        path = old_root / f"shard_{index:02d}" / "out" / "eval_results.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        source_files[str(path.resolve())] = file_digest(path)
        for row in read_jsonl(path):
            image_id = int(row["image_id"])
            if image_id in old_rows:
                raise RuntimeError(f"Duplicate original result for image {image_id}")
            if expected_shards.get(image_id) != index:
                raise RuntimeError(f"Image {image_id} is in the wrong original shard")
            config = row.get("halc_config") or {}
            if (config.get("parameters", {}).get("max_new_tokens") != 8192
                    or Path(config.get("model_path", "")).resolve() != model_path.resolve()
                    or config.get("prompt") != prompt
                    or config.get("shard_count") != 8
                    or config.get("shard_index") != index):
                raise RuntimeError(f"Original HALC configuration mismatch for image {image_id}")
            normalized_config = {key: value for key, value in config.items()
                                 if key != "shard_index"}
            if reference_config is None:
                reference_config = normalized_config
            elif normalized_config != reference_config:
                raise RuntimeError(f"Original HALC parameters differ for image {image_id}")
            if not str(row.get("generated_caption") or "").strip():
                raise RuntimeError(f"Empty original caption for image {image_id}")
            old_rows[image_id] = row
    if not old_rows:
        raise RuntimeError("No original HALC rows to reuse")
    return old_rows, source_files


def partition_missing(samples, old_rows, worker_count):
    missing = [sample for sample in samples if int(sample["image_id"]) not in old_rows]
    groups = [[] for _ in range(worker_count)]
    for index, sample in enumerate(missing):
        groups[index % worker_count].append(sample)
    return missing, groups


def ensure_not_running_on_this_host(old_root):
    result = subprocess.run(["ps", "-eo", "args"], text=True, capture_output=True, check=True)
    active = [line for line in result.stdout.splitlines()
              if "eval_halc_chair.py" in line and str(old_root) in line]
    if active:
        raise RuntimeError("Original HALC run is still active on this host; stop it before hybrid continuation")


def launch_missing(args, groups, gpu_list, output_dir, old_config):
    processes = []
    try:
        for index, (gpu, group) in enumerate(zip(gpu_list, groups)):
            if not group:
                continue
            worker = output_dir / ".new_shards" / f"worker_{index:02d}"
            worker.mkdir(parents=True, exist_ok=True)
            if worker_rows(index, group, output_dir, args.model_path, args.prompt,
                           args.new_max_tokens, old_config, require_complete=False) is not None:
                print(f"SKIP worker={index}: CHAIR rows already complete", flush=True)
                continue
            test_json = worker / "test.json"
            expected_test = json.dumps(group, ensure_ascii=False, indent=2) + "\n"
            if test_json.exists() and test_json.read_text(encoding="utf-8") != expected_test:
                raise RuntimeError(f"Existing worker has a different image subset: {test_json}")
            test_json.write_text(expected_test, encoding="utf-8")
            command = [
                str(args.python_bin), str(RES_OPD / "baselines/eval_halc_chair.py"),
                "--model-family", "qwen3vl", "--model-path", str(args.model_path),
                "--test-json", str(test_json), "--output-dir", str(worker / "out"),
                "--detector-config", str(args.detector_config),
                "--detector-weights", str(args.detector_weights),
                "--blip-model-path", str(args.blip_model_path),
                "--prompt", args.prompt, "--max-samples", "0",
                "--max-new-tokens", str(args.new_max_tokens),
                "--seed", str(old_config["seed"]),
                "--shard-count", "1", "--shard-index", "0",
                "--allow-unvalidated-full-run",
            ]
            for name, flag in PARAMETER_FLAGS.items():
                command.extend([flag, str(old_config["parameters"][name])])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_path = worker / "eval.log"
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((index, process, log_path))
            print(f"START worker={index} gpu={gpu} images={len(group)} log={log_path}", flush=True)
        failures = []
        for index, process, log_path in processes:
            code = process.wait()
            print(f"DONE worker={index} exit={code} log={log_path}", flush=True)
            if code:
                failures.append((index, code, log_path))
        if failures:
            raise RuntimeError(f"HALC continuation failed: {failures}; completed rows are kept for retry")
    except BaseException:
        for _, process, _ in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _, process, _ in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        raise


def worker_rows(index, group, output_dir, model_path, prompt, max_tokens, old_config,
                require_complete):
    path = output_dir / ".new_shards" / f"worker_{index:02d}" / "out/eval_results.jsonl"
    expected = {int(sample["image_id"]) for sample in group}
    actual = {}
    for row in read_jsonl(path):
        image_id = int(row["image_id"])
        if image_id in actual or image_id not in expected:
            raise RuntimeError(f"Duplicate or unexpected new image {image_id} in {path}")
        config = row.get("halc_config") or {}
        if (config.get("parameters", {}).get("max_new_tokens") != max_tokens
                or Path(config.get("model_path", "")).resolve() != model_path.resolve()
                or config.get("prompt") != prompt
                or config.get("seed") != old_config["seed"]
                or config.get("adapter_revision") != old_config["adapter_revision"]
                or config.get("source_commit") != old_config["source_commit"]
                or {key: value for key, value in config.get("parameters", {}).items()
                    if key != "max_new_tokens"}
                != {key: value for key, value in old_config["parameters"].items()
                    if key != "max_new_tokens"}):
            raise RuntimeError(f"New HALC configuration mismatch for image {image_id}")
        if not str(row.get("generated_caption") or "").strip():
            raise RuntimeError(f"Empty new caption for image {image_id}")
        actual[image_id] = row
    if set(actual) != expected:
        if require_complete:
            raise RuntimeError(f"New worker {index} incomplete: {len(actual)}/{len(expected)}")
        return None
    return actual


def collect_new_rows(groups, output_dir, model_path, prompt, max_tokens, old_config):
    new_rows = {}
    for index, group in enumerate(groups):
        if group:
            new_rows.update(worker_rows(index, group, output_dir, model_path, prompt,
                                        max_tokens, old_config, require_complete=True))
    return new_rows


def build_mixed_rows(samples, old_rows, new_rows, new_max_tokens):
    expected = {int(sample["image_id"]) for sample in samples}
    if set(old_rows) & set(new_rows) or set(old_rows) | set(new_rows) != expected:
        raise RuntimeError("Hybrid CHAIR rows are overlapping or incomplete")
    rows = []
    for sample in samples:
        image_id = int(sample["image_id"])
        row = dict(old_rows.get(image_id) or new_rows[image_id])
        old = image_id in old_rows
        row["presentation_only"] = True
        row["presentation_origin"] = "original_8192" if old else f"continuation_{new_max_tokens}"
        rows.append(row)
    return rows


def presentation_output_dir(old_root, new_max_tokens):
    official_dir = old_root.parent.parent
    if (old_root.parent.name != ".shards"
            or official_dir.name != "chair_external_halc_qwen3vl_adapted_official"
            or "eval_results" not in official_dir.parts):
        raise ValueError("--old-shard-root must be a HALC CHAIR run inside eval_results")
    return official_dir / f"presentation_mixed_8192_{new_max_tokens}"


def build_aux_config(old_config, new_max_tokens, gpu_count, old_count, new_count):
    config = dict(old_config)
    config["parameters"] = dict(old_config["parameters"])
    config["parameters"]["max_new_tokens"] = new_max_tokens
    config["shard_count"] = gpu_count
    config["shard_index"] = 0
    config["presentation_only"] = True
    config["not_for_paper"] = True
    config["chair_decode_caps"] = {"8192": old_count, str(new_max_tokens): new_count}
    config["aux_decode_max_new_tokens"] = new_max_tokens
    return config


def run_aux(args, output_dir):
    command = [
        str(args.python_bin), str(RES_OPD / "baselines/run_external_aux_benchmarks.py"),
        "--model-path", str(args.model_path), "--methods", "halc",
        "--benchmarks", "pope,amber", "--halc-dir", str(output_dir),
        "--detector-config", str(args.detector_config),
        "--gpu-list", args.gpu_list, "--base-port", str(args.aux_base_port),
        "--min-free-mib", str(args.aux_min_free_mib),
        "--wait-seconds", str(args.aux_wait_seconds),
        "--startup-timeout", str(args.aux_startup_timeout),
    ]
    print(f"HALC_PRESENTATION_AUX=START output={output_dir}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"HALC_PRESENTATION_AUX=PASS output={output_dir}", flush=True)


def interrupted(_signum, _frame):
    raise KeyboardInterrupt("HALC presentation continuation interrupted")


def main():
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--old-shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--detector-config", type=Path, default=DEFAULT_DINO_CONFIG)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--blip-model-path", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-list", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--prompt", default="Describe the image.")
    parser.add_argument("--new-max-tokens", type=int, default=2048)
    parser.add_argument("--aux-base-port", type=int, default=24100)
    parser.add_argument("--aux-min-free-mib", type=int, default=36000)
    parser.add_argument("--aux-wait-seconds", type=int, default=7200)
    parser.add_argument("--aux-startup-timeout", type=int, default=1200)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--confirm-original-stopped", action="store_true")
    args = parser.parse_args()
    old_root = args.old_shard_root.resolve()
    if not 2048 <= args.new_max_tokens < 8192:
        raise ValueError("New token cap must be between 2048 and 8191")
    expected_output = presentation_output_dir(old_root, args.new_max_tokens)
    output_dir = args.output_dir.resolve() if args.output_dir else expected_output
    if output_dir != expected_output:
        raise ValueError(f"Hybrid results must be kept in the separate directory {expected_output}")
    gpu_list = [part.strip() for part in args.gpu_list.split(",")]
    if not gpu_list or len(gpu_list) != len(set(gpu_list)) or any(not x.isdigit() for x in gpu_list):
        raise ValueError("--gpu-list needs distinct GPU indices")
    samples = json.loads(args.test_json.read_text(encoding="utf-8"))
    if len(samples) != 1000:
        raise RuntimeError(f"Expected 1000 CHAIR test images, got {len(samples)}")
    old_rows, source_files = inspect_original(old_root, samples, args.model_path, args.prompt)
    old_config = next(iter(old_rows.values()))["halc_config"]
    for label, path, is_dir in (
        ("model", args.model_path, True),
        ("GroundingDINO config", args.detector_config, False),
        ("GroundingDINO weights", args.detector_weights, False),
        ("BLIP", args.blip_model_path, True),
        ("Python", args.python_bin, False),
    ):
        if not (path.is_dir() if is_dir else path.is_file()):
            raise FileNotFoundError(f"{label} is missing: {path}")
    if (Path(old_config["detector_weights"]).resolve() != args.detector_weights.resolve()
            or Path(old_config["blip_model_path"]).resolve() != args.blip_model_path.resolve()):
        raise RuntimeError("GroundingDINO or BLIP weights differ from the original run")
    missing, groups = partition_missing(samples, old_rows, len(gpu_list))
    print(f"PRESENTATION_ONLY old={len(old_rows)} missing={len(missing)} "
          f"old_cap=8192 new_cap={args.new_max_tokens} workers={len(gpu_list)}", flush=True)
    if args.preflight_only:
        return
    if not args.confirm_original_stopped:
        raise RuntimeError("Pass --confirm-original-stopped after stopping the old run on its host")
    ensure_not_running_on_this_host(old_root)
    metadata = {
        "presentation_only": True,
        "not_for_paper": True,
        "old_shard_root": str(old_root),
        "old_source_sha256": source_files,
        "old_count": len(old_rows),
        "old_decoder_config": {key: value for key, value in old_config.items()
                               if key != "shard_index"},
        "new_max_tokens": args.new_max_tokens,
        "model_path": str(args.model_path.resolve()),
        "test_json_sha256": file_digest(args.test_json),
        "gpu_count": len(gpu_list),
        "warning": (f"Mixed 8192/{args.new_max_tokens} token caps; exploratory only, "
                    "not a standard HALC benchmark."),
    }
    metadata_path = output_dir / "presentation_manifest.json"
    if metadata_path.is_file() and json.loads(metadata_path.read_text()) != metadata:
        raise RuntimeError(f"Existing presentation run has different inputs: {metadata_path}")
    write_json(metadata_path, metadata)
    if missing:
        launch_missing(args, groups, gpu_list, output_dir, old_config)
    if {path: file_digest(Path(path)) for path in source_files} != source_files:
        raise RuntimeError("Original HALC shards changed during continuation; refusing to mix")
    new_rows = collect_new_rows(groups, output_dir, args.model_path, args.prompt,
                                args.new_max_tokens, old_config)
    rows = build_mixed_rows(samples, old_rows, new_rows, args.new_max_tokens)
    stage = output_dir / ".merge_shards" / "shard_00" / "out"
    write_jsonl(stage / "eval_results.jsonl", rows)
    subprocess.run([
        str(args.python_bin), str(RES_OPD / "eval/merge_sharded_eval.py"),
        "--final-output-dir", str(output_dir),
        "--shard-root", str(stage.parent.parent), "--eval-mode", "chair",
        "--experiment-name", f"presentation-halc-hybrid-{args.model_path.name}",
        "--seed", "42",
    ], cwd=ROOT, check=True)
    metrics_path = output_dir / "chair_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("total_results") != 1000 or metrics.get("num_samples") != 1000:
        raise RuntimeError("Merged CHAIR result is not 1000/1000")
    metrics.update({"presentation_only": True, "not_for_paper": True,
                    "old_8192_rows": len(old_rows), "new_capped_rows": len(new_rows),
                    "mixed_decode_max_new_tokens": [8192, args.new_max_tokens]})
    write_json(metrics_path, metrics)
    aux_config_path = output_dir / "halc_config.json"
    aux_config = build_aux_config(old_config, args.new_max_tokens, len(gpu_list),
                                  len(old_rows), len(new_rows))
    if aux_config_path.is_file() and json.loads(aux_config_path.read_text()) != aux_config:
        raise RuntimeError(f"Existing auxiliary decoder configuration differs: {aux_config_path}")
    write_json(aux_config_path, aux_config)
    print(f"HALC_PRESENTATION_HYBRID=PASS rows=1000 old={len(old_rows)} "
          f"new={len(new_rows)} output={output_dir}", flush=True)
    run_aux(args, output_dir)


if __name__ == "__main__":
    main()
