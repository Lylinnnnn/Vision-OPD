#!/usr/bin/env python3
"""Smoke-test a HALC API with small POPE and AMBER subsets before full evaluation."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


RES_OPD = Path(__file__).resolve().parents[1]
ROOT = RES_OPD.parent
BAD_PREFIXES = ("[ERROR]", "[API_ERROR]", "[FUTURE_ERROR]")
POPE_SPLITS = ("pope_adv", "pope_pop", "pope_random")


def read_jsonl(path):
    if not path.is_file():
        raise RuntimeError(f"Missing smoke output: {path}")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def check_answers(rows, expected, label, answer_key="model_answer"):
    if len(rows) != expected:
        raise RuntimeError(f"{label}: expected {expected} rows, got {len(rows)}")
    bad = [row for row in rows if not str(row.get(answer_key) or "").strip()
           or str(row[answer_key]).startswith(BAD_PREFIXES)]
    if bad:
        raise RuntimeError(f"{label}: {len(bad)} invalid answers; first={bad[0].get(answer_key)!r}")
    return rows


def check_pope(output_dir, expected):
    for split in POPE_SPLITS:
        rows = check_answers(
            read_jsonl(output_dir / "pope" / split / "eval_results.jsonl"),
            expected, f"POPE {split}",
        )
        if any(not row.get("sample_uid") for row in rows) or len({row["sample_uid"] for row in rows}) != expected:
            raise RuntimeError(f"POPE {split}: duplicate or missing sample IDs")
        if any(row.get("pred_answer") not in ("yes", "no") for row in rows):
            raise RuntimeError(f"POPE {split}: invalid yes/no extraction")
        print(f"POPE_SMOKE=PASS split={split} rows={len(rows)} "
              f"answers={[row['model_answer'][:80] for row in rows]}", flush=True)


def check_amber(output_dir, expected, evaluation_type, discriminative):
    rows = check_answers(
        read_jsonl(output_dir / "amber/raw_results.jsonl"),
        expected, f"AMBER {evaluation_type}",
    )
    if any(row.get("id") is None for row in rows) or len({row["id"] for row in rows}) != expected:
        raise RuntimeError(f"AMBER {evaluation_type}: duplicate or missing IDs")
    if any(row.get("is_discriminative") is not discriminative for row in rows):
        raise RuntimeError(f"AMBER {evaluation_type}: unexpected query type")
    responses = json.loads(
        (output_dir / "amber" / f"amber_{evaluation_type}_responses.json").read_text(encoding="utf-8")
    )
    if len(responses) != expected or any(
        not str(row.get("response") or "").strip()
        or str(row["response"]).startswith(BAD_PREFIXES)
        for row in responses
    ):
        raise RuntimeError(f"AMBER {evaluation_type}: invalid official-format responses")
    print(f"AMBER_SMOKE=PASS type={evaluation_type} rows={len(rows)} "
          f"answers={[row['model_answer'][:80] for row in rows]}", flush=True)


def port_is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def wait_for_server(process, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"HALC API exited with {process.returncode} before startup")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    raise TimeoutError(f"HALC API did not start within {timeout}s")


def stop_server(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_eval(label, command, output_dir, env):
    log_path = output_dir / f"{label}.log"
    print(f"RUN {label}: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{label} exited {result.returncode}; see {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--halc-dir", type=Path, required=True)
    parser.add_argument("--detector-config", type=Path, default=(
        RES_OPD / "third_party/baselines/HALC/decoder_zoo/GroundingDINO"
        / "groundingdino/config/GroundingDINO_SwinT_OGC.py"))
    parser.add_argument("--data-root", type=Path, default=Path("/data/oss_bucket_0/yanlin"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=29300)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()
    if args.samples < 1 or args.samples > 10:
        parser.error("--samples must be between 1 and 10")
    if not args.gpu.isdigit():
        parser.error("--gpu must be a single GPU index")
    if not port_is_free(args.port):
        parser.error(f"Port {args.port} is already in use")
    config_path = args.halc_dir / "halc_config.json"
    if not config_path.is_file():
        parser.error(f"Missing pinned HALC config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if Path(config.get("model_path", "")).resolve() != args.model_path.resolve():
        parser.error("HALC config model path does not match --model-path")
    model_weights = ("model.safetensors", "model.safetensors.index.json",
                     "pytorch_model.bin", "pytorch_model.bin.index.json")
    if not any((args.model_path / name).is_file() for name in model_weights):
        parser.error(f"Model weights missing under {args.model_path}")
    required = [args.model_path / "config.json", args.detector_config,
                Path(config["detector_weights"]),
                Path(config["blip_model_path"]) / "config.json",
                RES_OPD / "data/test_1000.json"]
    required.extend(args.data_root / "AMBER/data/query" / f"query_{name}.json"
                    for name in ("generative", "discriminative"))
    for path in required:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    test_samples = json.loads((RES_OPD / "data/test_1000.json").read_text(encoding="utf-8"))
    image_root = args.data_root / "COCO/coco2017val/val2017"
    missing_coco = []
    for sample in test_samples:
        current = Path(str(sample.get("image_path") or ""))
        file_name = sample.get("file_name") or current.name
        if not current.is_file() and not (image_root / file_name).is_file():
            missing_coco.append(file_name)
    if missing_coco:
        parser.error(f"Missing {len(missing_coco)} COCO images; first={missing_coco[0]}")
    sys.path.insert(0, str(RES_OPD / "eval"))
    from eval_amber import resolve_image_path

    for name in ("generative", "discriminative"):
        path = args.data_root / "AMBER/data/query" / f"query_{name}.json"
        queries = json.loads(path.read_text(encoding="utf-8"))
        if len(queries) < args.samples:
            parser.error(f"AMBER {name} has fewer than {args.samples} queries")
        for item in queries[:args.samples]:
            resolve_image_path(args.data_root / "AMBER", None, item["image"])
    run_dir = (args.halc_dir / ".smoke_aux"
               / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{socket.gethostname()}_{os.getpid()}")
    run_dir.mkdir(parents=True)
    model_id = f"external-halc-{args.model_path.name}"
    env = os.environ.copy()
    env["RES_OPD_DATA_ROOT"] = str(args.data_root.resolve())
    server_env = dict(env, CUDA_VISIBLE_DEVICES=args.gpu)
    command = [sys.executable, str(RES_OPD / "baselines/external_decode_api.py"),
               "--method", "halc", "--model-path", str(args.model_path),
               "--config", str(config_path), "--detector-config", str(args.detector_config),
               "--model-id", model_id, "--port", str(args.port)]
    server_log = run_dir / "server.log"
    print(f"SMOKE_OUTPUT={run_dir} GPU={args.gpu} port={args.port}", flush=True)
    with server_log.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(command, cwd=ROOT, env=server_env,
                                  stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    api_base = f"http://127.0.0.1:{args.port}/v1/"
    try:
        wait_for_server(server, args.port, args.startup_timeout)
        common = ["--api-base", api_base, "--model-name", model_id,
                  "--parallel-workers", "1", "--max-retries", "1",
                  "--max-samples", str(args.samples)]
        pope_out = run_dir / "pope_smoke"
        run_eval("pope", [sys.executable, str(RES_OPD / "eval/eval_pope.py"),
                          *common, "--test-json", str(RES_OPD / "data/test_1000.json"),
                          "--benchmark", ",".join(POPE_SPLITS),
                          "--pope-source", "res-opd-test",
                          "--max-new-tokens", "64", "--output-dir", str(pope_out)],
                 run_dir, env)
        check_pope(pope_out, args.samples)
        for evaluation_type, cap, discriminative in (("g", 128, False), ("d", 16, True)):
            amber_out = run_dir / f"amber_{evaluation_type}_smoke"
            run_eval(f"amber_{evaluation_type}", [
                sys.executable, str(RES_OPD / "eval/eval_amber.py"),
                *common, "--amber-root", str(args.data_root / "AMBER"),
                "--evaluation-type", evaluation_type,
                "--max-new-tokens-generative", str(cap),
                "--max-new-tokens-discriminative", str(cap),
                "--output-dir", str(amber_out), "--skip-official-eval",
            ], run_dir, env)
            check_amber(amber_out, args.samples, evaluation_type, discriminative)
        print(f"HALC_AUX_SMOKE=PASS model={args.model_path.name} output={run_dir}", flush=True)
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
