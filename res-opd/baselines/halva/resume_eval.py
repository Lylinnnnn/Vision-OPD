#!/usr/bin/env python3
"""Resume a HALVA sharded evaluation after a failed or interrupted run."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
RUN_SCRIPT = Path(__file__).with_name("run_coco5k.sh")
MODEL_NAMES = {
    "qwen2b": "Qwen3VL-2B-Instruct",
    "qwen4b": "Qwen3VL-4B-Instruct",
    "qwen8b": "Qwen3VL-8B-Instruct",
    "internvl2b": "InternVL3-2B-hf",
}
ERROR_PREFIXES = ("[ERROR]", "[API_ERROR]", "[FUTURE_ERROR]")


def run_capture(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return result.stdout


def candidate_roots(output: Path, tmp_root: Path, run_name: str, model: str) -> list[Path]:
    roots = list((output / ".shards").glob("*"))
    roots += list(tmp_root.glob(f"halva_{run_name}_{model}_resume*"))
    if run_name == "halva_coco5k_e1":
        roots += list(tmp_root.glob(f"halva_{model}_resume_*"))
    return sorted({path.resolve() for path in roots if (path / "shard_00" / "out").is_dir()})


def select_root(output: Path, tmp_root: Path, run_name: str, model: str,
                explicit: Path | None) -> Path:
    if explicit is not None:
        source = explicit.resolve()
        if not (source / "shard_00" / "out").is_dir():
            raise RuntimeError(f"Not a HALVA shard root: {source}")
        return source
    roots = candidate_roots(output, tmp_root, run_name, model)
    if len(roots) != 1:
        listing = "\n".join(f"  {path}" for path in roots) or "  (none)"
        raise RuntimeError(
            f"Found {len(roots)} candidate shard roots. Pass --shard-root explicitly:\n{listing}"
        )
    return roots[0]


def validate_shards(root: Path, count: int) -> None:
    missing = [index for index in range(count)
               if not (root / f"shard_{index:02d}" / "out").is_dir()]
    if missing:
        raise RuntimeError(f"Shard root {root} is missing out directories: {missing}")


def process_identity(pid: int) -> tuple[int, str] | None:
    status = Path(f"/proc/{pid}/status")
    stat = Path(f"/proc/{pid}/stat")
    try:
        uid_line = next(line for line in status.read_text().splitlines() if line.startswith("Uid:"))
        uid = int(uid_line.split()[1])
        stat_text = stat.read_text()
        fields = stat_text[stat_text.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return uid, fields[19]
    except (OSError, StopIteration, IndexError, ValueError):
        return None


def gpu_pids() -> set[int]:
    output = run_capture(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    return {int(row[0].strip()) for row in csv.reader(output.splitlines())
            if row and row[0].strip().isdigit()}


def old_parent_pids(merged: Path) -> set[int]:
    found = set()
    for line in run_capture(["ps", "-ww", "-eo", "pid=,uid=,args="]).splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() \
                and int(parts[1]) == os.getuid() \
                and "eval_after_merge_sharded_8gpu.sh" in parts[2] \
                and str(merged) in parts[2]:
            found.add(int(parts[0]))
    return found


def stop_old_tmux_sessions(root: Path) -> None:
    try:
        panes = run_capture(["tmux", "list-panes", "-a", "-F",
                             "#{session_name}\t#{pane_start_command}"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return
    for line in panes.splitlines():
        session, _, command = line.partition("\t")
        if str(root / "shard_") in command:
            print(f"Stopping old shard tmux session: {session}", flush=True)
            subprocess.run(["tmux", "kill-session", "-t", session], check=False)


def ports_busy(ports: list[int]) -> list[int]:
    return [port for port in ports
            if run_capture(["ss", "-H", "-ltn", f"( sport = :{port} )"]).strip()]


def choose_free_base(preferred: int, count: int, excluded: set[int], blocks: int = 40) -> int:
    for offset in range(blocks):
        base = preferred + 100 * offset
        if base in excluded or base + 10 * (count - 1) > 65535:
            continue
        ports = [base + 10 * index for index in range(count)]
        if not ports_busy(ports):
            return base
    raise RuntimeError(f"No free block of {count} shard ports from {preferred}; "
                       "select another --base-port")


def port_conflict_in_logs(root: Path, count: int) -> bool:
    for index in range(count):
        path = root / f"shard_{index:02d}" / "eval.log"
        if not path.is_file():
            continue
        log = path.read_text(encoding="utf-8", errors="replace")
        if "Address already in use" in log or "VLLM_PORT_CLEANUP=False and port" in log:
            return True
        if "is already in use" in log and "Port " in log:
            return True
    return False


def stop_processes(pids: set[int], timeout: float = 15.0) -> None:
    owned = {pid: identity for pid in pids
             if (identity := process_identity(pid)) is not None and identity[0] == os.getuid()}
    ignored = sorted(pids - owned.keys())
    if ignored:
        print(f"Skipping vanished or other-user processes: {ignored}", flush=True)
    if not owned:
        return
    print(f"Stopping current-user processes: {sorted(owned)}", flush=True)
    for pid, identity in owned.items():
        if process_identity(pid) == identity:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(process_identity(pid) == identity for pid, identity in owned.items()):
            break
        time.sleep(0.5)
    for pid, identity in owned.items():
        if process_identity(pid) == identity:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    time.sleep(1)
    remaining = [pid for pid, identity in owned.items() if process_identity(pid) == identity]
    if remaining:
        raise RuntimeError(f"Processes did not exit: {remaining}")


def stage_root(source: Path, tmp_root: Path, run_name: str, model: str) -> Path:
    if source.is_relative_to(tmp_root.resolve()):
        return source
    target = tmp_root / f"halva_{run_name}_{model}_resume_{source.name}"
    if target.exists():
        raise RuntimeError(f"Staging destination already exists; inspect it first: {target}")
    print(f"Moving shard answers to local temporary storage: {source} -> {target}", flush=True)
    shutil.move(str(source), str(target))
    return target


def recover_amber(root: Path, count: int, stamp: str) -> None:
    for index in range(count):
        directory = root / f"shard_{index:02d}" / "out" / "amber"
        raw = directory / "raw_results.jsonl"
        response = directory / "amber_a_responses.json"
        if not raw.is_file():
            continue
        rows = []
        malformed = 0
        missing_newline = False
        with raw.open(encoding="utf-8") as stream:
            for line in stream:
                missing_newline = not line.endswith("\n")
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1
        if malformed or missing_newline:
            backup = raw.with_name(f"{raw.name}.before_resume_{stamp}")
            shutil.copy2(raw, backup)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                             prefix="raw_recovered_", delete=False) as stream:
                temporary = Path(stream.name)
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            temporary.replace(raw)
        try:
            existing = json.loads(response.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = []
        if not isinstance(existing, list):
            raise RuntimeError(f"Invalid AMBER response file shape: {response}")
        by_id = {int(row["id"]): row for row in existing
                 if isinstance(row, dict) and row.get("response") and "id" in row}
        before = len(by_id)
        for row in rows:
            answer = str(row.get("model_answer") or "").strip()
            official = str(row.get("official_response") or "").strip()
            if answer and official and not answer.startswith(ERROR_PREFIXES) \
                    and not official.startswith(ERROR_PREFIXES):
                by_id[int(row["id"])] = {"id": int(row["id"]), "response": official}
        if response.exists():
            shutil.copy2(response, response.with_name(f"{response.name}.before_resume_{stamp}"))
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                         prefix="amber_recovered_", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump([by_id[key] for key in sorted(by_id)], stream, ensure_ascii=False)
        temporary.replace(response)
        print(f"shard_{index:02d} AMBER recovered={len(by_id)} previously_indexed={before} "
              f"raw_rows={len(rows)} malformed={malformed}", flush=True)


def archive_status(root: Path, count: int, stamp: str) -> None:
    for index in range(count):
        shard = root / f"shard_{index:02d}"
        for name in ("exit_code", "eval.log"):
            path = shard / name
            if path.exists():
                path.rename(shard / f"{name}.before_resume_{stamp}")


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                json.loads(line)
                count += 1
    return count


def validate_result(output: Path) -> None:
    required = [output / "chair_metrics.json", output / "pope" / "pope_summary.json",
                output / "amber" / "amber_metrics.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Evaluation returned success but metrics are missing: {missing}")
    expected = {output / "eval_results.jsonl": 1000,
                output / "amber" / "raw_results.jsonl": 15220}
    expected.update({output / "pope" / split / "eval_results.jsonl": 5223
                     for split in ("pope_adv", "pope_pop", "pope_random")})
    for path, count in expected.items():
        if not path.is_file() or count_jsonl(path) != count:
            raise RuntimeError(f"Incomplete merged output: {path} (expected {count} rows)")
    responses = json.loads((output / "amber" / "amber_a_responses.json").read_text(encoding="utf-8"))
    if len(responses) != 15220 or len({int(row["id"]) for row in responses}) != 15220:
        raise RuntimeError("AMBER official response count is incomplete or contains duplicate ids")
    amber = json.loads(required[-1].read_text(encoding="utf-8"))
    if amber.get("official_returncode") != 0:
        raise RuntimeError(f"AMBER official scoring did not succeed: {required[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODEL_NAMES)
    parser.add_argument("--run-name", default="halva_coco5k_e1")
    parser.add_argument("--shard-root", type=Path, help="Exact old .shards run or staged /tmp directory")
    parser.add_argument("--tmp-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--base-port", type=int, default=35400)
    parser.add_argument("--gpu-list", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--clear-user-gpus", action="store_true",
                        help="Stop ALL GPU processes owned by this user on this machine")
    parser.add_argument("--vllm-timeout", type=int, default=3600)
    parser.add_argument("--port-attempts", type=int, default=3,
                        help="Retry with another free port block only after a port-conflict failure")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpus = [value.strip() for value in args.gpu_list.split(",") if value.strip()]
    if len(gpus) != 8 or len(set(gpus)) != 8:
        parser.error("Expected eight distinct GPU indices for the existing eight-way shards")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_name):
        parser.error("--run-name must be one safe path component")
    if args.port_attempts < 1:
        parser.error("--port-attempts must be positive")
    model_path = Path(os.environ.get("HALVA_MODEL_PATH") or MODEL_NAMES[args.model])
    output = (ROOT / "eval_results/external_baselines/full" / model_path.name /
              "train5000_test1000_original_sr1p0" / f"{args.run_name}_official_v2")
    work_root = Path(os.environ.get("HALVA_WORK_ROOT") or ROOT / "checkpoints/halva_coco5k")
    merged = work_root / args.run_name / args.model / "merged" / "config.json"
    if not merged.is_file():
        raise RuntimeError(f"Exported model is missing; resume cannot proceed: {merged}")
    source = select_root(output, args.tmp_root, args.run_name, args.model, args.shard_root)
    validate_shards(source, len(gpus))
    ports = [args.base_port + 10 * index for index in range(len(gpus))]
    print(f"Model={args.model} run={args.run_name} source={source} output={output}", flush=True)
    print(f"GPUs={gpus} ports={ports} clear_user_gpus={args.clear_user_gpus}", flush=True)
    if args.dry_run:
        return
    lock_path = args.tmp_root / f"halva_resume_{args.run_name}_{args.model}.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another resume is already running on this machine: {lock_path}") from exc
    if args.clear_user_gpus:
        print("WARNING: stopping every current-user GPU process on this machine.", flush=True)
        stop_processes(gpu_pids() | old_parent_pids(merged.parent))
        stop_old_tmux_sessions(source)
        remaining = {pid for pid in gpu_pids()
                     if (identity := process_identity(pid)) is not None and identity[0] == os.getuid()}
        if remaining:
            raise RuntimeError(f"Current-user GPU processes remain after cleanup: {sorted(remaining)}")
    root = stage_root(source, args.tmp_root, args.run_name, args.model)
    validate_shards(root, len(gpus))
    env = os.environ.copy()
    env.update({
        "HALVA_RUN_NAME": args.run_name,
        "HALVA_GPU_LIST": ",".join(gpus),
        "HALVA_EVAL_SHARDS": str(len(gpus)),
        "HALVA_VLLM_TIMEOUT": str(args.vllm_timeout),
        "HALVA_EVAL_MAX_SAMPLES": "1000",
        "HALVA_AUX_MAX_SAMPLES": "0",
        "HALVA_SKIP_AMBER_METRICS": "False",
        "RES_OPD_DATA_ROOT": env.get("RES_OPD_DATA_ROOT", "/data/oss_bucket_0/yanlin"),
        "SHARDED_EVAL_WORK_DIR": str(root),
        "SHARDED_EVAL_KEEP_SHARDS": "True",
    })
    tried = set()
    for attempt in range(1, args.port_attempts + 1):
        base_port = choose_free_base(args.base_port, len(gpus), tried)
        tried.add(base_port)
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
        recover_amber(root, len(gpus), stamp)
        archive_status(root, len(gpus), stamp)
        env["HALVA_BASE_PORT"] = str(base_port)
        print(f"HALVA_RESUME attempt={attempt}/{args.port_attempts} base_port={base_port}", flush=True)
        result = subprocess.run(["bash", str(RUN_SCRIPT), args.model, "eval"], cwd=REPO,
                                env=env, check=False)
        if result.returncode == 0:
            validate_result(output)
            print(f"HALVA_RESUME=PASS output={output} shards={root}", flush=True)
            return
        if not port_conflict_in_logs(root, len(gpus)) or attempt == args.port_attempts:
            raise RuntimeError(f"HALVA eval failed (exit={result.returncode}); answers retained at {root}")
        print("Port conflict detected; retaining shard answers and selecting another port block.",
              flush=True)
        stop_old_tmux_sessions(root)
        if args.clear_user_gpus:
            stop_processes(gpu_pids())


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"HALVA_RESUME=FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
