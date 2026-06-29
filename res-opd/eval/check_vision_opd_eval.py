#!/usr/bin/env python3
"""Check whether Vision-OPD auxiliary benchmark outputs are complete."""

import argparse
import json
import sys
from pathlib import Path


BENCHMARK_JSON_MAP = {
    "zoombench": "zoombench.json",
    "vstar": "vstar.json",
    "hrbench-4k": "hr_bench_4k.json",
    "hrbench-8k": "hr_bench_8k.json",
    "mme-realworld": "MME_RealWorld.json",
    "mme-realworld-cn": "MME_RealWorld_CN.json",
    "mme-realworld-lite": "MME_RealWorld_Lite.json",
    "mmstar": "mmstar.json",
    "pope": "POPE.json",
    "pope_adv": "POPE_adv.json",
    "pope_pop": "POPE_pop.json",
    "pope_random": "POPE_random.json",
    "cv-bench": "cv_bench.json",
    "mmvp": "mmvp.json",
    "visualprobe": "visualprobe.json",
}


def split_csv(value):
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


def find_judge_files(result_root, benchmark):
    root = Path(result_root)
    candidates = []
    direct_dir = root / benchmark / "judge"
    if direct_dir.exists():
        candidates.extend(direct_dir.glob("*_answer.jsonl"))
    direct_dir = root / "vision_opd" / "judge" / benchmark
    if direct_dir.exists():
        candidates.extend(direct_dir.glob("*_answer.jsonl"))
    candidates.extend(root.glob(f"*/{benchmark}/judge/*_answer.jsonl"))
    candidates.extend(root.glob(f"*/vision_opd/judge/{benchmark}/*_answer.jsonl"))
    return sorted(set(candidates))


def check_one(result_root, benchmark, benchmark_data_dir):
    bench_json = BENCHMARK_JSON_MAP.get(benchmark)
    if not bench_json:
        return {
            "benchmark": benchmark,
            "complete": False,
            "reason": f"unsupported benchmark: {benchmark}",
        }

    benchmark_json = Path(benchmark_data_dir) / bench_json
    if not benchmark_json.exists():
        return {
            "benchmark": benchmark,
            "complete": False,
            "reason": f"benchmark json missing: {benchmark_json}",
        }

    expected = len(load_records(benchmark_json))
    best = None
    for path in find_judge_files(result_root, benchmark):
        try:
            records = load_records(path)
        except Exception as exc:  # noqa: BLE001 - report corrupt files as incomplete.
            current = {
                "benchmark": benchmark,
                "path": str(path),
                "complete": False,
                "records": 0,
                "expected": expected,
                "judged": 0,
                "reason": f"failed to read judge output: {exc}",
            }
        else:
            judged = sum(1 for item in records if str(item.get("judge", "")).strip())
            complete = len(records) == expected and judged == expected
            current = {
                "benchmark": benchmark,
                "path": str(path),
                "complete": complete,
                "records": len(records),
                "expected": expected,
                "judged": judged,
                "reason": "ok" if complete else "incomplete judge records",
            }
        if current["complete"]:
            return current
        if best is None or current.get("records", 0) > best.get("records", 0):
            best = current

    if best is not None:
        return best
    return {
        "benchmark": benchmark,
        "complete": False,
        "records": 0,
        "expected": expected,
        "judged": 0,
        "reason": f"judge output missing under {Path(result_root)}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--benchmarks", required=True, help="Comma-separated Vision-OPD benchmark names")
    parser.add_argument("--benchmark-data-dir", required=True)
    parser.add_argument("--json", action="store_true", help="Print machine-readable status")
    args = parser.parse_args()

    statuses = [
        check_one(args.result_root, benchmark, args.benchmark_data_dir)
        for benchmark in split_csv(args.benchmarks)
    ]
    all_complete = bool(statuses) and all(item["complete"] for item in statuses)

    if args.json:
        print(json.dumps({"complete": all_complete, "benchmarks": statuses}, ensure_ascii=False, indent=2))
    else:
        for item in statuses:
            marker = "OK" if item["complete"] else "MISS"
            detail = item.get("path") or item.get("reason", "")
            counts = ""
            if "expected" in item:
                counts = f" records={item.get('records', 0)}/{item.get('expected', 0)} judged={item.get('judged', 0)}"
            print(f"{marker} {item['benchmark']}:{counts} {detail}")

    return 0 if all_complete else 1


if __name__ == "__main__":
    sys.exit(main())
