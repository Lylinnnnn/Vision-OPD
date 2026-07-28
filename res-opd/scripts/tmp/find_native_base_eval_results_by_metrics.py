#!/usr/bin/env python3
"""Find the three 2B Native Base CHAIR outputs using metrics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGETS = {
    "2b/native_base/sr025": {
        "CHAIRi": 0.2150,
        "CHAIRs": 0.0697,
        "ObjF1": 0.6971,
    },
    "2b/native_base/sr05": {
        "CHAIRi": 0.2320,
        "CHAIRs": 0.0730,
        "ObjF1": 0.7210,
    },
    "2b/native_base/sr075": {
        "CHAIRi": 0.2262,
        "CHAIRs": 0.0708,
        "ObjF1": 0.7289,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find the three Native Base eval outputs by their "
            "paper CHAIR metrics. Directory and experiment names are ignored."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Evaluation-result tree to scan recursively.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Decimal places used for matching paper metrics (default: 4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def metric_signature(metrics: dict, decimals: int) -> tuple[float, float, float] | None:
    keys = ("CHAIRi", "CHAIRs", "ObjF1")
    if not all(isinstance(metrics.get(key), (int, float)) for key in keys):
        return None
    return tuple(round(float(metrics[key]), decimals) for key in keys)


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluation root does not exist: {root}")

    target_signatures = {
        label: metric_signature(expected, args.decimals)
        for label, expected in TARGETS.items()
    }
    matches = {label: [] for label in TARGETS}
    scanned = 0

    for metrics_path in sorted(root.rglob("chair_metrics*.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        scanned += 1
        signature = metric_signature(metrics, args.decimals)
        if signature is None:
            continue
        for label, target_signature in target_signatures.items():
            if signature != target_signature:
                continue
            eval_results = metrics_path.parent / "eval_results.jsonl"
            matches[label].append(
                {
                    "metrics_path": str(metrics_path.resolve()),
                    "eval_results": (
                        str(eval_results.resolve()) if eval_results.is_file() else None
                    ),
                    "actual": {
                        key: float(metrics[key])
                        for key in ("CHAIRi", "CHAIRs", "ObjF1")
                    },
                }
            )

    report = {
        "root": str(root),
        "match_decimals": args.decimals,
        "metrics_files_scanned": scanned,
        "targets": {
            label: {"expected": TARGETS[label], "matches": matches[label]}
            for label in TARGETS
        },
    }

    print(f"Scanned {scanned} chair_metrics*.json files under {root}")
    failed = False
    for label, expected in TARGETS.items():
        found = matches[label]
        print(f"\n[{label}] expected={expected}")
        if not found:
            print("  NOT FOUND")
            failed = True
            continue
        for index, item in enumerate(found, 1):
            print(f"  MATCH {index}: {item['metrics_path']}")
            print(f"    eval_results: {item['eval_results'] or 'MISSING'}")
            print(f"    actual: {item['actual']}")
            if item["eval_results"] is None:
                failed = True

    output = args.output
    if output is None:
        output = root / "native_base_eval_results_metric_matches.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\nWrote report: {output}")

    if failed:
        print("At least one target is missing or has no sibling eval_results.jsonl.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
