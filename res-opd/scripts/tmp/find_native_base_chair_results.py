#!/usr/bin/env python3
"""Resolve the three 2B low-resolution Base CHAIR runs by metrics first.

The historical result layout places these evaluations under experiment
directories whose names are not reliable enough for the paper-wide manifest.
This helper scans every legacy ``chair_metrics.json`` below a result root,
matches the paper's legacy metric triples first, and uses path hints only to
break ties.  On success it writes a temporary manifest whose three affected
rows are pinned to exact metrics paths; it never edits the source manifest or
evaluation outputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional


TARGETS = {
    "2b/native_base/sr025": {
        "expected": {"CHAIRi": 0.2150, "CHAIRs": 0.0697, "ObjF1": 0.6971},
        "ratio_regex": r"sr0(?:p|[._-])25",
    },
    "2b/native_base/sr05": {
        "expected": {"CHAIRi": 0.2320, "CHAIRs": 0.0730, "ObjF1": 0.7210},
        "ratio_regex": r"sr0(?:p|[._-])5(?:0)?",
    },
    "2b/native_base/sr075": {
        "expected": {"CHAIRi": 0.2262, "CHAIRs": 0.0708, "ObjF1": 0.7289},
        "ratio_regex": r"sr0(?:p|[._-])75",
    },
}

FILTER_KEYS = {
    "experiment_name_include",
    "experiment_name_exclude",
    "experiment_name_regex",
    "result_path_include",
    "result_path_exclude",
    "result_path_regex",
    "metrics_path_exact",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Find the three 2B low-resolution Base CHAIR files by legacy metrics."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/"
            "res-opd/eval_results/instruct"
        ),
        help="Result tree recursively scanned for chair_metrics.json.",
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=repo_root / "res-opd/eval/manifests/paper_all_chair_legacy.json",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=repo_root
        / "res-opd/eval_results/paper_all_chair_legacy.native_base_resolved.json",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=repo_root
        / "res-opd/eval_results/native_base_chair_match_audit.json",
    )
    parser.add_argument("--tolerance", type=float, default=0.000051)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidates(root: Path) -> list[dict]:
    if not root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {root}")
    candidates = []
    for metrics_path in sorted(root.rglob("chair_metrics.json")):
        eval_results = metrics_path.parent / "eval_results.jsonl"
        if not eval_results.is_file():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING unreadable metrics {metrics_path}: {exc}", file=sys.stderr)
            continue
        candidates.append(
            {
                "metrics_path": str(metrics_path.resolve()),
                "eval_results": str(eval_results.resolve()),
                "metrics": metrics,
            }
        )
    return candidates


def path_score(path: str, ratio_regex: str) -> tuple[int, list[str]]:
    lower = path.lower()
    score = 0
    reasons = []
    checks = [
        (bool(re.search(ratio_regex, lower)), 8, "ratio"),
        ("2b" in lower, 4, "2b"),
        ("frozen-rkl" in lower, 3, "frozen-rkl"),
        (bool(re.search(r"tr1(?:p0|[._-]0)?", lower)), 2, "tr1.0"),
        ("global_step" in lower, 1, "global_step"),
    ]
    for passed, weight, reason in checks:
        if passed:
            score += weight
            reasons.append(reason)
    for unwanted in ("riskmask", "randommask", "entropymask", "sft", "fixed-rkl"):
        if unwanted in lower:
            score -= 8
            reasons.append(f"exclude:{unwanted}")
    return score, reasons


def rank_for_target(candidates: list[dict], target: dict, tolerance: float) -> list[dict]:
    expected = target["expected"]
    ranked = []
    for candidate in candidates:
        metrics = candidate["metrics"]
        if not all(isinstance(metrics.get(key), (int, float)) for key in expected):
            continue
        diffs = {key: abs(float(metrics[key]) - value) for key, value in expected.items()}
        score, reasons = path_score(candidate["metrics_path"], target["ratio_regex"])
        ranked.append(
            {
                "metrics_path": candidate["metrics_path"],
                "eval_results": candidate["eval_results"],
                "actual": {key: float(metrics[key]) for key in expected},
                "diff": diffs,
                "exact_metric_match": all(value <= tolerance for value in diffs.values()),
                "max_abs_diff": max(diffs.values()),
                "sum_abs_diff": sum(diffs.values()),
                "path_score": score,
                "path_reasons": reasons,
            }
        )
    ranked.sort(
        key=lambda item: (
            not item["exact_metric_match"],
            item["max_abs_diff"],
            item["sum_abs_diff"],
            -item["path_score"],
            item["metrics_path"],
        )
    )
    return ranked


def resolve_exact(
    ranked: list[dict],
) -> tuple[Optional[dict], Optional[str], list[dict]]:
    exact = [item for item in ranked if item["exact_metric_match"]]
    if not exact:
        return None, "no_exact_metric_match", []

    for item in exact:
        item["eval_results_sha256"] = sha256(Path(item["eval_results"]))
    by_hash: dict[str, list[dict]] = {}
    for item in exact:
        by_hash.setdefault(item["eval_results_sha256"], []).append(item)
    if len(by_hash) == 1:
        aliases = next(iter(by_hash.values()))
        selected = sorted(
            aliases,
            key=lambda item: (-item["path_score"], len(item["metrics_path"]), item["metrics_path"]),
        )[0]
        return selected, None, aliases

    best_score = max(item["path_score"] for item in exact)
    best = [item for item in exact if item["path_score"] == best_score]
    best_hashes = {item["eval_results_sha256"] for item in best}
    if len(best_hashes) == 1:
        selected = sorted(best, key=lambda item: (len(item["metrics_path"]), item["metrics_path"]))[0]
        aliases = by_hash[selected["eval_results_sha256"]]
        return selected, None, aliases
    return None, "multiple_distinct_exact_matches_with_tied_path_score", exact


def write_resolved_manifest(base_path: Path, output_path: Path, selected: dict[str, dict]) -> None:
    manifest = json.loads(base_path.read_text(encoding="utf-8"))
    output = copy.deepcopy(manifest)
    rows_by_label = {row.get("label"): row for row in output.get("rows", [])}
    missing = sorted(set(selected) - set(rows_by_label))
    if missing:
        raise KeyError(f"Base manifest is missing target labels: {missing}")
    for label, match in selected.items():
        row = rows_by_label[label]
        for key in FILTER_KEYS:
            row.pop(key, None)
        row["metrics_path_exact"] = match["metrics_path"]
    output["description"] = (
        manifest.get("description", "")
        + " The three 2B Native Base rows are pinned by the temporary metrics-first resolver."
    ).strip()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    candidates = load_candidates(args.root.expanduser().resolve())
    if not candidates:
        raise RuntimeError(f"No usable chair_metrics.json files found under {args.root}")

    audit = {
        "root": str(args.root.expanduser().resolve()),
        "tolerance": args.tolerance,
        "metrics_files_scanned": len(candidates),
        "targets": {},
    }
    selected = {}
    unresolved = []
    for label, target in TARGETS.items():
        ranked = rank_for_target(candidates, target, args.tolerance)
        resolved, reason, aliases = resolve_exact(ranked)
        audit["targets"][label] = {
            "expected": target["expected"],
            "resolved": resolved,
            "equivalent_aliases": aliases,
            "unresolved_reason": reason,
            "top_candidates": ranked[: max(args.top_k, 1)],
        }
        if resolved is None:
            unresolved.append(label)
        else:
            selected[label] = resolved

        print(f"\n[{label}] expected={target['expected']}")
        if resolved:
            print(f"  SELECTED {resolved['metrics_path']}")
            print(f"  actual={resolved['actual']} path_score={resolved['path_score']}")
        else:
            print(f"  UNRESOLVED: {reason}")
            for item in ranked[: min(args.top_k, 10)]:
                print(
                    f"  nearest max_diff={item['max_abs_diff']:.6f} "
                    f"sum_diff={item['sum_abs_diff']:.6f} path_score={item['path_score']} "
                    f"actual={item['actual']} {item['metrics_path']}"
                )

    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(f"\nWrote audit: {args.audit_json}")
    if unresolved:
        print(
            "Could not safely resolve: " + ", ".join(unresolved) + ". "
            "No derived manifest was written.",
            file=sys.stderr,
        )
        return 2

    write_resolved_manifest(args.base_manifest, args.output_manifest, selected)
    print(f"Wrote resolved manifest: {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
