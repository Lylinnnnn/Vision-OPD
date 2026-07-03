#!/usr/bin/env python3
"""Recompute AMBER metrics after filtering contaminated thinking outputs.

The script reads AMBER raw_results.jsonl from one or more eval directories,
builds filtered official-format response JSON files, and runs the AMBER scorer
into a probe output directory. Original eval results are never modified.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROBE_DIR = Path(__file__).resolve().parent
RES_OPD_ROOT = PROBE_DIR.parent
REPO_ROOT = RES_OPD_ROOT.parent
EVAL_DIR = RES_OPD_ROOT / "eval"
sys.path.insert(0, str(EVAL_DIR))

from eval_amber import extract_final_response_text, extract_yes_no, parse_official_stdout  # noqa: E402


THINKY_PREFIX_RE = re.compile(
    r"^\s*(so[, ]|got it|let('|’)s|let me|first[, ]|okay[, ]|we need|i need|looking at)",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter AMBER raw results, then recompute official AMBER metrics in "
            "a probe directory without touching the original eval outputs."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "One or more LABEL=PATH entries. PATH can be raw_results.jsonl, "
            "an amber directory, or an eval dataset directory containing amber/."
        ),
    )
    parser.add_argument("--amber-root", default="/home/liuyanlin.lyl/notebook/data/AMBER")
    parser.add_argument("--evaluation-type", default="a", choices=["a", "g", "d", "de", "da", "dr"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--filter-scope",
        default="generative",
        choices=["generative", "all"],
        help="Whether to drop only generative bad rows or all bad rows.",
    )
    parser.add_argument(
        "--filter-mode",
        default="missing_think_close",
        choices=["missing_think_close", "missing_or_loop"],
        help=(
            "missing_think_close drops rows whose model_answer lacks </think>. "
            "missing_or_loop also drops loop-like rows with high repeated ngrams."
        ),
    )
    parser.add_argument(
        "--drop-id-mode",
        default="per_input",
        choices=["per_input", "union"],
        help=(
            "per_input drops bad rows independently for each model. "
            "union first collects bad ids across all inputs, then drops the same ids from every model."
        ),
    )
    parser.add_argument("--loop-repeat-threshold", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default=str(PROBE_DIR / "results" / "amber_filtered_metrics"),
        help="Probe output directory. One subdirectory per label is created.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_raw_results(path_text: str) -> Path:
    path = Path(path_text)
    candidates = []
    if path.is_file():
        candidates.append(path)
    else:
        candidates.extend([path / "raw_results.jsonl", path / "amber" / "raw_results.jsonl"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find AMBER raw_results.jsonl under: {path_text}")


def parse_labeled_inputs(entries: list[str]) -> list[tuple[str, Path]]:
    out = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected LABEL=PATH input, got: {entry}")
        label, path_text = entry.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in input: {entry}")
        out.append((label, resolve_raw_results(path_text)))
    return out


def load_annotations(amber_root: Path) -> dict[int, dict[str, Any]]:
    annotation_path = amber_root / "data" / "annotations.json"
    data = load_json(annotation_path)
    out = {}
    for idx, item in enumerate(data, start=1):
        if isinstance(item, dict):
            out[int(item.get("id", idx))] = item
    return out


def is_generative(record: dict[str, Any], annotations: dict[int, dict[str, Any]]) -> bool:
    try:
        item_id = int(record.get("id"))
    except (TypeError, ValueError):
        return not bool(record.get("is_discriminative"))
    annotation = annotations.get(item_id)
    if annotation:
        return annotation.get("type") == "generative"
    return item_id < 1005 or not bool(record.get("is_discriminative"))


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())


def max_repeated_ngram_count(words: list[str], n: int = 5) -> int:
    if len(words) < n:
        return 0
    counts = {}
    for idx in range(len(words) - n + 1):
        gram = tuple(words[idx : idx + n])
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values()) if counts else 0


def is_bad_thinking_row(record: dict[str, Any], loop_threshold: int, filter_mode: str) -> tuple[bool, str]:
    raw = str(record.get("model_answer", "") or "")
    if not raw.strip():
        return True, "empty_model_answer"
    has_close = "</think>" in raw
    if not has_close:
        if THINKY_PREFIX_RE.search(raw) or "<think>" in raw:
            return True, "missing_think_close"
        return True, "missing_think_close"
    if filter_mode == "missing_or_loop":
        repeat = max_repeated_ngram_count(word_tokens(extract_final_response_text(raw)))
        if repeat >= loop_threshold:
            return True, f"loop_like_repeat5_{repeat}"
    return False, ""


def official_response(record: dict[str, Any], generative: bool) -> str:
    raw = str(record.get("model_answer", "") or "")
    if generative:
        return str(record.get("official_response") or extract_final_response_text(raw))
    return str(record.get("official_response") or extract_yes_no(raw))


def build_filtered_response_file(
    label: str,
    raw_path: Path,
    out_dir: Path,
    annotations: dict[int, dict[str, Any]],
    filter_scope: str,
    filter_mode: str,
    loop_threshold: int,
    forced_drop_ids: set[int] | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(raw_path)
    kept = []
    dropped = []
    for row in rows:
        item_id = int(row["id"])
        generative = is_generative(row, annotations)
        should_check = filter_scope == "all" or generative
        bad = False
        reason = ""
        if forced_drop_ids is not None and item_id in forced_drop_ids and should_check:
            bad, reason = True, "union_forced_drop"
        elif should_check:
            bad, reason = is_bad_thinking_row(row, loop_threshold, filter_mode)
        if bad:
            dropped.append(
                {
                    "id": item_id,
                    "image": row.get("image"),
                    "is_generative": generative,
                    "reason": reason,
                    "model_answer_prefix": str(row.get("model_answer", ""))[:300],
                }
            )
            continue
        kept.append({"id": item_id, "response": official_response(row, generative)})

    out_dir.mkdir(parents=True, exist_ok=True)
    response_path = out_dir / "amber_filtered_responses.json"
    dropped_path = out_dir / "dropped_rows.jsonl"
    with response_path.open("w", encoding="utf-8") as f:
        json.dump(sorted(kept, key=lambda x: int(x["id"])), f, ensure_ascii=False, indent=2)
    with dropped_path.open("w", encoding="utf-8") as f:
        for row in dropped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "label": label,
        "raw_path": str(raw_path),
        "response_path": str(response_path),
        "dropped_path": str(dropped_path),
        "raw_n": len(rows),
        "kept_n": len(kept),
        "dropped_n": len(dropped),
        "dropped_generative_n": sum(1 for r in dropped if r["is_generative"]),
        "dropped_discriminative_n": sum(1 for r in dropped if not r["is_generative"]),
    }


def collect_bad_ids(
    raw_paths: list[Path],
    annotations: dict[int, dict[str, Any]],
    filter_scope: str,
    filter_mode: str,
    loop_threshold: int,
) -> tuple[set[int], dict[str, Any]]:
    bad_ids: set[int] = set()
    per_path = {}
    for raw_path in raw_paths:
        path_bad = []
        for row in read_jsonl(raw_path):
            item_id = int(row["id"])
            generative = is_generative(row, annotations)
            should_check = filter_scope == "all" or generative
            if not should_check:
                continue
            bad, reason = is_bad_thinking_row(row, loop_threshold, filter_mode)
            if bad:
                bad_ids.add(item_id)
                path_bad.append({"id": item_id, "reason": reason, "is_generative": generative})
        per_path[str(raw_path)] = path_bad
    return bad_ids, {"bad_ids": sorted(bad_ids), "per_path": per_path}


def run_amber_eval(response_path: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    scorer = RES_OPD_ROOT / "scripts" / "run_amber_parallel.py"
    cmd = [
        sys.executable,
        str(scorer),
        "--inference_data",
        str(response_path),
        "--evaluation_type",
        args.evaluation_type,
        "--workers",
        str(args.workers),
        "--amber_root",
        args.amber_root,
        "--output_metrics_json",
        str(out_dir / "amber_metrics_raw_counts.json"),
    ]
    amber_root = Path(args.amber_root)
    cmd.extend(
        [
            "--word_association",
            str(amber_root / "data" / "relation.json"),
            "--safe_words",
            str(amber_root / "data" / "safe_words.txt"),
            "--annotation",
            str(amber_root / "data" / "annotations.json"),
            "--metrics",
            str(amber_root / "data" / "metrics.txt"),
        ]
    )
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    official_log = out_dir / "official_eval.log"
    official_log.write_text(proc.stdout, encoding="utf-8")
    return {
        "official_returncode": proc.returncode,
        "official_log": str(official_log),
        "metrics": parse_official_stdout(proc.stdout),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_summary_md(path: Path, summaries: list[dict[str, Any]]) -> None:
    metric_keys = [
        ("generative_task_chair", "Gen CHAIR"),
        ("generative_task_cover", "Gen Cover"),
        ("generative_task_hal", "Gen HAL"),
        ("generative_task_cog", "Gen COG"),
        ("descriminative_task_f1", "Disc F1"),
        ("exsitence_f1", "Existence F1"),
        ("attribute_f1", "Attribute F1"),
        ("relation_f1", "Relation F1"),
    ]
    header = ["Model", "Raw N", "Kept", "Dropped", "Drop gen"]
    header.extend(name for _, name in metric_keys)
    lines = [
        "# Filtered AMBER Metrics",
        "",
        "Rows are recomputed after dropping contaminated thinking rows into a probe-only output directory.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for summary in summaries:
        row = [
            summary["label"],
            str(summary["raw_n"]),
            str(summary["kept_n"]),
            str(summary["dropped_n"]),
            str(summary["dropped_generative_n"]),
        ]
        metrics = summary.get("metrics", {})
        for key, _ in metric_keys:
            row.append(fmt(metrics.get(key)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    inputs = parse_labeled_inputs(args.inputs)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(Path(args.amber_root))
    forced_drop_ids = None
    union_payload = None
    if args.drop_id_mode == "union":
        forced_drop_ids, union_payload = collect_bad_ids(
            [raw_path for _, raw_path in inputs],
            annotations=annotations,
            filter_scope=args.filter_scope,
            filter_mode=args.filter_mode,
            loop_threshold=args.loop_repeat_threshold,
        )
        with (output_root / "union_drop_ids.json").open("w", encoding="utf-8") as f:
            json.dump(union_payload, f, ensure_ascii=False, indent=2)

    summaries = []
    for label, raw_path in inputs:
        label_dir = output_root / label
        summary = build_filtered_response_file(
            label=label,
            raw_path=raw_path,
            out_dir=label_dir,
            annotations=annotations,
            filter_scope=args.filter_scope,
            filter_mode=args.filter_mode,
            loop_threshold=args.loop_repeat_threshold,
            forced_drop_ids=forced_drop_ids,
        )
        eval_summary = run_amber_eval(Path(summary["response_path"]), label_dir, args)
        summary.update(eval_summary)
        with (label_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        if summary["official_returncode"] != 0:
            raise RuntimeError(f"AMBER filtered eval failed for {label}. See {summary['official_log']}")
        summaries.append(summary)

    payload = {
        "amber_root": args.amber_root,
        "evaluation_type": args.evaluation_type,
        "filter_scope": args.filter_scope,
        "filter_mode": args.filter_mode,
        "drop_id_mode": args.drop_id_mode,
        "union_drop_ids": sorted(forced_drop_ids) if forced_drop_ids is not None else None,
        "summaries": summaries,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    write_summary_md(output_root / "summary.md", summaries)
    print(f"Wrote: {output_root / 'summary.json'}")
    print(f"Wrote: {output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
