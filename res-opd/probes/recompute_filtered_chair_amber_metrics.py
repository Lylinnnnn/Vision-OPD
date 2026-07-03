#!/usr/bin/env python3
"""Recompute CHAIR and AMBER after filtering contaminated thinking outputs.

This probe reads one or more eval dataset directories, filters rows whose
thinking block appears unclosed or loop-like, and writes recomputed metrics into
res-opd/probes/results without modifying the original eval outputs.
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
from robust_chair_analysis import (  # noqa: E402
    aggregate_from_arrays,
    build_double_word_dict,
    compute_per_sample,
    parse_official_synonyms,
)


THINKY_PREFIX_RE = re.compile(
    r"^\s*(so[, ]|got it|let('|’)s|let me|first[, ]|okay[, ]|we need|i need|looking at)",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter contaminated thinking rows and recompute CHAIR/AMBER metrics "
            "in a probe-only output directory."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "One or more LABEL=PATH entries. PATH is usually an eval dataset "
            "directory containing eval_results.jsonl and amber/raw_results.jsonl."
        ),
    )
    parser.add_argument(
        "--tasks",
        default="chair,amber",
        help="Comma-separated tasks to recompute: chair,amber.",
    )
    parser.add_argument("--amber-root", default="/home/liuyanlin.lyl/notebook/data/AMBER")
    parser.add_argument("--amber-evaluation-type", default="a", choices=["a", "g", "d", "de", "da", "dr"])
    parser.add_argument("--amber-workers", type=int, default=16)
    parser.add_argument(
        "--amber-filter-scope",
        default="generative",
        choices=["generative", "all"],
        help="For AMBER, drop only generative bad rows or all bad rows.",
    )
    parser.add_argument(
        "--filter-mode",
        default="missing_think_close",
        choices=["missing_think_close", "missing_or_loop"],
        help="missing_or_loop also drops rows with repeated final-answer 5-grams.",
    )
    parser.add_argument(
        "--thinking-detection",
        default="auto",
        choices=["auto", "always", "never"],
        help=(
            "auto requires </think> only when a file looks like thinking output; "
            "always requires it for every non-empty row; never only applies loop filtering."
        ),
    )
    parser.add_argument(
        "--drop-id-mode",
        default="per_input",
        choices=["per_input", "union"],
        help="union drops the same bad image/sample ids from every input for each task.",
    )
    parser.add_argument("--loop-repeat-threshold", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default=str(PROBE_DIR / "results" / "filtered_chair_amber_metrics"),
        help="Probe output directory. One subdirectory per label is created.",
    )
    return parser.parse_args()


def parse_tasks(value: str) -> set[str]:
    tasks = {item.strip().lower() for item in value.split(",") if item.strip()}
    unknown = tasks - {"chair", "amber"}
    if unknown:
        raise ValueError(f"Unsupported task(s): {sorted(unknown)}")
    if not tasks:
        raise ValueError("--tasks cannot be empty")
    return tasks


def read_json(path: Path) -> Any:
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_chair_results(path_text: str) -> Path:
    path = Path(path_text)
    candidates = []
    if path.is_file():
        candidates.append(path)
    else:
        candidates.extend([path / "eval_results.jsonl", path / "chair" / "eval_results.jsonl"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Could not find CHAIR eval_results.jsonl for {path_text}. Checked: {checked}")


def resolve_amber_raw_results(path_text: str) -> Path:
    path = Path(path_text)
    candidates = []
    if path.is_file():
        candidates.append(path)
    else:
        candidates.extend([path / "amber" / "raw_results.jsonl", path / "raw_results.jsonl"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Could not find AMBER raw_results.jsonl for {path_text}. Checked: {checked}")


def parse_labeled_inputs(entries: list[str], tasks: set[str]) -> list[dict[str, Any]]:
    out = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected LABEL=PATH input, got: {entry}")
        label, path_text = entry.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in input: {entry}")
        item: dict[str, Any] = {"label": label, "input_path": path_text}
        if "chair" in tasks:
            item["chair_path"] = resolve_chair_results(path_text)
        if "amber" in tasks:
            item["amber_path"] = resolve_amber_raw_results(path_text)
        out.append(item)
    return out


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())


def max_repeated_ngram_count(words: list[str], n: int = 5) -> int:
    if len(words) < n:
        return 0
    counts: dict[tuple[str, ...], int] = {}
    for idx in range(len(words) - n + 1):
        gram = tuple(words[idx : idx + n])
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values()) if counts else 0


def audit_text_for_chair(row: dict[str, Any]) -> str:
    return str(row.get("raw_generated_caption") or row.get("generated_caption") or "")


def audit_text_for_amber(row: dict[str, Any]) -> str:
    return str(row.get("model_answer") or "")


def looks_like_thinking_output(rows: list[dict[str, Any]], text_getter) -> bool:
    for row in rows:
        text = text_getter(row)
        if "</think>" in text or "<think>" in text or THINKY_PREFIX_RE.search(text):
            return True
    return False


def should_require_think_close(
    rows: list[dict[str, Any]],
    text_getter,
    thinking_detection: str,
) -> bool:
    if thinking_detection == "always":
        return True
    if thinking_detection == "never":
        return False
    return looks_like_thinking_output(rows, text_getter)


def is_bad_text(
    text: str,
    require_think_close: bool,
    filter_mode: str,
    loop_threshold: int,
) -> tuple[bool, str]:
    if not text.strip():
        return True, "empty_answer"
    has_close = "</think>" in text
    if require_think_close and not has_close:
        return True, "missing_think_close"
    if not require_think_close and not has_close and ("<think>" in text or THINKY_PREFIX_RE.search(text)):
        return True, "looks_like_unclosed_thinking"
    if filter_mode == "missing_or_loop":
        final_text = extract_final_response_text(text)
        repeat = max_repeated_ngram_count(word_tokens(final_text))
        if repeat >= loop_threshold:
            return True, f"loop_like_repeat5_{repeat}"
    return False, ""


def load_amber_annotations(amber_root: Path) -> dict[int, dict[str, Any]]:
    annotation_path = amber_root / "data" / "annotations.json"
    data = read_json(annotation_path)
    out = {}
    for idx, item in enumerate(data, start=1):
        if isinstance(item, dict):
            out[int(item.get("id", idx))] = item
    return out


def is_amber_generative(record: dict[str, Any], annotations: dict[int, dict[str, Any]]) -> bool:
    try:
        item_id = int(record.get("id"))
    except (TypeError, ValueError):
        return not bool(record.get("is_discriminative"))
    annotation = annotations.get(item_id)
    if annotation:
        return annotation.get("type") == "generative"
    return item_id < 1005 or not bool(record.get("is_discriminative"))


def amber_official_response(record: dict[str, Any], generative: bool) -> str:
    raw = audit_text_for_amber(record)
    if generative:
        return str(record.get("official_response") or extract_final_response_text(raw))
    return str(record.get("official_response") or extract_yes_no(raw))


def collect_bad_chair_ids(
    paths: list[Path],
    args: argparse.Namespace,
) -> tuple[set[int], dict[str, Any]]:
    bad_ids: set[int] = set()
    per_path = {}
    for path in paths:
        rows = read_jsonl(path)
        require_close = should_require_think_close(rows, audit_text_for_chair, args.thinking_detection)
        path_bad = []
        for row in rows:
            image_id = row.get("image_id")
            if image_id is None:
                continue
            text = audit_text_for_chair(row)
            bad, reason = is_bad_text(text, require_close, args.filter_mode, args.loop_repeat_threshold)
            if bad:
                image_id_int = int(image_id)
                bad_ids.add(image_id_int)
                path_bad.append({"image_id": image_id_int, "reason": reason})
        per_path[str(path)] = path_bad
    return bad_ids, {"bad_ids": sorted(bad_ids), "per_path": per_path}


def collect_bad_amber_ids(
    paths: list[Path],
    annotations: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[set[int], dict[str, Any]]:
    bad_ids: set[int] = set()
    per_path = {}
    for path in paths:
        rows = read_jsonl(path)
        require_close = should_require_think_close(rows, audit_text_for_amber, args.thinking_detection)
        path_bad = []
        for row in rows:
            item_id = int(row["id"])
            generative = is_amber_generative(row, annotations)
            should_check = args.amber_filter_scope == "all" or generative
            if not should_check:
                continue
            text = audit_text_for_amber(row)
            bad, reason = is_bad_text(text, require_close, args.filter_mode, args.loop_repeat_threshold)
            if bad:
                bad_ids.add(item_id)
                path_bad.append({"id": item_id, "reason": reason, "is_generative": generative})
        per_path[str(path)] = path_bad
    return bad_ids, {"bad_ids": sorted(bad_ids), "per_path": per_path}


def recompute_chair(
    label: str,
    chair_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    forced_drop_ids: set[int] | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(chair_path)
    require_close = should_require_think_close(rows, audit_text_for_chair, args.thinking_detection)
    kept_rows = []
    dropped_rows = []
    for row in rows:
        image_id = row.get("image_id")
        text = audit_text_for_chair(row)
        reason = ""
        if image_id is None:
            reason = "missing_image_id"
        elif not row.get("generated_caption") or str(row.get("generated_caption")).startswith("[ERROR]"):
            reason = "empty_or_error_caption"
        elif forced_drop_ids is not None and int(image_id) in forced_drop_ids:
            reason = "union_forced_drop"
        else:
            bad, reason = is_bad_text(text, require_close, args.filter_mode, args.loop_repeat_threshold)
            if not bad:
                reason = ""
        if reason:
            dropped_rows.append(
                {
                    "image_id": image_id,
                    "reason": reason,
                    "generated_caption_prefix": str(row.get("generated_caption", ""))[:300],
                    "raw_generated_caption_prefix": str(row.get("raw_generated_caption", ""))[:300],
                }
            )
        else:
            kept_rows.append(row)

    if not kept_rows:
        raise RuntimeError(f"CHAIR kept zero rows after filtering for {label}: {chair_path}")

    mscoco_objects, inverse_synonym_dict = parse_official_synonyms()
    double_word_dict = build_double_word_dict()
    eval_records = [
        {
            "image_id": row["image_id"],
            "generated_text": extract_final_response_text(str(row.get("generated_caption", ""))),
            "gt_objects": set(row.get("gt_objects", [])),
        }
        for row in kept_rows
    ]
    sample_dicts = compute_per_sample(
        eval_records,
        mscoco_objects,
        inverse_synonym_dict,
        double_word_dict,
    )
    metrics = aggregate_from_arrays(sample_dicts, list(sample_dicts.keys()))
    metrics.update(
        {
            "num_samples": len(kept_rows),
            "raw_n": len(rows),
            "dropped_n": len(dropped_rows),
            "require_think_close": require_close,
            "source_path": str(chair_path),
        }
    )

    chair_dir = out_dir / "chair"
    write_jsonl(chair_dir / "filtered_eval_results.jsonl", kept_rows)
    write_jsonl(chair_dir / "dropped_rows.jsonl", dropped_rows)
    write_json(chair_dir / "chair_metrics.json", metrics)
    return {
        "label": label,
        "task": "chair",
        "raw_n": len(rows),
        "kept_n": len(kept_rows),
        "dropped_n": len(dropped_rows),
        "require_think_close": require_close,
        "metrics": metrics,
        "source_path": str(chair_path),
        "filtered_results_path": str(chair_dir / "filtered_eval_results.jsonl"),
        "dropped_path": str(chair_dir / "dropped_rows.jsonl"),
        "metrics_path": str(chair_dir / "chair_metrics.json"),
    }


def build_amber_filtered_response_file(
    label: str,
    amber_path: Path,
    out_dir: Path,
    annotations: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    forced_drop_ids: set[int] | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(amber_path)
    require_close = should_require_think_close(rows, audit_text_for_amber, args.thinking_detection)
    kept = []
    dropped = []
    for row in rows:
        item_id = int(row["id"])
        generative = is_amber_generative(row, annotations)
        should_check = args.amber_filter_scope == "all" or generative
        reason = ""
        if forced_drop_ids is not None and item_id in forced_drop_ids and should_check:
            reason = "union_forced_drop"
        elif should_check:
            bad, reason = is_bad_text(
                audit_text_for_amber(row),
                require_close,
                args.filter_mode,
                args.loop_repeat_threshold,
            )
            if not bad:
                reason = ""
        if reason:
            dropped.append(
                {
                    "id": item_id,
                    "image": row.get("image"),
                    "is_generative": generative,
                    "reason": reason,
                    "model_answer_prefix": audit_text_for_amber(row)[:300],
                }
            )
            continue
        kept.append({"id": item_id, "response": amber_official_response(row, generative)})

    if not kept:
        raise RuntimeError(f"AMBER kept zero rows after filtering for {label}: {amber_path}")

    amber_dir = out_dir / "amber"
    response_path = amber_dir / "amber_filtered_responses.json"
    dropped_path = amber_dir / "dropped_rows.jsonl"
    write_json(response_path, sorted(kept, key=lambda x: int(x["id"])))
    write_jsonl(dropped_path, dropped)
    return {
        "label": label,
        "task": "amber",
        "raw_n": len(rows),
        "kept_n": len(kept),
        "dropped_n": len(dropped),
        "dropped_generative_n": sum(1 for row in dropped if row["is_generative"]),
        "dropped_discriminative_n": sum(1 for row in dropped if not row["is_generative"]),
        "require_think_close": require_close,
        "source_path": str(amber_path),
        "response_path": str(response_path),
        "dropped_path": str(dropped_path),
    }


def run_amber_eval(response_path: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    scorer = RES_OPD_ROOT / "scripts" / "run_amber_parallel.py"
    amber_root = Path(args.amber_root)
    cmd = [
        sys.executable,
        str(scorer),
        "--inference_data",
        str(response_path),
        "--evaluation_type",
        args.amber_evaluation_type,
        "--workers",
        str(args.amber_workers),
        "--amber_root",
        str(amber_root),
        "--output_metrics_json",
        str(out_dir / "amber" / "amber_metrics_raw_counts.json"),
        "--word_association",
        str(amber_root / "data" / "relation.json"),
        "--safe_words",
        str(amber_root / "data" / "safe_words.txt"),
        "--annotation",
        str(amber_root / "data" / "annotations.json"),
        "--metrics",
        str(amber_root / "data" / "metrics.txt"),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    amber_dir = out_dir / "amber"
    official_log = amber_dir / "official_eval.log"
    official_log.write_text(proc.stdout, encoding="utf-8")
    return {
        "official_returncode": proc.returncode,
        "official_log": str(official_log),
        "metrics": parse_official_stdout(proc.stdout),
        "metrics_path": str(amber_dir / "amber_metrics_raw_counts.json"),
    }


def recompute_amber(
    label: str,
    amber_path: Path,
    out_dir: Path,
    annotations: dict[int, dict[str, Any]],
    args: argparse.Namespace,
    forced_drop_ids: set[int] | None = None,
) -> dict[str, Any]:
    summary = build_amber_filtered_response_file(
        label=label,
        amber_path=amber_path,
        out_dir=out_dir,
        annotations=annotations,
        args=args,
        forced_drop_ids=forced_drop_ids,
    )
    eval_summary = run_amber_eval(Path(summary["response_path"]), out_dir, args)
    summary.update(eval_summary)
    if summary["official_returncode"] != 0:
        raise RuntimeError(f"AMBER filtered eval failed for {label}. See {summary['official_log']}")
    write_json(out_dir / "amber" / "summary.json", summary)
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def get_metric(summary: dict[str, Any] | None, key: str) -> Any:
    if not summary:
        return None
    return summary.get("metrics", {}).get(key)


def write_summary_md(path: Path, model_summaries: list[dict[str, Any]]) -> None:
    header = [
        "Model",
        "CHAIR raw/kept/drop",
        "CHAIRi",
        "CHAIRs",
        "ObjF1",
        "AMBER raw/kept/drop",
        "Gen CHAIR",
        "Gen HAL",
        "Gen Cover",
        "Disc F1",
    ]
    lines = [
        "# Filtered CHAIR + AMBER Metrics",
        "",
        "Metrics are recomputed after filtering contaminated thinking rows. Original eval outputs are not modified.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for item in model_summaries:
        chair = item.get("chair")
        amber = item.get("amber")
        chair_counts = "n/a"
        amber_counts = "n/a"
        if chair:
            chair_counts = f"{chair['raw_n']}/{chair['kept_n']}/{chair['dropped_n']}"
        if amber:
            amber_counts = f"{amber['raw_n']}/{amber['kept_n']}/{amber['dropped_n']}"
        row = [
            item["label"],
            chair_counts,
            fmt(get_metric(chair, "CHAIRi")),
            fmt(get_metric(chair, "CHAIRs")),
            fmt(get_metric(chair, "ObjF1")),
            amber_counts,
            fmt(get_metric(amber, "generative_task_chair")),
            fmt(get_metric(amber, "generative_task_hal")),
            fmt(get_metric(amber, "generative_task_cover")),
            fmt(get_metric(amber, "descriminative_task_f1")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    tasks = parse_tasks(args.tasks)
    inputs = parse_labeled_inputs(args.inputs, tasks)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    annotations = load_amber_annotations(Path(args.amber_root)) if "amber" in tasks else {}
    forced_chair_ids = None
    forced_amber_ids = None
    union_payload: dict[str, Any] = {}
    if args.drop_id_mode == "union":
        if "chair" in tasks:
            forced_chair_ids, union_payload["chair"] = collect_bad_chair_ids(
                [item["chair_path"] for item in inputs],
                args,
            )
        if "amber" in tasks:
            forced_amber_ids, union_payload["amber"] = collect_bad_amber_ids(
                [item["amber_path"] for item in inputs],
                annotations,
                args,
            )
        write_json(output_root / "union_drop_ids.json", union_payload)

    model_summaries = []
    for item in inputs:
        label = item["label"]
        label_dir = output_root / label
        model_summary: dict[str, Any] = {"label": label, "input_path": item["input_path"]}
        if "chair" in tasks:
            chair_summary = recompute_chair(
                label,
                item["chair_path"],
                label_dir,
                args,
                forced_drop_ids=forced_chair_ids,
            )
            model_summary["chair"] = chair_summary
        if "amber" in tasks:
            amber_summary = recompute_amber(
                label,
                item["amber_path"],
                label_dir,
                annotations,
                args,
                forced_drop_ids=forced_amber_ids,
            )
            model_summary["amber"] = amber_summary
        write_json(label_dir / "summary.json", model_summary)
        model_summaries.append(model_summary)

    payload = {
        "tasks": sorted(tasks),
        "amber_root": args.amber_root,
        "amber_evaluation_type": args.amber_evaluation_type,
        "amber_filter_scope": args.amber_filter_scope if "amber" in tasks else None,
        "filter_mode": args.filter_mode,
        "thinking_detection": args.thinking_detection,
        "drop_id_mode": args.drop_id_mode,
        "loop_repeat_threshold": args.loop_repeat_threshold,
        "summaries": model_summaries,
    }
    write_json(output_root / "summary.json", payload)
    write_summary_md(output_root / "summary.md", model_summaries)
    print(f"Wrote: {output_root / 'summary.json'}")
    print(f"Wrote: {output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
