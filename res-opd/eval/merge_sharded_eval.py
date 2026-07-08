#!/usr/bin/env python3
"""Merge deterministic eval shards and recompute benchmark metrics."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[1]
VISION_EVAL_DIR = REPO_ROOT / "eval"
sys.path.insert(0, str(EVAL_DIR))

from robust_chair_analysis import (  # noqa: E402
    aggregate_from_arrays,
    build_double_word_dict,
    compute_per_sample,
    parse_official_synonyms,
)
from eval_chair import extract_final_response_text as chair_final_text  # noqa: E402
from eval_chair import record_final_answer_available  # noqa: E402
from eval_pope import compute_metrics as pope_compute_metrics  # noqa: E402
from eval_pope import parse_benchmarks as parse_pope_benchmarks  # noqa: E402
from eval_amber import parse_official_stdout  # noqa: E402


VISION_BENCHMARK_JSON_MAP = {
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


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prefer_record(old, new, answer_field):
    if old is None:
        return new
    old_answer = str(old.get(answer_field, "") or "")
    new_answer = str(new.get(answer_field, "") or "")
    old_bad = not old_answer or old_answer.startswith("[ERROR]") or old_answer.startswith("[API_ERROR]")
    new_bad = not new_answer or new_answer.startswith("[ERROR]") or new_answer.startswith("[API_ERROR]")
    if old_bad and not new_bad:
        return new
    return new if old_bad == new_bad else old


def dedupe_jsonl(paths, key_fn, answer_field):
    merged = {}
    order = []
    for path in paths:
        for row in read_jsonl(path):
            key = key_fn(row)
            if key is None:
                continue
            if key not in merged:
                order.append(key)
                merged[key] = row
            else:
                merged[key] = prefer_record(merged[key], row, answer_field)
    return [merged[key] for key in order]


def shard_out_dirs(shard_root: Path):
    dirs = sorted(path / "out" for path in shard_root.glob("shard_*") if (path / "out").exists())
    if not dirs:
        raise FileNotFoundError(f"No shard output dirs found under {shard_root}")
    return dirs


def normalize_eval_mode(mode: str):
    out = []
    for raw in (mode or "").lower().replace(" ", "").split(","):
        if raw in {"", "none"}:
            continue
        if raw == "all":
            out.extend(["chair", "pope", "vision"])
        elif raw in {"frequent", "coco"}:
            out.extend(["chair", "pope"])
        elif raw in {"vision-opd", "aux", "auxiliary"}:
            out.append("vision")
        elif raw in {"cvbench", "cv_bench"}:
            out.append("cv-bench")
        else:
            out.append(raw)
    deduped = []
    for item in out:
        if item not in deduped:
            deduped.append(item)
    return deduped


def has_task(tasks, name):
    return name in tasks


def vision_benchmarks(tasks, explicit):
    benches = []
    if "vision" in tasks:
        benches.extend([x.strip() for x in explicit.split(",") if x.strip()])
    if "mmstar" in tasks and "mmstar" not in benches:
        benches.append("mmstar")
    if "cv-bench" in tasks and "cv-bench" not in benches:
        benches.append("cv-bench")
    return benches


def merge_chair(args, shard_dirs):
    rows = dedupe_jsonl(
        [path / "eval_results.jsonl" for path in shard_dirs],
        lambda row: row.get("image_id"),
        "generated_caption",
    )
    if not rows:
        raise RuntimeError("CHAIR merge found zero rows across all shards.")
    final_path = args.final_output_dir / "eval_results.jsonl"
    write_jsonl(final_path, rows)

    strict_final_answer = any(
        row.get("raw_generated_caption") is not None
        or row.get("thinking_status") is not None
        or row.get("final_answer_available") is not None
        for row in rows
    )
    candidate_rows = [
        row for row in rows
        if row.get("image_id") is not None
        and row.get("generated_caption")
        and not str(row.get("generated_caption")).startswith("[ERROR]")
    ]
    scored_rows = [
        row for row in candidate_rows
        if record_final_answer_available(row, strict_final_answer)
    ]
    invalid_rows = [
        row for row in candidate_rows
        if not record_final_answer_available(row, strict_final_answer)
    ]
    invalid_final_answer = len(invalid_rows)
    unclosed_thinking = len(
        [
            row for row in invalid_rows
            if row.get("thinking_status") == "unclosed"
            or "</think>" not in str(row.get("raw_generated_caption") or row.get("generated_caption") or "")
        ]
    )
    mscoco_objects, inverse_synonym_dict = parse_official_synonyms()
    double_word_dict = build_double_word_dict()
    eval_records = [
        {
            "image_id": row["image_id"],
            "generated_text": chair_final_text(row["generated_caption"]),
            "gt_objects": set(row.get("gt_objects", [])),
        }
        for row in scored_rows
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
            "num_samples": len(scored_rows),
            "total_results": len(rows),
            "invalid_final_answer": invalid_final_answer,
            "unclosed_thinking": unclosed_thinking,
            "final_answer_valid_rate": len(scored_rows) / len(rows) if rows else 0.0,
            "strict_final_answer": strict_final_answer,
            "merged_from_shards": True,
            "shard_count": len(shard_dirs),
        }
    )
    with open(args.final_output_dir / "chair_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"MERGE CHAIR: rows={len(rows)} scored={len(scored_rows)} -> {final_path}")


def merge_pope(args, shard_dirs):
    all_metrics = {}
    for benchmark in parse_pope_benchmarks(args.pope_benchmark):
        rows = dedupe_jsonl(
            [path / "pope" / benchmark / "eval_results.jsonl" for path in shard_dirs],
            lambda row: row.get("sample_uid") or row.get("id") or row.get("question_id"),
            "model_answer",
        )
        if not rows:
            raise RuntimeError(f"POPE merge found zero rows for benchmark={benchmark}.")
        bench_dir = args.final_output_dir / "pope" / benchmark
        result_path = bench_dir / "eval_results.jsonl"
        write_jsonl(result_path, rows)
        metrics = pope_compute_metrics(rows)
        metrics.update(
            {
                "benchmark": benchmark,
                "pope_source": args.pope_source,
                "merged_from_shards": True,
                "shard_count": len(shard_dirs),
            }
        )
        with open(bench_dir / "pope_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        all_metrics[benchmark] = metrics
        print(f"MERGE POPE {benchmark}: rows={len(rows)}")

    macro_f1 = sum(m["f1"] for m in all_metrics.values()) / len(all_metrics) if all_metrics else 0.0
    macro_acc = sum(m["accuracy"] for m in all_metrics.values()) / len(all_metrics) if all_metrics else 0.0
    summary = {"benchmarks": all_metrics, "macro_accuracy": macro_acc, "macro_f1": macro_f1}
    summary_path = args.final_output_dir / "pope" / "pope_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def merge_amber(args, shard_dirs):
    out_dir = args.final_output_dir / "amber"
    out_dir.mkdir(parents=True, exist_ok=True)
    response_name = f"amber_{args.amber_eval_type}_responses.json"
    by_id = {}
    for path in [shard / "amber" / response_name for shard in shard_dirs]:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for row in json.load(f):
                by_id[int(row["id"])] = row
    response_path = out_dir / response_name
    if not by_id:
        raise RuntimeError("AMBER merge found zero official response rows across all shards.")
    with open(response_path, "w", encoding="utf-8") as f:
        json.dump([by_id[key] for key in sorted(by_id)], f, ensure_ascii=False, indent=2)

    raw_rows = dedupe_jsonl(
        [shard / "amber" / "raw_results.jsonl" for shard in shard_dirs],
        lambda row: row.get("id"),
        "model_answer",
    )
    write_jsonl(out_dir / "raw_results.jsonl", raw_rows)
    print(f"MERGE AMBER: official_rows={len(by_id)} raw_rows={len(raw_rows)}")
    if args.amber_skip_official_eval:
        return

    parallel_evaluator = REPO_ROOT / "res-opd" / "scripts" / "run_amber_parallel.py"
    use_parallel = args.amber_official_eval_workers > 1 and parallel_evaluator.exists()
    if use_parallel:
        cmd = [
            sys.executable,
            str(parallel_evaluator),
            "--inference_data",
            str(response_path),
            "--evaluation_type",
            args.amber_eval_type,
            "--workers",
            str(args.amber_official_eval_workers),
            "--amber_root",
            str(args.amber_root),
            "--output_metrics_json",
            str(out_dir / "amber_metrics_raw_counts.json"),
        ]
    else:
        evaluator = args.amber_root / "inference.py"
        if not evaluator.exists():
            raise FileNotFoundError(f"AMBER official inference.py not found: {evaluator}")
        cmd = [
            sys.executable,
            str(evaluator),
            "--inference_data",
            str(response_path),
            "--evaluation_type",
            args.amber_eval_type,
        ]
    cmd.extend(
        [
            "--word_association",
            str(args.amber_word_association or (args.amber_root / "data" / "relation.json")),
            "--safe_words",
            str(args.amber_safe_words or (args.amber_root / "data" / "safe_words.txt")),
            "--annotation",
            str(args.amber_annotation or (args.amber_root / "data" / "annotations.json")),
            "--metrics",
            str(args.amber_metrics or (args.amber_root / "data" / "metrics.txt")),
        ]
    )
    proc = subprocess.run(
        cmd,
        cwd=str(args.amber_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    official_log = out_dir / "official_eval.log"
    official_log.write_text(proc.stdout, encoding="utf-8")
    summary = {
        "benchmark": "amber",
        "evaluation_type": args.amber_eval_type,
        "official_eval_mode": "parallel" if use_parallel else "official",
        "official_eval_workers": args.amber_official_eval_workers if use_parallel else 1,
        "official_returncode": proc.returncode,
        "official_log": str(official_log),
        "response_path": str(response_path),
        "metrics": parse_official_stdout(proc.stdout),
        "merged_from_shards": True,
        "shard_count": len(shard_dirs),
    }
    with open(out_dir / "amber_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if proc.returncode != 0:
        raise RuntimeError(f"AMBER evaluator failed. See {official_log}")


def vision_key(row):
    for key in ("sample_uid", "uid", "index", "question_id", "id"):
        value = row.get(key)
        if value is not None and str(value) != "":
            return f"{key}:{value}"
    images = row.get("images") or []
    image0 = images[0] if isinstance(images, list) and images else ""
    return json.dumps({"image": image0, "query": row.get("query", "")}, ensure_ascii=False, sort_keys=True)


def run_cmd(cmd, cwd=None):
    print("RUN:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def merge_vision(args, shard_dirs, tasks):
    benches = vision_benchmarks(tasks, args.vision_benchmark)
    if not benches:
        return
    model_tag = f"{args.experiment_name}_seed{args.seed}"
    for bench in benches:
        output_bench = f"{bench}{args.vision_benchmark_output_suffix}"
        answer_name = f"{model_tag}_answer.jsonl"
        rows = dedupe_jsonl(
            [shard / output_bench / "model_answer" / answer_name for shard in shard_dirs],
            vision_key,
            "model_answer",
        )
        if not rows:
            raise RuntimeError(f"Vision benchmark merge found zero rows for benchmark={bench}.")
        answer_dir = args.final_output_dir / output_bench / "model_answer"
        judge_dir = args.final_output_dir / output_bench / "judge"
        answer_path = answer_dir / answer_name
        write_jsonl(answer_path, rows)
        print(f"MERGE VISION {bench}: rows={len(rows)} -> {answer_path}")

        judge_cmd = [
            sys.executable,
            str(VISION_EVAL_DIR / "judge_qwenlm.py"),
            "--benchmark",
            bench,
            "--model",
            model_tag,
            "--answer_dir",
            str(answer_dir),
            "--judge_dir",
            str(judge_dir),
            "--mcq_extract_mode",
            args.mcq_extract_mode,
            "--no_benchmark_subdir",
        ]
        if args.rule_only_judge:
            judge_cmd.append("--rule_only")
        if args.judge_api_base:
            judge_cmd.extend(["--api_base", args.judge_api_base])
        if args.judge_api_key:
            judge_cmd.extend(["--api_key", args.judge_api_key])
        if args.judge_model:
            judge_cmd.extend(["--judge_model", args.judge_model])
        if args.judge_model_path:
            judge_cmd.extend(["--judge_model_path", args.judge_model_path])
        if args.judge_max_tokens:
            judge_cmd.extend(["--judge_max_tokens", str(args.judge_max_tokens)])
        run_cmd(judge_cmd, cwd=VISION_EVAL_DIR)

        benchmark_json = args.vision_benchmark_data_dir / VISION_BENCHMARK_JSON_MAP[bench]
        run_cmd(
            [
                sys.executable,
                str(VISION_EVAL_DIR / "cal_acc.py"),
                "--benchmark",
                bench,
                "--judge_json",
                str(judge_dir / answer_name),
                "--benchmark_json",
                str(benchmark_json),
            ],
            cwd=VISION_EVAL_DIR,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-output-dir", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--eval-mode", required=True)
    parser.add_argument("--pope-benchmark", default="pope_adv,pope_pop,pope_random")
    parser.add_argument("--pope-source", default="res-opd-test")
    parser.add_argument("--vision-benchmark", default="mmstar,cv-bench")
    parser.add_argument("--vision-benchmark-data-dir", type=Path, default=Path("/home/liuyanlin.lyl/notebook/data"))
    parser.add_argument("--vision-benchmark-output-suffix", default="")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seed", default="42")
    parser.add_argument("--rule-only-judge", action="store_true")
    parser.add_argument("--mcq-extract-mode", default="official")
    parser.add_argument("--judge-api-base", default="")
    parser.add_argument("--judge-api-key", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--judge-model-path", default="")
    parser.add_argument("--judge-max-tokens", default="2048")
    parser.add_argument("--amber-root", type=Path, default=Path("/home/liuyanlin.lyl/notebook/data/AMBER"))
    parser.add_argument("--amber-eval-type", default="a")
    parser.add_argument("--amber-skip-official-eval", action="store_true")
    parser.add_argument("--amber-official-eval-workers", type=int, default=1)
    parser.add_argument("--amber-word-association", type=Path, default=None)
    parser.add_argument("--amber-safe-words", type=Path, default=None)
    parser.add_argument("--amber-annotation", type=Path, default=None)
    parser.add_argument("--amber-metrics", type=Path, default=None)
    args = parser.parse_args()
    if args.amber_official_eval_workers <= 0:
        raise ValueError("--amber-official-eval-workers must be positive")

    tasks = normalize_eval_mode(args.eval_mode)
    shard_dirs = shard_out_dirs(args.shard_root)
    args.final_output_dir.mkdir(parents=True, exist_ok=True)

    if has_task(tasks, "chair"):
        merge_chair(args, shard_dirs)
    if has_task(tasks, "pope"):
        merge_pope(args, shard_dirs)
    if has_task(tasks, "amber"):
        merge_amber(args, shard_dirs)
    merge_vision(args, shard_dirs, tasks)

    run_cmd(
        [sys.executable, str(EVAL_DIR / "collect_benchmark_summary.py"), "--output-dir", str(args.final_output_dir)]
    )
    print(f"Merged sharded eval results into: {args.final_output_dir}")


if __name__ == "__main__":
    main()
