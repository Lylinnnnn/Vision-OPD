#!/usr/bin/env python3
"""Summarize the three paper controls from evaluated checkpoints only."""

import argparse
import json
import re
from pathlib import Path


SCHEMA = "official_chair_caption_and_mention_v1"
PROMPT = "Describe the image."
RATIOS = ("0.25", "0.5", "0.75")
SUFFIXES = (
    "three_tables_fill_official_v2", "chair_native_sweep_official_v2", "chair_official_v2",
    "chair_describe_image_official", "native_base_official_v2", "official_v2",
)
STEP_RE = re.compile(r"^global_step_(\d+)$")
WORD_RE = re.compile(r"\b\w+(?:[-']\w+)*\b")


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_judge_rows(path):
    try:
        rows = read_json(path)
    except json.JSONDecodeError:
        rows = read_jsonl(path)
    return rows if isinstance(rows, list) else []


def final_text(row):
    text = str(row.get("generated_caption") or "")
    return text.rsplit("</think>", 1)[-1].strip()


def ratio_tag(value):
    return value.replace(".", "p")


def experiment_name(scale, ratio, rho=None, frozen=False):
    batch = "b32" if scale == "2b" else "b16"
    prefix = (f"Res-OPD-Qwen3VL-{scale.upper()}-Instruct-orig-sr1.0-"
              f"tr{ratio}-a1.0-frozen-rkl")
    if frozen:
        return f"{prefix}-{batch}-rn4-full5k-e1"
    return f"{prefix}-riskmask-nll-p{rho}-{batch}-rn4-full5k-e1"


def datasets_for(eval_root, scale, experiment, student_ratio="1.0"):
    family = eval_root / f"Qwen3VL-{scale.upper()}-Instruct" if experiment is None else eval_root / experiment
    dataset = family / f"train5000_test1000_original_sr{ratio_tag(student_ratio)}"
    return dataset


def result_dirs(dataset):
    if not dataset.is_dir():
        return []
    children = [path for path in dataset.iterdir() if path.is_dir() and path.name != ".shards"]
    rank = {name: index for index, name in enumerate(SUFFIXES)}
    children.sort(key=lambda path: (rank.get(path.name, len(rank)), path.name))
    return children + [dataset]


def valid_chair(directory, expected_ratio, samples):
    metrics_path = directory / "chair_metrics.json"
    results_path = directory / "eval_results.jsonl"
    if not metrics_path.is_file() or not results_path.is_file():
        return None
    metrics = read_json(metrics_path)
    if (metrics.get("metric_schema") != SCHEMA or metrics.get("eval_prompt") != PROMPT
            or metrics.get("total_captions") != samples
            or metrics.get("total_results") != samples
            or metrics.get("invalid_final_answer", 0) != 0):
        return None
    records = read_jsonl(results_path)
    if len(records) != samples or len({row.get("image_id") for row in records}) != samples:
        return None
    if any(row.get("eval_prompt") not in (None, PROMPT)
           or float(row.get("eval_student_ratio", expected_ratio)) != float(expected_ratio)
           for row in records):
        return None
    correct = int(metrics["total_correct_mentions"])
    hall = int(metrics["total_hallucinated_mentions"])
    total = int(metrics["total_object_mentions"])
    if correct + hall != total or abs(hall / max(total, 1) - metrics["CHAIRi"]) > 1e-9:
        return None
    if abs(metrics["hallucinated_captions"] / samples - metrics["CHAIRs"]) > 1e-9:
        return None
    lengths = [final_text(row) for row in records]
    if any(not text for text in lengths):
        return None
    recall = metrics.get("ObjRecall")
    unique_gt = None
    if isinstance(recall, (int, float)) and recall > 0:
        unique_gt = round(metrics["total_correct"] / recall)
        if abs(metrics["total_correct"] / unique_gt - recall) > 1e-8:
            unique_gt = None
    return {
        "CHAIRi": metrics["CHAIRi"], "CHAIRs": metrics["CHAIRs"],
        "ObjF1": metrics["ObjF1"],
        "response_words_mean": sum(len(WORD_RE.findall(text)) for text in lengths) / samples,
        "response_chars_mean": sum(len(text) for text in lengths) / samples,
        "gt_correct_mentions": correct, "hall_mentions": hall,
        "unique_gt_categories": unique_gt,
        "source": str(metrics_path),
    }


def judge_accuracy(directory, benchmark):
    paths = sorted(directory.glob(f"{benchmark}/judge/*_answer.jsonl"))
    paths += sorted(directory.glob(f"vision_opd/judge/{benchmark}/*_answer.jsonl"))
    if len(paths) != 1:
        return None
    records = read_judge_rows(paths[0])
    if not records or any(str(row.get("judge", "")).strip().lower() not in ("yes", "no") for row in records):
        return None
    if benchmark == "cv-bench":
        breakdown = paths[0].parent.parent / "cvbench_breakdown.json"
        if breakdown.is_file():
            value = read_json(breakdown).get("accuracy")
            if isinstance(value, (int, float)):
                return value, str(breakdown)
    return sum(row["judge"].strip().lower() == "yes" for row in records) / len(records), str(paths[0])


def auxiliary(directories, issues):
    values = {}
    sources = {}
    for directory in directories:
        pope_path = directory / "pope/pope_summary.json"
        if "POPE" not in values and pope_path.is_file():
            try:
                pope = read_json(pope_path)
                if all(name in pope.get("benchmarks", {}) for name in ("pope_adv", "pope_pop", "pope_random")):
                    value = pope.get("macro_f1")
                    if isinstance(value, (int, float)):
                        values["POPE"] = value
                        sources["POPE"] = str(pope_path)
            except (OSError, ValueError, TypeError) as error:
                issues.append(f"{pope_path}: {error}")
        amber_path = directory / "amber/amber_metrics.json"
        if "A-CHAIR" not in values and amber_path.is_file():
            try:
                amber = read_json(amber_path)
                if amber.get("official_returncode") == 0 and amber.get("evaluation_type") == "a":
                    metrics = amber.get("metrics") or {}
                    for key, field in (("A-CHAIR", "generative_task_chair"), ("A-HAL", "generative_task_hal")):
                        value = metrics.get(field)
                        if isinstance(value, (int, float)):
                            values[key] = value
                            sources[key] = str(amber_path)
            except (OSError, ValueError, TypeError) as error:
                issues.append(f"{amber_path}: {error}")
        for benchmark, key in (("mmstar", "MMStar"), ("cv-bench", "CV-Bench")):
            if key not in values:
                try:
                    result = judge_accuracy(directory, benchmark)
                    if result:
                        values[key], sources[key] = result
                except (OSError, ValueError, TypeError) as error:
                    issues.append(f"{directory}/{benchmark}: {error}")
    return values, sources


def summarize_dataset(dataset, student_ratio, samples, issues):
    directories = result_dirs(dataset)
    chair = None
    for directory in directories:
        try:
            chair = valid_chair(directory, student_ratio, samples)
        except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError) as error:
            issues.append(f"{directory}: invalid CHAIR: {error}")
        if chair:
            break
    aux, sources = auxiliary(directories, issues)
    if not chair:
        issues.append(f"Missing valid official CHAIR: {dataset}")
    return chair or {}, aux, sources


def collect(eval_root, samples=1000):
    rows, issues = [], []

    def add(table, scale, method, ratio, rho=None, frozen=False):
        if method in ("Native Base", "Base"):
            dataset = datasets_for(eval_root, scale, None, ratio if method == "Native Base" else "1.0")
            candidates = [(None, dataset)]
        else:
            name = experiment_name(scale, ratio, rho=rho, frozen=frozen)
            candidates = []
            for path in sorted(eval_root.glob(f"{name}_global_step_*")):
                match = STEP_RE.fullmatch(path.name.removeprefix(f"{name}_"))
                if path.is_dir() and match:
                    candidates.append((int(match.group(1)), datasets_for(eval_root, scale, path.name)))
            candidates.sort(key=lambda item: item[0])
            if not candidates:
                issues.append(f"No evaluated checkpoint directory: {name}")
                candidates = [(None, datasets_for(eval_root, scale, f"{name}_global_step_*"))]
        for step, dataset in candidates:
            chair, aux, sources = summarize_dataset(dataset, "1.0" if method != "Native Base" else ratio, samples, issues)
            rows.append({"table": table, "size": scale.upper(), "method": method, "ratio": ratio,
                         "rho": rho, "step": step, "chair": chair, "aux": aux,
                         "sources": {"CHAIR": chair.get("source"), **sources},
                         "dataset": str(dataset)})

    for scale in ("2b", "8b"):
        for ratio in RATIOS:
            add("native", scale, "Native Base", ratio)
            add("native", scale, "Res-OPSD", ratio, rho=30)
    add("teacher", "2b", "Base", "1.0")
    add("teacher", "2b", "Blank Teacher", "0.0", rho=30)
    for ratio in (*RATIOS, "1.0"):
        add("teacher", "2b", "Res-OPSD", ratio, rho=30)
    add("rho", "2b", "Frozen RKL", "0.5", frozen=True)
    for rho in (10, 20, 30, 40):
        add("rho", "2b", "Res-OPSD", "0.5", rho=rho)
    return rows, issues


def fmt(value, percent=False, digits=2):
    if value is None:
        return "--"
    if isinstance(value, int) and not percent:
        return str(value)
    return f"{value * (100 if percent else 1):.{digits}f}"


def markdown(rows, issues):
    lines = [
        "# Three paper tables: refreshed evaluation inventory", "",
        "All available evaluated steps are shown; no best/worst checkpoint selection. "
        "CHAIR is official caption/mention scoring with `Describe the image.` on 1,000 images. "
        "Response length is computed directly from final `generated_caption`: English words/image "
        "and characters/image, excluding any `<think>` prefix, not tokenizer tokens. "
        "GT correct mentions and hallucinated mentions are absolute counts across 1,000 captions, "
        "including repeated object mentions. Unique GT categories is a separate reference count. "
        "CHAIR/POPE/MMStar/CV-Bench are percentages; AMBER retains reported points. "
        "`--` means no verified result, not zero. See source index for per-metric provenance.", "",
    ]
    specs = (
        ("native", "Table 2 - Native compressed-view controls", "Ratio is sr for Native Base and tr for Res-OPSD.",
         ("Size", "Method", "Ratio", "Step", "CHAIRi", "CHAIRs", "ObjF1", "Response chars/img", "Words/img", "GT correct mentions", "Hall mentions", "Unique GT", "POPE", "A-CHAIR", "A-HAL", "ID")),
        ("teacher", "Table 3 - Teacher-ratio ablation (Qwen3VL-2B, p=0.30)", "The Blank Teacher row has tr=0.00.",
         ("Method", "Ratio", "Step", "CHAIRi", "CHAIRs", "ObjF1", "Response chars/img", "Words/img", "GT correct mentions", "Hall mentions", "Unique GT", "A-CHAIR", "A-HAL", "MMStar", "CV-Bench", "ID")),
        ("rho", "Table 4 - Top-rho sensitivity (Qwen3VL-2B, tr=0.50)", "Kept=1.00 for Frozen RKL.",
         ("Method", "Kept", "Step", "CHAIRi", "CHAIRs", "ObjF1", "Response chars/img", "Words/img", "GT correct mentions", "Hall mentions", "Unique GT", "POPE", "A-CHAIR", "A-HAL", "MMStar", "CV-Bench", "ID")),
    )
    index = 0
    indexed = []
    for table, title, note, columns in specs:
        lines += [f"## {title}", "", note, "", "| " + " | ".join(columns) + " |",
                  "| " + " | ".join("---:" if column not in ("Size", "Method", "ID") else "---" for column in columns) + " |"]
        for row in (item for item in rows if item["table"] == table):
            index += 1
            row["id"] = f"T{index:03d}"
            indexed.append(row)
            chair, aux = row["chair"], row["aux"]
            cells = {
                "Size": row["size"], "Method": row["method"],
                "Ratio": "--" if row["method"] == "Base" else f"{float(row['ratio']):.2f}",
                "Kept": ("1.00" if row["method"] == "Frozen RKL" else
                         f"{row['rho'] / 100:.2f}" if row["rho"] is not None else "--"),
                "Step": fmt(row["step"]), "CHAIRi": fmt(chair.get("CHAIRi"), True),
                "CHAIRs": fmt(chair.get("CHAIRs"), True), "ObjF1": fmt(chair.get("ObjF1"), True),
                "Words/img": fmt(chair.get("response_words_mean")),
                "Response chars/img": fmt(chair.get("response_chars_mean")),
                "GT correct mentions": fmt(chair.get("gt_correct_mentions")),
                "Hall mentions": fmt(chair.get("hall_mentions")),
                "Unique GT": fmt(chair.get("unique_gt_categories")),
                "POPE": fmt(aux.get("POPE"), True),
                "A-CHAIR": fmt(aux.get("A-CHAIR")), "A-HAL": fmt(aux.get("A-HAL")),
                "MMStar": fmt(aux.get("MMStar"), True),
                "CV-Bench": fmt(aux.get("CV-Bench"), True), "ID": row["id"],
            }
            lines.append("| " + " | ".join(cells[column] for column in columns) + " |")
        lines.append("")
    lines += ["## Source index", "", "Each row is one checkpoint or one Native/Base inference ratio. "
              "Metric paths may differ because CHAIR was re-evaluated separately; they must share the exact experiment directory.", "",
              "| ID | Experiment dataset | CHAIR | POPE | AMBER | MMStar | CV-Bench |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in indexed:
        sources = row["sources"]
        lines.append("| " + " | ".join([row["id"], f"`{row['dataset']}`"] +
                     [f"`{sources[key]}`" if sources.get(key) else "--"
                      for key in ("CHAIR", "POPE", "A-CHAIR", "MMStar", "CV-Bench")]) + " |")
    if issues:
        lines += ["", "## Missing or invalid", ""]
        lines += [f"- {issue}" for issue in issues]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("res-opd/eval_results/instruct/full"))
    parser.add_argument("--output-md", type=Path, default=Path("res-opd/paper/tmp/three_tables_official_v2_summary.md"))
    parser.add_argument("--expected-samples", type=int, default=1000)
    args = parser.parse_args()
    rows, issues = collect(args.eval_root, args.expected_samples)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown(rows, issues), encoding="utf-8")
    complete_chair = sum(bool(row["chair"]) for row in rows)
    print(f"THREE_TABLES_SUMMARY=WRITTEN rows={len(rows)} official_chair={complete_chair} "
          f"missing_or_invalid={len(issues)} md={args.output_md}")


if __name__ == "__main__":
    main()
