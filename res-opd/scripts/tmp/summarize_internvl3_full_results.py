#!/usr/bin/env python3
"""Summarize every evaluated InternVL3 checkpoint without selecting a step."""

import argparse
import json
import re
from pathlib import Path


STEP_RE = re.compile(r"^(.+)_global_step_(\d+)$")
POPE_SPLITS = ("pope_adv", "pope_pop", "pope_random")
OFFICIAL_CHAIR_SCHEMA = "official_chair_caption_and_mention_v1"
METHOD_ORDER = ("Base", "RiskMask", "Frozen RKL", "Low-res SFT", "RandomMask", "EntropyMask", "RiskMask (entropy)", "Other")
DEFAULT_BASE_DIR = Path("res-opd/eval_results/internvl/full/InternVL3-2B-hf/base_chair_original_1000_8gpu")


def load_json(path, issues):
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        issues.append(f"{path}: {exc}")
        return None


def load_judge(path, issues):
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows:
            raise ValueError("empty judge output")
        judged = sum(bool(str(row.get("judge", "")).strip()) for row in rows)
        accuracy = sum(str(row.get("judge", "")).strip().lower() == "yes" for row in rows) / len(rows)
        return {"accuracy": accuracy if judged == len(rows) else None,
                "rows": len(rows), "judged": judged}
    except (OSError, ValueError, TypeError) as exc:
        issues.append(f"{path}: {exc}")
        return None


def method_name(name):
    if "internvl3-2b-hf-tr05-fixedmb1-" in name.lower():
        return "RiskMask"
    for marker, label in (
        ("riskmask-entropy", "RiskMask (entropy)"),
        ("riskmask", "RiskMask"),
        ("randommask", "RandomMask"),
        ("entropymask", "EntropyMask"),
        ("sft-lowres", "Low-res SFT"),
        ("frozen-rkl", "Frozen RKL"),
    ):
        if marker in name.lower():
            return label
    return "Other"


def discover(root):
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluation root does not exist: {root}")
    for experiment in sorted(root.iterdir()):
        if not experiment.is_dir():
            continue
        match = STEP_RE.fullmatch(experiment.name)
        if not match:
            continue
        for dataset in sorted(child for child in experiment.iterdir() if child.is_dir()):
            suffixes = sorted(child for child in dataset.iterdir() if child.is_dir() and child.name != ".shards")
            for output in suffixes:
                yield match.group(1), int(match.group(2)), dataset.name, output


def collect(root, base_dir):
    rows = []
    issues = []
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Baseline result directory does not exist: {base_dir}")
    base_metrics = sorted(path for path in base_dir.rglob("chair_metrics.json") if ".shards" not in path.parts)
    if not base_metrics:
        raise FileNotFoundError(f"No baseline chair_metrics.json found under: {base_dir}")
    if len(base_metrics) != 1:
        raise ValueError(f"Expected one baseline chair_metrics.json, found {len(base_metrics)} under: {base_dir}")
    candidates = [("InternVL3-2B-hf", None, "original", path.parent, True) for path in base_metrics]
    candidates.extend((name, step, dataset, output, False) for name, step, dataset, output in discover(root))
    for name, step, dataset, output, is_base in candidates:
        chair = load_json(output / "chair_metrics.json", issues)
        if is_base and chair is None:
            raise ValueError(f"Cannot read baseline CHAIR metrics: {output / 'chair_metrics.json'}")
        pope = load_json(output / "pope" / "pope_summary.json", issues)
        amber = load_json(output / "amber" / "amber_metrics.json", issues)
        vision = {}
        for benchmark in ("mmstar", "cv-bench"):
            candidates = sorted(output.glob(f"{benchmark}/judge/*_answer.jsonl"))
            candidates += sorted(output.glob(f"vision_opd/judge/{benchmark}/*_answer.jsonl"))
            if len(candidates) > 1:
                issues.append(f"{output}: multiple {benchmark} judge files; metric omitted")
            elif candidates:
                data = load_judge(candidates[0], issues)
                if data is not None:
                    if benchmark == "cv-bench":
                        breakdown = load_json(candidates[0].parent.parent / "cvbench_breakdown.json", issues)
                        if breakdown is not None and data["judged"] == data["rows"]:
                            data["accuracy"] = breakdown.get("accuracy")
                            data["accuracy_2d"] = breakdown.get("accuracy_2d")
                            data["accuracy_3d"] = breakdown.get("accuracy_3d")
                    vision[benchmark] = data
        rows.append({"name": name, "method": "Base" if is_base else method_name(name), "step": step,
                     "dataset": dataset, "suffix": output.name, "output": output,
                     "chair": chair, "pope": pope, "amber": amber, "vision": vision,
                     "is_base": is_base})
    rows.sort(key=lambda row: (METHOD_ORDER.index(row["method"]), row["step"] if row["step"] is not None else -1,
                               row["dataset"], row["suffix"], str(row["output"])))
    return rows, issues


def cell(value, digits=4):
    if value is None or isinstance(value, bool):
        return "--"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|")


def value(row, *keys):
    current = row
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def markdown(rows, issues, root):
    lines = ["# InternVL3-2B 全 checkpoint 评估汇总", "",
             f"训练后结果目录：`{root}`。Base 单列；训练方法按方法和数值 step 排序，不选择最优 checkpoint。",
             "每个 ID 对应唯一的结果目录。数值保留原始单位：CHAIR/POPE/MMStar/CV-Bench 为 0-1，AMBER 为官方输出的点数；`--` 表示该目录无对应指标。", ""]
    if not rows:
        lines.append("No checkpoint evaluation directories found.")
        return "\n".join(lines) + "\n"
    lines += ["## CHAIR", "",
              "| ID | Method | Step | CHAIRi↓ | CHAIRs↓ | ObjPrec↑ | ObjRecall↑ | ObjF1↑ | Captions |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for index, row in enumerate(rows, 1):
        chair = row["chair"] or {}
        captions = chair.get("total_captions", chair.get("num_samples"))
        vals = [cell(chair.get(key)) for key in ("CHAIRi", "CHAIRs", "ObjPrec", "ObjRecall", "ObjF1")]
        lines.append(f"| R{index:03d} | {row['method']} | {cell(row['step'], 0)} | {' | '.join(vals)} | {cell(captions, 0)} |")
    lines += ["", "## CHAIR 对象计数", "",
              "Correct/Hall mentions 是包含重复提及的对象 mention 绝对数；RepRate 来自同一 `chair_metrics.json`。", "",
              "| ID | Method | Step | Correct mentions | Hall mentions | RepRate↓ | Schema | Prompt |",
              "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for index, row in enumerate(rows, 1):
        chair = row["chair"] or {}
        captions = chair.get("total_captions", chair.get("num_samples"))
        schema = "official" if chair.get("metric_schema") == OFFICIAL_CHAIR_SCHEMA else "unverified"
        if not chair:
            schema = "missing"
        if isinstance(captions, int) and captions != 1000:
            schema += "/partial"
        lines.append(f"| R{index:03d} | {row['method']} | {cell(row['step'], 0)} | "
                     f"{cell(chair.get('total_correct_mentions'), 0)} | {cell(chair.get('total_hallucinated_mentions'), 0)} | "
                     f"{cell(chair.get('RepRate'))} | {schema} | {cell(chair.get('eval_prompt'))} |")
    lines += ["", "## POPE / AMBER", "",
              "POPE Overall 是三个 split 的 macro F1；AMBER CHAIR/HAL 为 generative-task 指标。Base 目录只有 CHAIR 时，这里显示 `--`。", "",
              "| ID | Method | Step | POPE F1↑ | POPE Acc↑ | Adv F1↑ | Pop F1↑ | Random F1↑ | AMBER CHAIR↓ | AMBER HAL↓ | AMBER Disc F1↑ |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for index, row in enumerate(rows, 1):
        pope, amber = row["pope"] or {}, row["amber"] or {}
        amber_metrics = amber.get("metrics") if amber.get("official_returncode") == 0 else None
        splits = pope.get("benchmarks") or {}
        vals = [value(pope, "macro_f1"), value(pope, "macro_accuracy")]
        vals += [value(splits, split, "f1") for split in POPE_SPLITS]
        vals += [value(amber_metrics, key) for key in ("generative_task_chair", "generative_task_hal", "discriminative_task_f1")]
        lines.append(f"| R{index:03d} | {row['method']} | {cell(row['step'], 0)} | {' | '.join(cell(item) for item in vals)} |")
    lines += ["", "## MMStar / CV-Bench", "",
              "CV-Bench Overall/2D/3D 优先使用 `cvbench_breakdown.json`，没有 breakdown 时 Overall 为 judge 样本准确率。", "",
              "| ID | Method | Step | MMStar↑ | CV Overall↑ | CV 2D↑ | CV 3D↑ |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for index, row in enumerate(rows, 1):
        vision = row["vision"]
        vals = [value(vision, "mmstar", "accuracy"), value(vision, "cv-bench", "accuracy"),
                value(vision, "cv-bench", "accuracy_2d"), value(vision, "cv-bench", "accuracy_3d")]
        lines.append(f"| R{index:03d} | {row['method']} | {cell(row['step'], 0)} | {' | '.join(cell(item) for item in vals)} |")
    lines += ["", "## Source Index", "",
              "| ID | Experiment | Step | Dataset | Output suffix | Result directory |",
              "| --- | --- | ---: | --- | --- | --- |"]
    for index, row in enumerate(rows, 1):
        lines.append(f"| R{index:03d} | `{row['name']}` | {cell(row['step'], 0)} | `{row['dataset']}` | `{row['suffix']}` | `{row['output']}` |")
    if issues:
        lines += ["", "## Read Warnings", ""]
        lines += [f"- {issue}" for issue in issues]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("res-opd/eval_results/internvl3/full"))
    parser.add_argument("--base-result-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-md", type=Path, default=Path("res-opd/plan/internvl3_full_checkpoint_eval_summary.md"))
    args = parser.parse_args()
    rows, issues = collect(args.eval_root.resolve(), args.base_result_dir.resolve())
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown(rows, issues, args.eval_root.resolve()), encoding="utf-8")
    print(f"INTERNVL3_SUMMARY=PASS rows={len(rows)} warnings={len(issues)} output={args.output_md}")


if __name__ == "__main__":
    main()
