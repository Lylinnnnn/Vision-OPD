#!/usr/bin/env python3
"""Summarize every evaluated InternVL3 checkpoint without selecting a step."""

import argparse
import json
import re
from pathlib import Path


STEP_RE = re.compile(r"^(.+)_global_step_(\d+)$")
POPE_SPLITS = ("pope_adv", "pope_pop", "pope_random")
OFFICIAL_CHAIR_SCHEMA = "official_chair_caption_and_mention_v1"


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


def collect(root):
    rows = []
    issues = []
    for name, step, dataset, output in discover(root):
        chair = load_json(output / "chair_metrics.json", issues)
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
        rows.append({"name": name, "method": method_name(name), "step": step,
                     "dataset": dataset, "suffix": output.name, "output": output,
                     "chair": chair, "pope": pope, "amber": amber, "vision": vision})
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
    lines = ["# InternVL3 Checkpoint Evaluation Summary", "",
             f"Source root: `{root}`. One row per checkpoint, dataset, and evaluation output directory; no step selection.",
             "Scores are reported in their source JSON units (CHAIR, POPE, MMStar and CV-Bench as fractions; AMBER as official points). `--` means missing.",
             "CHAIR schema `official` is shown only when `metric_schema` explicitly confirms official caption/mention counting. Incomplete outputs are retained and flagged.", ""]
    if not rows:
        lines.append("No checkpoint evaluation directories found.")
        return "\n".join(lines) + "\n"
    lines += ["## CHAIR", "",
              "| ID | Method | Step | CHAIRi↓ | CHAIRs↓ | ObjPrec↑ | ObjRecall↑ | ObjF1↑ | RepRate↓ | Correct mentions | Hall mentions | Captions | Schema | Prompt |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for index, row in enumerate(rows, 1):
        chair = row["chair"] or {}
        captions = chair.get("total_captions", chair.get("num_samples"))
        schema = "official" if chair.get("metric_schema") == OFFICIAL_CHAIR_SCHEMA else "unverified"
        if not chair:
            schema = "missing"
        if isinstance(captions, int) and captions != 1000:
            schema += "/partial"
        vals = [cell(chair.get(key)) for key in ("CHAIRi", "CHAIRs", "ObjPrec", "ObjRecall", "ObjF1", "RepRate")]
        vals += [cell(chair.get(key), 0) for key in ("total_correct_mentions", "total_hallucinated_mentions")]
        lines.append(f"| R{index:03d} | {row['method']} | {row['step']} | {' | '.join(vals)} | {cell(captions, 0)} | {schema} | {cell(chair.get('eval_prompt'))} |")
    lines += ["", "## Other Benchmarks", "",
              "POPE is the macro F1 across available splits; individual split F1 scores are shown. AMBER CHAIR/HAL are generative-task metrics. CV-Bench overall follows `cvbench_breakdown.json` when present.", "",
              "| ID | POPE F1↑ | POPE Acc↑ | Adv F1↑ | Pop F1↑ | Random F1↑ | AMBER CHAIR↓ | AMBER HAL↓ | AMBER Disc F1↑ | MMStar↑ | CV-Bench↑ | CV 2D↑ | CV 3D↑ | Flags |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for index, row in enumerate(rows, 1):
        pope, amber, vision = row["pope"] or {}, row["amber"] or {}, row["vision"]
        amber_metrics = amber.get("metrics") if amber.get("official_returncode") == 0 else None
        flags = []
        if row["chair"] is None:
            flags.append("CHAIR missing")
        elif row["chair"].get("num_samples") not in (None, 1000):
            flags.append("CHAIR partial")
        splits = pope.get("benchmarks") or {}
        if any(split not in splits for split in POPE_SPLITS):
            flags.append("POPE incomplete")
        if amber.get("official_returncode") != 0:
            flags.append("AMBER missing/failed")
        for benchmark in ("mmstar", "cv-bench"):
            result = vision.get(benchmark)
            if result is None:
                flags.append(f"{benchmark} missing")
            elif result["judged"] != result["rows"]:
                flags.append(f"{benchmark} unjudged")
        vals = [value(pope, "macro_f1"), value(pope, "macro_accuracy")]
        vals += [value(splits, split, "f1") for split in POPE_SPLITS]
        vals += [value(amber_metrics, key) for key in ("generative_task_chair", "generative_task_hal", "discriminative_task_f1")]
        vals += [value(vision, "mmstar", "accuracy"), value(vision, "cv-bench", "accuracy"),
                 value(vision, "cv-bench", "accuracy_2d"), value(vision, "cv-bench", "accuracy_3d")]
        lines.append(f"| R{index:03d} | {' | '.join(cell(item) for item in vals)} | {', '.join(flags) or 'complete'} |")
    lines += ["", "## Source Index", "",
              "| ID | Experiment | Dataset | Output suffix | Result directory |",
              "| --- | --- | --- | --- | --- |"]
    for index, row in enumerate(rows, 1):
        lines.append(f"| R{index:03d} | `{row['name']}` | `{row['dataset']}` | `{row['suffix']}` | `{row['output']}` |")
    if issues:
        lines += ["", "## Read Warnings", ""]
        lines += [f"- {issue}" for issue in issues]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=Path("res-opd/eval_results/internvl3/full"))
    parser.add_argument("--output-md", type=Path, default=Path("res-opd/plan/internvl3_full_checkpoint_eval_summary.md"))
    args = parser.parse_args()
    rows, issues = collect(args.eval_root.resolve())
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown(rows, issues, args.eval_root.resolve()), encoding="utf-8")
    print(f"INTERNVL3_SUMMARY=PASS rows={len(rows)} warnings={len(issues)} output={args.output_md}")


if __name__ == "__main__":
    main()
