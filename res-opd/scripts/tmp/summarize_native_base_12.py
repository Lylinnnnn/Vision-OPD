#!/usr/bin/env python3
"""Summarize the twelve official native compressed Base evaluations."""

import argparse
import json
from pathlib import Path

import eval_native_base_compressed_12 as native


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def collect(repo, suffix):
    rows = []
    incomplete = []
    for model_key, ratio in native.jobs("all", native.RATIOS):
        output = native.result_dir(repo, model_key, ratio, suffix)
        if not native.complete(output, ratio):
            incomplete.append(f"{model_key} sr={ratio}: {output}")
            continue
        chair_path = output / "chair_metrics.json"
        pope_path = output / "pope/pope_summary.json"
        amber_path = output / "amber/amber_metrics.json"
        chair = read_json(chair_path)
        pope = read_json(pope_path)
        amber = read_json(amber_path)
        rows.append({
            "model": native.MODELS[model_key][0],
            "student_ratio": ratio,
            "chair": {key: chair.get(key) for key in (
                "CHAIRi", "CHAIRs", "ObjPrec", "ObjRecall", "ObjF1",
                "total_correct_mentions", "total_hallucinated_mentions",
                "total_captions", "metric_schema", "eval_prompt")},
            "pope": {
                "macro_f1": pope["macro_f1"],
                "macro_accuracy": pope.get("macro_accuracy"),
                "split_f1": {split: pope["benchmarks"][split]["f1"]
                             for split in native.POPE_SPLITS},
            },
            "amber": {key: amber["metrics"].get(key) for key in (
                "generative_task_chair", "generative_task_hal",
                "discriminative_task_f1")},
            "sources": {
                "chair": str(chair_path),
                "pope": str(pope_path),
                "amber": str(amber_path),
            },
        })
    if incomplete:
        raise RuntimeError("Incomplete or invalid evaluations:\n" + "\n".join(incomplete))
    return rows


def percentage(value):
    return "--" if value is None else f"{100 * value:.2f}"


def points(value):
    return "--" if value is None else f"{value:.2f}"


def markdown(rows, suffix):
    lines = [
        "# Native Compressed Base: 12 Official Evaluations", "",
        f"Result suffix: `{suffix}`. All 12 rows passed the evaluation script's completeness check: "
        "official mention/caption CHAIR, `Describe the image.`, 1,000 captions, "
        "POPE adversarial/popular/random splits, and successful official AMBER evaluation.",
        "The model checkpoint is unchanged within each family; only its inference image ratio varies. "
        "Scores below are percentages (AMBER retains its reported points).", "",
        "| ID | Model | Image ratio | CHAIRi↓ | CHAIRs↓ | ObjF1↑ | POPE F1↑ | A-CHAIR↓ | A-HAL↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(rows, 1):
        chair, pope, amber = row["chair"], row["pope"], row["amber"]
        lines.append(
            f"| N{index:02d} | {row['model']} | {row['student_ratio']} | "
            f"{percentage(chair['CHAIRi'])} | {percentage(chair['CHAIRs'])} | "
            f"{percentage(chair['ObjF1'])} | {percentage(pope['macro_f1'])} | "
            f"{points(amber['generative_task_chair'])} | "
            f"{points(amber['generative_task_hal'])} |"
        )
    lines.extend([
        "", "## Object Accounting", "",
        "Official CHAIR mention counts include repeated mentions; ObjF1 uses unique object categories.", "",
        "| ID | ObjPrec↑ | ObjRecall↑ | Correct mentions | Hallucinated mentions | POPE Adv F1↑ | POPE Pop F1↑ | POPE Random F1↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for index, row in enumerate(rows, 1):
        chair, pope = row["chair"], row["pope"]
        splits = pope["split_f1"]
        lines.append(
            f"| N{index:02d} | {percentage(chair['ObjPrec'])} | "
            f"{percentage(chair['ObjRecall'])} | "
            f"{chair['total_correct_mentions']} | {chair['total_hallucinated_mentions']} | "
            f"{percentage(splits['pope_adv'])} | {percentage(splits['pope_pop'])} | "
            f"{percentage(splits['pope_random'])} |"
        )
    lines.extend(["", "## Sources", ""])
    for index, row in enumerate(rows, 1):
        lines.append(f"- N{index:02d}: `{Path(row['sources']['chair']).parent}`")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-suffix", default="native_base_official_v2")
    parser.add_argument("--output-md", type=Path,
                        default=Path("res-opd/paper/tmp/native_base_official_v2_12_summary.md"))
    parser.add_argument("--output-json", type=Path,
                        default=Path("res-opd/paper/tmp/native_base_official_v2_12_summary.json"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    rows = collect(repo, args.output_suffix)
    for path in (args.output_md, args.output_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown(rows, args.output_suffix), encoding="utf-8")
    args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    print(f"NATIVE_BASE_SUMMARY=PASS rows={len(rows)} md={args.output_md} json={args.output_json}")


if __name__ == "__main__":
    main()
