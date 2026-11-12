#!/usr/bin/env python3
"""Summarize latest matched teacher-scale CHAIR/POPE/AMBER in one Markdown file."""

import argparse
from pathlib import Path

import audit_teacher_scale_tr05_chair as chair_audit
import check_main_tr05_pope_amber as aux_audit
import summarize_main_tr05_inventory as inventory
import summarize_teacher_scale_table4 as table4


def value(x, percent=False):
    if x is None or x == "--":
        return "--"
    return f"{float(x) * 100:.2f}" if percent else str(x)


def base_row(size, eval_root, issues):
    dataset = eval_root / "instruct/full" / f"Qwen3VL-{size}-Instruct" / inventory.DATASET
    directories = aux_audit.output_dirs(dataset)
    chair_state, chair, chair_source = chair_audit.inspect_chair(
        directories, dataset, issues, f"{size}/Base", 1.0)
    pope = aux_audit.inspect_all(directories, "pope", aux_audit.EXPECTED_POPE)
    amber = aux_audit.inspect_all(directories, "amber", aux_audit.EXPECTED_AMBER)
    return {"student": size, "teacher": "--", "view": "Base", "step": "--",
            "chair_values": chair, "chair_source": chair_source,
            "results": {"POPE": pope, "AMBER": amber},
            "states": {"CHAIR": chair_state, "POPE": pope.state, "AMBER": amber.state}}


def line(row):
    chair = row["chair_values"]
    results = row["results"]
    cells = [row["student"], row["teacher"], row["view"], str(row["step"])]
    cells += [value(chair.get(key), True) for key in ("CHAIRi", "CHAIRs", "ObjF1")]
    cells += [f"{chair['chars_per_image']:.2f}" if "chars_per_image" in chair else "--",
              str(chair.get("correct_mentions", "--")), str(chair.get("hall_mentions", "--"))]
    cells += [results["POPE"].values[0], results["AMBER"].values[1]]
    cells += ["; ".join(f"{key}:{state}" for key, state in row["states"].items()
                            if state not in ("新版指标齐全", "已推理有结果")) or "齐全"]
    return "| " + " | ".join(map(str, cells)) + " |"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=inventory.ROOT / "res-opd/eval_results")
    parser.add_argument("--oss-root", type=Path, default=table4.DEFAULT_OSS_ROOT)
    parser.add_argument("--matched-aux-suffix", default="teacher_scale_matched_official_v2")
    parser.add_argument("--output-md", type=Path,
                        default=Path("res-opd/plan/fixed_teacher_latest_official_v2.md"))
    args = parser.parse_args()
    groups, step_rows, issues = table4.collect(args.oss_root, args.eval_root,
                                               args.matched_aux_suffix)
    chosen = []
    for group in groups:
        if (group["student"], group["teacher"], group["view"]) == (
                "4B", "4B frozen", "sr=0.5,tr=1.0"):
            continue
        rows = [row for row in step_rows if (row["student"], row["teacher"], row["view"])
                == (group["student"], group["teacher"], group["view"])]
        if rows:
            matched = [row for row in rows if row["states"]["CHAIR"] == "新版指标齐全"]
            chosen.append(max(matched or rows, key=lambda row: row["step"]))
    ordered = []
    for size in ("2B", "4B"):
        ordered.append(base_row(size, args.eval_root, issues))
        ordered.extend(row for row in chosen if row["student"] == size)
    lines = ["# Fixed-teacher controls: latest matched evaluation", "",
             "`tr=0.50`: original student view (`sr=1.0`); `sr=0.5`: compressed student view (`tr=1.0`).",
             "Same-size teacher is frozen; larger teacher is fixed. All use uniform RKL, no RiskMask.",
             "Latest means highest step with valid official CHAIR, or highest discovered step if none is valid; never best score.",
             "Resp. Len. is mean final-answer characters per image. POPE F1 is three-split macro F1; A-HAL is official AMBER percentage.",
             "`--` means no protocol-verified result; it is not zero. CHAIR/ObjF1/POPE are displayed as percentages.",
             "", "## Main Table", "",
             "| Student | Teacher | View | Step | CHAIRi↓ | CHAIRs↓ | ObjF1↑ | Resp. Len. | Correct↑ | Hall.↓ | POPE F1↑ | A-HAL↓ | Status |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    lines += [line(row) for row in ordered]
    lines += ["", "## All Steps And Sources", "",
              table4.render(groups, step_rows, issues, args.oss_root,
                            args.matched_aux_suffix)]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"FIXED_TEACHER_LATEST=WRITTEN rows={len(ordered)} all_steps={len(step_rows)} md={args.output_md}")


if __name__ == "__main__":
    main()
