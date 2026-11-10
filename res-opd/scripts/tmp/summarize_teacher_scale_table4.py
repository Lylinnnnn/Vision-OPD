#!/usr/bin/env python3
"""Build the non-Base teacher-scale table from exact checkpoints and evaluated rows."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import audit_teacher_scale_tr05_chair as chair_audit
import check_main_tr05_pope_amber as aux_audit
import summarize_main_tr05_inventory as inventory
import summarize_teacher_scale_tr05 as teacher_scale


DEFAULT_OSS_ROOT = Path("/data/oss_bucket_0/yanlin/ckpt/OPD/v4")
METRICS = ("CHAIR", "POPE", "AMBER", "MMStar", "CV-Bench")


def controls():
    return [(student, teacher, view, spec)
            for student, teacher, view, spec in teacher_scale.table_specs()
            if spec.method != "Base"]


def final_step(student, teacher):
    return 312 if student == "4B" and teacher == "8B fixed" else 156


def observed_steps(spec, oss_root, eval_root):
    eval_parent = eval_root / "instruct/full"
    prefix = f"{spec.experiment}_global_step_"
    steps = {int(path.name.removeprefix(prefix))
             for path in eval_parent.glob(f"{prefix}*")
             if path.is_dir() and path.name.removeprefix(prefix).isdigit()}
    if oss_root.is_dir():
        steps.update(int(path.name.removeprefix("global_step_"))
                     for path in teacher_scale.checkpoint_dirs(spec, oss_root))
    return sorted(steps)


def checkpoint_state(oss_root, spec, step, aliases):
    checkpoint = oss_root / spec.oss_name / f"global_step_{step}"
    if not oss_root.is_dir():
        return "OSS 未挂载，无法核验", checkpoint
    if checkpoint.is_dir():
        if all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0
               for name in ("model.safetensors", "config.json")):
            return "完整", checkpoint
        return "checkpoint 不完整", checkpoint
    if aliases:
        return "疑似其他命名，待核验", checkpoint
    return "缺少 checkpoint", checkpoint


def inspect_one(student, teacher, view, spec, step, oss_root, eval_root, issues, aliases=()):
    ckpt_state, checkpoint = checkpoint_state(oss_root, spec, step, aliases)
    dataset = (eval_root / "instruct/full" / f"{spec.experiment}_global_step_{step}"
               / inventory.DATASET)
    directories = aux_audit.output_dirs(dataset)
    chair_state, chair_values, chair_source = chair_audit.inspect_chair(
        directories, dataset, issues, f"{student}/{teacher}/{view}/s{step}")
    pope = aux_audit.inspect_all(directories, "pope", aux_audit.EXPECTED_POPE)
    amber = aux_audit.inspect_all(directories, "amber", aux_audit.EXPECTED_AMBER)
    mmstar = aux_audit.inspect_vision_all(directories, "mmstar", 1500)
    cvbench = aux_audit.inspect_vision_all(directories, "cv-bench", 2638)
    results = {"POPE": pope, "AMBER": amber, "MMStar": mmstar, "CV-Bench": cvbench}
    states = {"CHAIR": chair_state, **{name: result.state for name, result in results.items()}}
    return {
        "student": student, "teacher": teacher, "view": view, "step": step,
        "checkpoint": checkpoint, "ckpt_state": ckpt_state,
        "chair_values": chair_values, "chair_source": chair_source,
        "results": results, "states": states, "aliases": aliases,
    }


def collect(oss_root, eval_root):
    issues = []
    final_rows = []
    step_rows = []
    for student, teacher, view, spec in controls():
        ratio = view.split("=")[-1]
        aliases = teacher_scale.possible_aliases(
            spec, student, teacher.split()[0], ratio, oss_root
        ) if oss_root.is_dir() else []
        expected = final_step(student, teacher)
        steps = observed_steps(spec, oss_root, eval_root)
        final = inspect_one(student, teacher, view, spec, expected, oss_root,
                            eval_root, issues, aliases)
        final["observed_steps"] = steps
        final_rows.append(final)
        for step in steps:
            step_rows.append(final if step == expected else inspect_one(
                student, teacher, view, spec, step, oss_root, eval_root, issues, aliases))
    return final_rows, step_rows, issues


def format_ratio(value):
    return "--" if value is None else f"{value * 100:.2f}"


def table_cells(row):
    chair = row["chair_values"]
    results = row["results"]
    cells = [row["student"], row["teacher"], row["view"], str(row["step"])]
    cells.extend(format_ratio(chair.get(key)) for key in ("CHAIRi", "CHAIRs", "ObjF1"))
    length = chair.get("chars_per_image")
    cells.append("--" if length is None else f"{length:.2f}")
    cells.extend(str(chair.get(key, "--")) for key in ("correct_mentions", "hall_mentions"))
    cells.extend((results["POPE"].values[0], results["AMBER"].values[1],
                  results["MMStar"].values[0], results["CV-Bench"].values[0]))
    return cells


HEADER = (
    "| Student | Teacher | View | Step | CHAIRi↓ | CHAIRs↓ | ObjF1↑ | "
    "Resp. Len. | Correct↑ | Hall.↓ | POPE F1↑ | A-HAL↓ | MMStar Acc.↑ | "
    "CV-Bench Acc.↑ | Checkpoint | 缺口 |"
)
RULE = "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"


def gaps(row):
    missing = []
    if row["ckpt_state"] != "完整":
        missing.append(row["ckpt_state"])
    for metric in METRICS:
        state = row["states"][metric]
        if state in ("新版指标齐全", "已推理有结果"):
            continue
        missing.append(f"{metric}: {state}")
    return "; ".join(missing) if missing else "齐全"


def row_line(row):
    return "| " + " | ".join(table_cells(row) + [row["ckpt_state"], gaps(row)]) + " |"


def render(final_rows, step_rows, issues, oss_root):
    lines = [
        "# Table 4: teacher-scale uniform Frozen RKL (tr=0.50)", "",
        "仅统计原表的 7 组非 Base 控制：四组压缩教师视图改为 `tr=0.50`，三组原图教师视图仍为 `tr=1.0`。",
        "同尺寸为 frozen 原模型教师，跨尺寸为 fixed 大模型教师；学生始终用原图，且不使用 RiskMask。",
        "主表固定按一轮训练的最终 step 核查：batch 32 为 156，batch 16 为 312；不以较早 step 替代最终 step。",
        "CHAIR 必须是 1000 张、`Describe the image.`、original student、official caption/mention 口径。",
        "Resp. Len. 为最终回答平均字符数/图，不是 token；Correct/Hall. 为含重复的对象 mention 总数。",
        "POPE F1 为三个 5223 条 split 的 macro F1；AMBER A-HAL 为官方点数。百分数列均以百分数显示。",
        "有 `[ERROR]` 回答、缺少原始推理或只有孤立指标的目录不视为有效结果；`--` 不代表零。",
        f"OSS checkpoint 根目录：`{oss_root}`（{'已挂载' if oss_root.is_dir() else '未挂载，不能断言未训练'}）。",
        "", "## 论文对应最终 step", "", HEADER, RULE,
    ]
    lines.extend(row_line(row) for row in final_rows)
    lines.extend(["", "## 全部已发现 step", "", HEADER, RULE])
    if step_rows:
        lines.extend(row_line(row) for row in step_rows)
    else:
        lines.append("无已发现的 checkpoint 或评测 step。")
    lines.extend(["", "## 待补与异常", ""])
    pending = [row for row in final_rows if gaps(row) != "齐全"]
    if not pending:
        lines.append("无。")
    for row in pending:
        earlier = row["observed_steps"]
        available = f"; 发现 steps={earlier}" if earlier else ""
        lines.append(f"- {row['student']}/{row['teacher']}/{row['view']} "
                     f"最终 step={row['step']}: {gaps(row)}{available}; "
                     f"ckpt=`{row['checkpoint']}`")
        for metric in METRICS[1:]:
            result = row["results"][metric]
            if result.state not in ("已推理有结果", "未推理"):
                lines.append(f"  - {metric}: {result.count}; {result.detail or result.state}; "
                             f"evidence=`{result.source or '--'}`")
        for alias in row["aliases"]:
            lines.append(f"  - 疑似别名：`{alias}`")
    lines.extend(["", "## 指标与 checkpoint 来源", "",
                  "| Student | Teacher | View | Step | Checkpoint | CHAIR | POPE | AMBER | MMStar | CV-Bench |",
                  "| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |"])
    for row in final_rows + [item for item in step_rows if item not in final_rows]:
        sources = [row["checkpoint"], row["chair_source"]]
        sources.extend(row["results"][key].source or "--" for key in METRICS[1:])
        lines.append("| " + " | ".join(
            [row["student"], row["teacher"], row["view"], str(row["step"])]
            + [f"`{source}`" if source != "--" else "--" for source in sources]) + " |")
    if issues:
        lines.extend(["", "## 读取警告", ""])
        lines.extend(f"- {issue}" for issue in sorted(set(issues)))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oss-root", type=Path, default=DEFAULT_OSS_ROOT)
    parser.add_argument("--eval-root", type=Path,
                        default=inventory.ROOT / "res-opd/eval_results")
    parser.add_argument("--output-md", type=Path,
                        default=Path("res-opd/plan/teacher_scale_table4_tr05_summary.md"))
    args = parser.parse_args()
    final_rows, step_rows, issues = collect(args.oss_root, args.eval_root)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render(final_rows, step_rows, issues, args.oss_root), encoding="utf-8")
    counts = Counter(row["ckpt_state"] for row in final_rows)
    print(f"TABLE4_TR05=WRITTEN final={len(final_rows)} observed_steps={len(step_rows)} "
          f"checkpoint={dict(counts)} incomplete={sum(gaps(row) != '齐全' for row in final_rows)} "
          f"warnings={len(issues)} md={args.output_md}")


if __name__ == "__main__":
    main()
