#!/usr/bin/env python3
"""Build the two-direction non-Base teacher-scale table from evaluated rows."""

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
    configs = (
        ("2B", "2B", "1.0", "0.5"),
        ("2B", "4B", "1.0", "0.5"),
        ("2B", "2B", "0.5", "1.0"),
        ("2B", "4B", "0.5", "1.0"),
        ("4B", "4B", "1.0", "0.5"),
        ("4B", "8B", "1.0", "0.5"),
        ("4B", "8B", "0.5", "1.0"),
    )
    return [
        (student, f"{teacher} {'frozen' if student == teacher else 'fixed'}",
         f"sr={student_ratio},tr={teacher_ratio}",
         teacher_scale.experiment_spec(student, teacher, teacher_ratio, student_ratio))
        for student, teacher, student_ratio, teacher_ratio in configs
    ]


def student_ratio(view):
    return float(view.split(",", 1)[0].split("=", 1)[1])


def dataset_tag(view):
    return f"train5000_test1000_original_sr{str(student_ratio(view)).replace('.', 'p')}"


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


def matched_protocol_error(directory, benchmark, expected_student_ratio=1.0):
    if benchmark == "pope":
        summary = inventory.read_json(directory / "pope/pope_summary.json")
        splits = ("pope_adv", "pope_pop", "pope_random")
        for split in splits:
            metrics = summary["benchmarks"][split]
            if metrics.get("pope_source") != "res-opd-test":
                return f"{split} source is not res-opd-test"
            path = directory / "pope" / split / "eval_results.jsonl"
            if any(row.get("degradation_mode") != "original" or
                   row.get("student_ratio") != expected_student_ratio for row in inventory.jsonl(path)):
                return f"{split} student view is not verified as sr={expected_student_ratio}"
    else:
        path = directory / "amber/raw_results.jsonl"
        if any(row.get("degradation_mode") != "original" or
               row.get("student_ratio") != expected_student_ratio for row in inventory.jsonl(path)):
            return f"AMBER student view is not verified as sr={expected_student_ratio}"
    return None


def matched_aux(dataset, directories, benchmark, matched_suffix, expected_student_ratio):
    expected = aux_audit.EXPECTED_POPE if benchmark == "pope" else aux_audit.EXPECTED_AMBER
    matched_dir = dataset / matched_suffix
    matched = aux_audit.inspect_all([matched_dir], benchmark, expected)
    if matched.state == "已推理有结果":
        try:
            error = matched_protocol_error(matched_dir, benchmark, expected_student_ratio)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            error = str(exc)
        if error:
            empty = ("--",) if benchmark == "pope" else ("--", "--")
            return aux_audit.Inspection("协议待核实", matched.count, empty,
                                        matched.source, error)
    if matched.state != "未推理":
        return matched
    old = aux_audit.inspect_all([directory for directory in directories
                                 if directory.name != matched_suffix], benchmark, expected)
    if old.state != "未推理":
        empty = ("--",) if benchmark == "pope" else ("--", "--")
        return aux_audit.Inspection("旧结果待匹配复评", old.count, empty, old.source,
                                    f"旧结果状态={old.state}; 未在 {matched_suffix} 找到匹配复评")
    empty = ("--",) if benchmark == "pope" else ("--", "--")
    return aux_audit.Inspection("未匹配推理", old.count, empty, dataset / matched_suffix)


def inspect_one(student, teacher, view, spec, step, oss_root, eval_root, issues,
                matched_suffix, aliases=()):
    ckpt_state, checkpoint = checkpoint_state(oss_root, spec, step, aliases)
    expected_student_ratio = student_ratio(view)
    dataset = (eval_root / "instruct/full" / f"{spec.experiment}_global_step_{step}"
               / dataset_tag(view))
    directories = aux_audit.output_dirs(dataset)
    chair_state, chair_values, chair_source = chair_audit.inspect_chair(
        directories, dataset, issues, f"{student}/{teacher}/{view}/s{step}",
        expected_student_ratio)
    pope = matched_aux(dataset, directories, "pope", matched_suffix,
                       expected_student_ratio)
    amber = matched_aux(dataset, directories, "amber", matched_suffix,
                        expected_student_ratio)
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


def collect(oss_root, eval_root, matched_suffix="teacher_scale_matched_official_v2"):
    issues = []
    groups = []
    step_rows = []
    for student, teacher, view, spec in controls():
        ratio = view.split("=")[-1]
        aliases = teacher_scale.possible_aliases(
            spec, student, teacher.split()[0], ratio, oss_root,
            str(student_ratio(view))
        ) if oss_root.is_dir() else []
        steps = observed_steps(spec, oss_root, eval_root)
        complete = [step for step in steps
                    if checkpoint_state(oss_root, spec, step, aliases)[0] == "完整"]
        groups.append({"student": student, "teacher": teacher, "view": view,
                       "spec": spec, "steps": steps, "complete": complete,
                       "aliases": aliases})
        for step in steps:
            step_rows.append(inspect_one(student, teacher, view, spec, step,
                                         oss_root, eval_root, issues, matched_suffix, aliases))
    return groups, step_rows, issues


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


def group_status(group, oss_root):
    if not oss_root.is_dir():
        return "OSS 未挂载，无法核验"
    if group["complete"]:
        return "有完整 checkpoint"
    if group["aliases"]:
        return "疑似其他命名，待核验"
    if group["steps"]:
        return "仅不完整 checkpoint/旧评测，待核验"
    return "缺少 checkpoint（规范名称下未找到）"


def render(groups, step_rows, issues, oss_root, matched_suffix):
    lines = [
        "# Table 4: two-direction teacher-scale uniform Frozen RKL", "",
        "仅统计 7 组非 Base 控制：四组 `sr=1.0,tr=0.5` 为压缩教师→原图学生，"
        "三组 `sr=0.5,tr=1.0` 为原图教师→压缩学生。",
        "同尺寸为 frozen 原模型教师，跨尺寸为 fixed 大模型教师；两组都不使用 RiskMask。",
        "评测输入只按学生视角：前组原图 `sr=1.0`，后组降采样再放大的模糊图 `sr=0.5`；推理不加载教师。",
        "原表并未统一选训练最终 step；本清单逐一展示实际发现的 step，不自动挑选论文代表 checkpoint。",
        "CHAIR 必须是 1000 张、`Describe the image.`、与该行 `sr` 一致、official caption/mention 口径。",
        "Resp. Len. 为最终回答平均字符数/图，不是 token；Correct/Hall. 为含重复的对象 mention 总数。",
        "POPE F1 为三个 5223 条 split 的 macro F1；AMBER A-HAL 为官方点数。百分数列均以百分数显示。",
        "有 `[ERROR]` 回答、缺少原始推理或只有孤立指标的目录不视为有效结果；`--` 不代表零。",
        f"两种视角的 POPE/A-HAL 均仅从 `{matched_suffix}` 读取匹配复评；其他目录仅作旧结果线索，不填表。",
        f"OSS checkpoint 根目录：`{oss_root}`（{'已挂载' if oss_root.is_dir() else '未挂载，不能断言未训练'}）。",
        "", "## 七组实验与 checkpoint", "",
        "| Student | Teacher | View | Method | 完整 checkpoint steps | 发现的评测/ckpt steps | 状态 | OSS 实验目录 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for group in groups:
        state = group_status(group, oss_root)
        lines.append("| " + " | ".join((
            group["student"], group["teacher"], group["view"], "uniform Frozen RKL",
            ", ".join(map(str, group["complete"])) or "--",
            ", ".join(map(str, group["steps"])) or "--", state,
            f"`{oss_root / group['spec'].oss_name}`")) + " |")
    lines.extend(["", "## 全部已发现 step 的指标", "", HEADER, RULE])
    if step_rows:
        lines.extend(row_line(row) for row in step_rows)
    else:
        lines.append("无已发现的 checkpoint 或评测 step。")
    lines.extend(["", "## 待补与异常", ""])
    missing_groups = [group for group in groups if not group["complete"]]
    pending = [row for row in step_rows if gaps(row) != "齐全"]
    if not missing_groups and not pending:
        lines.append("无。")
    for group in missing_groups:
        lines.append(f"- {group['student']}/{group['teacher']}/{group['view']}: "
                     f"{group_status(group, oss_root)}; 发现 steps={group['steps'] or '--'}; "
                     f"expected=`{oss_root / group['spec'].oss_name}`")
        for alias in group["aliases"]:
            lines.append(f"  - 疑似别名：`{alias}`")
    for row in pending:
        lines.append(f"- {row['student']}/{row['teacher']}/{row['view']} "
                     f"step={row['step']}: {gaps(row)}; "
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
    for row in step_rows:
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
    parser.add_argument("--matched-aux-suffix", default="teacher_scale_matched_official_v2")
    args = parser.parse_args()
    groups, step_rows, issues = collect(args.oss_root, args.eval_root, args.matched_aux_suffix)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render(groups, step_rows, issues, args.oss_root,
                                     args.matched_aux_suffix), encoding="utf-8")
    counts = Counter(row["ckpt_state"] for row in step_rows)
    print(f"TABLE4_TR05=WRITTEN groups={len(groups)} observed_steps={len(step_rows)} "
          f"groups_without_verified_ckpt={sum(not group['complete'] for group in groups)} "
          f"checkpoint={dict(counts)} pending_step_results={sum(gaps(row) != '齐全' for row in step_rows)} "
          f"warnings={len(issues)} md={args.output_md}")


if __name__ == "__main__":
    main()
