#!/usr/bin/env python3
"""Inventory Table 5 teacher-scale controls from exact checkpoints and eval files."""

import argparse
import re
from pathlib import Path

import summarize_main_tr05_inventory as inventory


TABLE_METRICS = ("CHAIRi", "CHAIRs", "ObjF1", "POPE", "A-HAL", "MMStar", "CV-Bench")


def experiment_spec(student, teacher, ratio, student_ratio="1.0"):
    model = f"Qwen3VL-{student}-Instruct"
    batch = "b16" if student == "4B" and teacher == "8B" else "b32"
    local_prefix = f"Res-OPD-{model}-orig-sr{student_ratio}-tr{ratio}-a1.0-"
    oss_prefix = "ResOPD_" + ("Qwen3VL_4B_Instruct_" if student == "4B" else "")
    oss_prefix += f"orig_sr{student_ratio}_tr{ratio}_a1.0_"
    if student == teacher:
        local_method, oss_method = "frozen-rkl", "frozen_rkl"
        method = "Frozen RKL"
    else:
        local_method = f"fixed-rkl-teacherQwen3VL-{teacher}-Instruct"
        oss_method = f"fixed_rkl_teacherQwen3VL_{teacher}_Instruct"
        method = f"Fixed {teacher} teacher"
    return inventory.Spec(
        "Qwen3VL", student, method,
        f"{local_prefix}{local_method}-{batch}-rn4-full5k-e1",
        f"{oss_prefix}{oss_method}_{batch}_rn4_full5k-e1",
    )


def table_specs():
    rows = []
    for student, controls in (
        ("2B", (("2B", "0.5"), ("4B", "0.5"), ("2B", "1.0"), ("4B", "1.0"))),
        ("4B", (("4B", "0.5"), ("8B", "0.5"), ("8B", "1.0"))),
    ):
        rows.append((student, "--", "Base", inventory.Spec("Qwen3VL", student, "Base",
                                                        f"Qwen3VL-{student}-Instruct")))
        for teacher, ratio in controls:
            label = f"{teacher} frozen" if student == teacher else f"{teacher} fixed"
            rows.append((student, label, f"tr={ratio}", experiment_spec(student, teacher, ratio)))
    return rows


def checkpoint_dirs(spec, oss_root):
    parent = oss_root / spec.oss_name
    if not parent.is_dir():
        return []
    return sorted((path for path in parent.iterdir()
                   if path.is_dir() and inventory.STEP_RE.fullmatch(path.name)),
                  key=lambda path: int(inventory.STEP_RE.fullmatch(path.name).group(1)))


def possible_aliases(spec, student, teacher, ratio, oss_root, student_ratio="1.0"):
    if student == teacher or not oss_root.is_dir():
        return []
    prefix = "resopdorig" if student == "2B" else f"resopdqwen3vl{student.lower()}instructorig"
    teacher_tag = f"teacherqwen3vl{teacher.lower()}instruct"
    aliases = []
    for path in oss_root.iterdir():
        name = re.sub(r"[^a-z0-9]", "", path.name.lower())
        if (path.is_dir() and path.name != spec.oss_name and name.startswith(prefix)
                and f"sr{student_ratio.replace('.', '')}" in name
                and f"tr{ratio.replace('.', '')}" in name
                and "fixedrkl" in name and teacher_tag in name):
            aliases.append(path)
    return sorted(aliases)


def status_for(row, mounted, incomplete=(), aliases=(), parent_exists=False):
    values = row["values"]
    spec = row["spec"]
    if spec.method == "Base":
        return "齐全" if all(metric in values for metric in TABLE_METRICS) else "Base 待补推理"
    if not mounted:
        return "OSS 未挂载，checkpoint 待核实"
    if row["step"] is None:
        if incomplete:
            return "checkpoint 上传不完整，待核实"
        if aliases:
            return "疑似其他命名，待人工核对"
        if parent_exists:
            return "训练目录存在，尚无完整 checkpoint，待核实"
        return "待训练（OSS 无匹配 checkpoint）"
    if "checkpoint待核实" in row["status"]:
        return "checkpoint 待核实"
    if all(metric in values for metric in TABLE_METRICS):
        return "齐全"
    if "CHAIRi" not in values:
        return "有 checkpoint，待新版 CHAIR 推理"
    return "有 checkpoint，待补其他 benchmark"


def collect(oss_root):
    issues = []
    rows = []
    for student, teacher, view, spec in table_specs():
        found, mounted = inventory.collect_specs([spec], oss_root, issues)
        incomplete = []
        aliases = []
        parent_exists = False
        if mounted and spec.method != "Base":
            parent_exists = (oss_root / spec.oss_name).is_dir()
            step_dirs = checkpoint_dirs(spec, oss_root)
            complete = {int(path.name.split("_")[-1]) for path in step_dirs
                        if all((path / name).is_file() and (path / name).stat().st_size > 0
                               for name in ("model.safetensors", "config.json"))}
            incomplete = [path for path in step_dirs
                          if int(path.name.split("_")[-1]) not in complete]
            if not complete:
                aliases = possible_aliases(spec, student, teacher.split()[0], view.split("=")[-1], oss_root)
        for row in found:
            if spec.method != "Base" and row["step"] is None:
                row["checkpoint"] = f"expected:{oss_root / spec.oss_name}"
            rows.append({**row, "student": student, "teacher": teacher, "view": view,
                         "incomplete": incomplete, "aliases": aliases,
                         "status": status_for(row, mounted, incomplete, aliases, parent_exists)})
    return rows, issues, oss_root.is_dir()


def render(rows, issues, mounted, oss_root):
    lines = [
        "# Table 5 teacher-scale controls: tr=0.50", "",
        "原表将 tr=0.75 四组改为 tr=0.50；tr=1.0 三组不变。Base 共两组。",
        "同尺寸使用 frozen 原模型教师；跨尺寸使用 fixed 外部大模型教师。所有行均为 uniform RKL，\n"
        "不使用 RiskMask/RandomMask/EntropyMask。学生始终看 original (sr=1.0)。",
        "逐 step 列出 OSS 上完整 checkpoint 和精确实验名的评测，不按分数挑选；旧版 CHAIR 不填新表。",
        "无 checkpoint 只表示该行配置在当前 OSS 挂载下未找到；疑似别名或不完整上传单独列出。",
        "CHAIR 仅接受 `Describe the image.`、1000 张图、official caption/mention 口径；旧 CHAIR 不填入新表。",
        "POPE 是三个 split 的 macro F1；AMBER A-HAL 保留官方点数；其他指标显示百分数。",
        "Words/img、Chars/img 来自最终回答，Correct/Hall mentions 包含重复 mention；`--` 表示未核实，不是零。",
        f"OSS 挂载：`{oss_root}` {'可用' if mounted else '不可用，不能判断是否未训练'}。", "",
        "## 表格配置矩阵", "",
        "| Student | Teacher | View | TEACHER_MODE | MODEL_PATH basename | TEACHER_MODEL_PATH basename | tr | Batch | 实验目录 (OSS) |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for size, teacher, view, spec in table_specs():
        teacher_size = teacher.split()[0] if teacher != "--" else None
        mode = "--" if teacher_size is None else "frozen" if teacher_size == size else "fixed"
        teacher_model = f"Qwen3VL-{teacher_size}-Instruct" if mode == "fixed" else "--"
        batch = "--" if mode == "--" else "16" if size == "4B" and mode == "fixed" else "32"
        ratio = "--" if view == "Base" else view.split("=")[-1]
        lines.append("| " + " | ".join((size, teacher, view, mode,
                                        f"Qwen3VL-{size}-Instruct", teacher_model, ratio,
                                        batch, f"`{spec.oss_name}`" if spec.oss_name else "--")) + " |")
    lines += [
        "", "训练共同配置：`DATASET_VERSION=full DEGRADATION_MODE=original STUDENT_RATIO=1.0 "
        "ALPHA=1.0 OPD_SELECTIVE_WEIGHT=False ROLLOUT_N=4 TOTAL_EPOCHS=1`；"
        "`PPO_MINI_BATCH_SIZE=TRAIN_BATCH_SIZE`。",
        "4B `tr=1.0` 同尺寸教师并非原表要求的行，不在本清单中。", "",
        "## 全 step 结果", "",
        "| ID | Student | Teacher | View | Step | CHAIRi↓ | CHAIRs↓ | ObjF1↑ | Words/img | Chars/img | Correct mentions | Hall mentions | POPE F1↑ | A-HAL↓ | MMStar↑ | CV-Bench↑ | 状态 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    indexed = []
    for index, row in enumerate(rows, 1):
        row_id = f"S{index:03d}"
        indexed.append((row_id, row))
        values = row["values"]
        cells = [row_id, row["student"], row["teacher"], row["view"],
                 "--" if row["step"] is None else str(row["step"])]
        cells += [inventory.fmt(values.get(key), True) for key in TABLE_METRICS[:3]]
        cells += [inventory.fmt(values.get("words_per_image")),
                  inventory.fmt(values.get("chars_per_image")),
                  str(values.get("correct_mentions", "--")), str(values.get("hall_mentions", "--"))]
        cells += [inventory.fmt(values.get(key), key != "A-HAL") for key in TABLE_METRICS[3:]]
        lines.append("| " + " | ".join(cells + [row["status"]]) + " |")
    lines += ["", "## 待训练", ""]
    missing_training = [(row_id, row) for row_id, row in indexed
                        if row["status"].startswith("待训练")]
    if not missing_training:
        lines.append("无。")
    for row_id, row in missing_training:
        lines.append(f"- {row_id} {row['student']}/{row['teacher']}/{row['view']}: "
                     f"`{row['spec'].oss_name}`")
    lines += ["", "## 待推理或补评测", ""]
    pending_eval = [(row_id, row) for row_id, row in indexed
                    if row["status"] in ("有 checkpoint，待新版 CHAIR 推理", "有 checkpoint，待补其他 benchmark", "Base 待补推理")]
    if not pending_eval:
        lines.append("无。")
    for row_id, row in pending_eval:
        missing = ", ".join(metric for metric in TABLE_METRICS if metric not in row["values"])
        lines.append(f"- {row_id} {row['student']}/{row['teacher']}/{row['view']} "
                     f"step={row['step'] if row['step'] is not None else '--'}: "
                     f"缺 {missing}; checkpoint=`{row['checkpoint']}`")
    lines += ["", "## 待核实", ""]
    pending_check = [(row_id, row) for row_id, row in indexed
                     if "待核实" in row["status"] or "人工核对" in row["status"]]
    if not pending_check:
        lines.append("无。")
    for row_id, row in pending_check:
        lines.append(f"- {row_id} {row['student']}/{row['teacher']}/{row['view']}: "
                     f"{row['status']}; expected=`{row['checkpoint']}`")
        for path in row["incomplete"]:
            lines.append(f"  - incomplete: `{path}`")
        for path in row["aliases"]:
            lines.append(f"  - possible alias: `{path}`")
    extra_incomplete = sorted({str(path) for row in rows if row["step"] is not None
                               for path in row["incomplete"]})
    if extra_incomplete:
        lines += ["", "有完整 step 的实验中还发现不完整上传："]
        lines += [f"- `{path}`" for path in extra_incomplete]
    lines += ["", "## 指标来源", "",
              "| ID | Checkpoint | CHAIR | POPE | AMBER | MMStar | CV-Bench |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row_id, row in indexed:
        sources = row["sources"]
        cells = [row_id, str(row["checkpoint"])] + [str(sources.get(key, "--"))
            for key in ("CHAIRi", "POPE", "A-HAL", "MMStar", "CV-Bench")]
        lines.append("| " + " | ".join(f"`{cell}`" if cell != "--" else cell for cell in cells) + " |")
    if issues:
        lines += ["", "## 冲突与读取警告", ""]
        lines += [f"- {issue}" for issue in sorted(set(issues))]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oss-root", type=Path, default=Path("/data/oss_bucket_0/yanlin/ckpt/OPD/v4"))
    parser.add_argument("--output-md", type=Path,
                        default=Path("res-opd/plan/teacher_scale_tr05_official_v2_inventory.md"))
    args = parser.parse_args()
    rows, issues, mounted = collect(args.oss_root)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render(rows, issues, mounted, args.oss_root), encoding="utf-8")
    counts = {status: sum(row["status"] == status for row in rows)
              for status in sorted({row["status"] for row in rows})}
    print(f"TEACHER_SCALE_TR05=WRITTEN rows={len(rows)} status={counts} "
          f"warnings={len(issues)} md={args.output_md}")


if __name__ == "__main__":
    main()
