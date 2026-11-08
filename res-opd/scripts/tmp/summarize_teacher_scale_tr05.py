#!/usr/bin/env python3
"""Inventory Table 5 teacher-scale controls from exact checkpoints and eval files."""

import argparse
from pathlib import Path

import summarize_main_tr05_inventory as inventory


TABLE_METRICS = ("CHAIRi", "CHAIRs", "ObjF1", "POPE", "A-HAL", "MMStar", "CV-Bench")


def experiment_spec(student, teacher, ratio):
    model = f"Qwen3VL-{student}-Instruct"
    batch = "b16" if student == "4B" and teacher == "8B" else "b32"
    local_prefix = f"Res-OPD-{model}-orig-sr1.0-tr{ratio}-a1.0-"
    oss_prefix = "ResOPD_" + ("Qwen3VL_4B_Instruct_" if student == "4B" else "")
    oss_prefix += f"orig_sr1.0_tr{ratio}_a1.0_"
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


def status_for(row, mounted):
    values = row["values"]
    spec = row["spec"]
    if spec.method == "Base":
        return "齐全" if all(metric in values for metric in TABLE_METRICS) else "Base 待补推理"
    if not mounted:
        return "OSS 未挂载，checkpoint 待核实"
    if row["step"] is None:
        return "未训练（规范名称下无 checkpoint）"
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
        for row in found:
            if spec.method != "Base" and row["step"] is None:
                row["checkpoint"] = f"expected:{oss_root / spec.oss_name}"
            rows.append({**row, "student": student, "teacher": teacher, "view": view,
                         "status": status_for(row, mounted)})
    return rows, issues, oss_root.is_dir()


def render(rows, issues, mounted, oss_root):
    lines = [
        "# Table 5 teacher-scale controls: tr=0.50", "",
        "仅替换原表 tr=0.75 为 tr=0.50；tr=1.0 行不变。列出全部实际发现的 step，不按评测分数挑选 checkpoint。",
        "实验目录与 OSS checkpoint 使用规范名称精确匹配；缺失只表示该规范名称下未发现完整 checkpoint。",
        "CHAIR 仅接受 `Describe the image.`、1000 张图、official caption/mention 口径；旧 CHAIR 不填入新表。",
        "POPE 是三个 split 的 macro F1；AMBER A-HAL 保留官方点数；其他指标显示百分数。",
        "Words/img、Chars/img 来自最终回答，Correct/Hall mentions 包含重复 mention；`--` 表示未核实，不是零。",
        f"OSS 挂载：`{oss_root}` {'可用' if mounted else '不可用，不能判断是否未训练'}。", "",
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
    lines += ["", "## 待补清单", ""]
    for title, predicate in (
        ("未训练", lambda status: status.startswith("未训练")),
        ("有 checkpoint，待新版 CHAIR 推理", lambda status: status.startswith("有 checkpoint，待新版")),
        ("有 checkpoint，待补其他 benchmark", lambda status: status.startswith("有 checkpoint，待补")),
        ("Base 或 checkpoint 待核实", lambda status: "待核实" in status or status.startswith("Base ")),
    ):
        lines += [f"### {title}", ""]
        selected = [(row_id, row) for row_id, row in indexed if predicate(row["status"])]
        if not selected:
            lines.append("无。")
        for row_id, row in selected:
            missing = ", ".join(key for key in TABLE_METRICS if key not in row["values"])
            lines.append(f"- {row_id} {row['student']}/{row['teacher']}/{row['view']} "
                         f"step={row['step'] if row['step'] is not None else '--'}: "
                         f"缺 {missing}; checkpoint=`{row['checkpoint']}`")
        lines.append("")
    lines += ["## 指标来源", "",
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
