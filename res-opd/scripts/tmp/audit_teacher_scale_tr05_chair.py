#!/usr/bin/env python3
"""Audit Table 5 checkpoints and CHAIR generation versus official scoring."""

import argparse
from collections import Counter
from pathlib import Path

import summarize_main_tr05_inventory as inventory
import summarize_teacher_scale_tr05 as teacher_scale


def inspect_raw(path, metrics_path, expected_student_ratio=1.0):
    rows = list(inventory.jsonl(path))
    if (len(rows) != 1000 or any(not isinstance(row, dict) for row in rows)
            or len({row.get("image_id") for row in rows}) != 1000
            or any(not str(row.get("generated_caption") or "").strip()
                   or str(row.get("generated_caption") or "").startswith("[ERROR]")
                   or row.get("final_answer_available") is False for row in rows)):
        return "推理不完整", f"{path} ({len(rows)} rows or invalid captions)"

    metrics = inventory.read_json(metrics_path) if metrics_path.is_file() else {}
    prompts = {row.get("eval_prompt") for row in rows if row.get("eval_prompt") is not None}
    if not prompts and metrics.get("eval_prompt") is not None:
        prompts = {metrics["eval_prompt"]}
    ratios = {str(row.get("eval_student_ratio")) for row in rows
              if row.get("eval_student_ratio") is not None}
    if not ratios and metrics.get("eval_student_ratio") is not None:
        ratios = {str(metrics["eval_student_ratio"])}
    try:
        ratio_matches = bool(ratios) and all(float(value) == expected_student_ratio for value in ratios)
    except ValueError:
        ratio_matches = False
    if prompts != {inventory.CHAIR_PROMPT} or not ratio_matches:
        return ("已推理，协议待核实",
                f"{path} (prompt={sorted(prompts) or 'unknown'}, ratio={sorted(ratios) or 'unknown'})")
    if any(not row.get("gt_captions") for row in rows):
        return "已推理，参考标注待核实", f"{path} (gt_captions missing; cannot directly rescore)"
    return "已推理，待新版计分", str(path)


def inspect_chair(directories, dataset, issues, label, expected_student_ratio=1.0):
    raw_candidates = []
    metrics_only = []
    official_candidates = []
    for directory in directories:
        metrics_path = directory / "chair_metrics.json"
        raw_path = directory / "eval_results.jsonl"
        official_candidates.extend(sorted(directory.glob("chair_metrics*official*.json")))
        try:
            official = inventory.chair_result(directory, expected_student_ratio)
        except (OSError, ValueError, TypeError, KeyError) as error:
            issues.append(f"{label}: {directory}: official CHAIR invalid: {error}")
            official = None
        if official:
            values, source = official
            return "新版指标齐全", values, str(source)
        if raw_path.is_file():
            try:
                raw_candidates.append(inspect_raw(raw_path, metrics_path, expected_student_ratio))
            except (OSError, ValueError, TypeError, KeyError) as error:
                issues.append(f"{label}: {raw_path}: {error}")
                raw_candidates.append(("推理不完整", f"{raw_path} (unreadable)"))
        elif metrics_path.is_file():
            metrics_only.append(str(metrics_path))
    for state in ("已推理，待新版计分", "已推理，参考标注待核实",
                  "已推理，协议待核实", "推理不完整"):
        for status, evidence in raw_candidates:
            if status == state:
                if official_candidates and state == "已推理，待新版计分":
                    return "已推理，新版候选指标待核实", {}, f"{evidence}; candidate={official_candidates[0]}"
                return status, {}, evidence
    if metrics_only:
        return "仅有指标，原始生成缺失", {}, metrics_only[0]
    if official_candidates:
        return "仅有指标，原始生成缺失", {}, str(official_candidates[0])
    shards = sorted(dataset.glob("*/.shards/*/shard_*/eval_results.jsonl"))
    if shards:
        return "仅分片，未合并", {}, str(shards[-1])
    return "未推理", {}, "--"


def collect(oss_root):
    issues = []
    rows = []
    mounted = oss_root.is_dir()
    for student, teacher, view, spec in teacher_scale.table_specs():
        if spec.method == "Base":
            steps = [None]
            complete = set()
            incomplete = {}
            eval_steps = set()
            aliases = []
        else:
            step_dirs = teacher_scale.checkpoint_dirs(spec, oss_root) if mounted else []
            complete = {int(path.name.split("_")[-1]) for path in step_dirs
                        if all((path / name).is_file() and (path / name).stat().st_size > 0
                               for name in ("model.safetensors", "config.json"))}
            incomplete = {int(path.name.split("_")[-1]): path for path in step_dirs
                          if int(path.name.split("_")[-1]) not in complete}
            eval_steps = set(inventory.evaluation_steps(spec))
            steps = sorted(complete | set(incomplete) | eval_steps) or [None]
            aliases = teacher_scale.possible_aliases(
                spec, student, teacher.split()[0], view.split("=")[-1], oss_root
            ) if mounted and not complete and not incomplete else []

        for step in steps:
            if spec.method == "Base":
                dataset = spec.eval_root / spec.experiment / inventory.DATASET
                directories = inventory.base_dirs(spec)
                ckpt_status = "Base"
                checkpoint = spec.experiment
            else:
                dataset = spec.eval_root / f"{spec.experiment}_global_step_{step}" / inventory.DATASET
                directories = inventory.output_dirs(dataset) if step is not None else []
                checkpoint = str(oss_root / spec.oss_name / f"global_step_{step}") if step is not None \
                    else str(oss_root / spec.oss_name)
                if not mounted:
                    ckpt_status = "OSS未挂载，待核实"
                elif step in complete:
                    ckpt_status = "完整"
                elif step in incomplete:
                    ckpt_status = "上传不完整，待核实"
                elif step in eval_steps:
                    ckpt_status = "仅评测目录，checkpoint待核实"
                elif aliases:
                    ckpt_status = "疑似其他命名，待核实"
                elif (oss_root / spec.oss_name).is_dir():
                    ckpt_status = "训练目录存在但无完整step，待核实"
                else:
                    ckpt_status = "待训练候选（规范名称下无checkpoint）"

            chair_status, values, evidence = inspect_chair(
                directories, dataset, issues, f"{student}/{teacher}/{view}/s{step}")
            rows.append({"spec": spec, "student": student, "teacher": teacher, "view": view,
                         "step": step, "checkpoint": checkpoint, "ckpt_status": ckpt_status,
                         "chair_status": chair_status, "values": values, "chair_evidence": evidence,
                         "aliases": aliases})
    return rows, issues, mounted


def render(rows, issues, mounted, oss_root):
    lines = [
        "# Table 5 teacher-scale CHAIR audit: tr=0.50", "",
        "原表的四组 tr=0.75 改为 tr=0.50；三组 tr=1.0 不变，另有两个 Base。",
        "同尺寸为 frozen 原模型教师，跨尺寸为 fixed 大模型教师；均为 uniform RKL，无 RiskMask。",
        "学生使用 original (sr=1.0)。逐一列出实际 step，不按分数挑选。",
        "新版 CHAIR 要求 1000 张、`Describe the image.`、original 学生视图和 official caption/mention schema。",
        "`已推理，待新版计分` 表示完整生成可核查；prompt/视图/参考标注不明者单列，不能直接重算。",
        "只检查 CHAIR；不读取 POPE、AMBER、MMStar、CV-Bench。",
        f"OSS 挂载：`{oss_root}` {'可用' if mounted else '不可用，不能判断训练缺口'}。", "",
        "## 各 step CHAIR 状态", "",
        "| ID | Student | Teacher | View | Step | Checkpoint | CHAIR | CHAIRi↓ | CHAIRs↓ | ObjF1↑ | Words/img | Chars/img | Correct mentions | Hall mentions |",
        "| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    indexed = []
    for index, row in enumerate(rows, 1):
        row_id = f"S{index:03d}"
        indexed.append((row_id, row))
        values = row["values"]
        cells = [row_id, row["student"], row["teacher"], row["view"],
                 "--" if row["step"] is None else str(row["step"]),
                 row["ckpt_status"], row["chair_status"]]
        cells += [inventory.fmt(values.get(key), True) for key in ("CHAIRi", "CHAIRs", "ObjF1")]
        cells += [inventory.fmt(values.get("words_per_image")),
                  inventory.fmt(values.get("chars_per_image")),
                  str(values.get("correct_mentions", "--")),
                  str(values.get("hall_mentions", "--"))]
        lines.append("| " + " | ".join(cells) + " |")

    groups = (
        ("待训练候选", lambda row: row["ckpt_status"].startswith("待训练候选")),
        ("未推理", lambda row: row["chair_status"] == "未推理" and row["ckpt_status"] in ("完整", "Base")),
        ("已推理但无合规新版指标", lambda row: row["chair_status"].startswith("已推理")),
        ("推理未完成或原始生成缺失", lambda row: row["chair_status"] in
         ("推理不完整", "仅有指标，原始生成缺失", "仅分片，未合并")),
        ("checkpoint 待核实", lambda row: "待核实" in row["ckpt_status"]),
    )
    for title, predicate in groups:
        lines += ["", f"## {title}", ""]
        selected = [(row_id, row) for row_id, row in indexed if predicate(row)]
        if not selected:
            lines.append("无。")
        for row_id, row in selected:
            lines.append(f"- {row_id} {row['student']}/{row['teacher']}/{row['view']} "
                         f"step={row['step'] if row['step'] is not None else '--'}; "
                         f"checkpoint={row['ckpt_status']}; CHAIR={row['chair_status']}; "
                         f"ckpt=`{row['checkpoint']}`; evidence=`{row['chair_evidence']}`")
            for path in row["aliases"]:
                lines.append(f"  - possible alias: `{path}`")
    lines += ["", "## CHAIR 文件来源", "",
              "| ID | Checkpoint | CHAIR evidence |",
              "| --- | --- | --- |"]
    for row_id, row in indexed:
        lines.append(f"| {row_id} | `{row['checkpoint']}` | `{row['chair_evidence']}` |")
    if issues:
        lines += ["", "## 读取警告", ""]
        lines += [f"- {issue}" for issue in sorted(set(issues))]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oss-root", type=Path, default=Path("/data/oss_bucket_0/yanlin/ckpt/OPD/v4"))
    parser.add_argument("--output-md", type=Path,
                        default=Path("res-opd/plan/teacher_scale_tr05_chair_audit.md"))
    args = parser.parse_args()
    rows, issues, mounted = collect(args.oss_root)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render(rows, issues, mounted, args.oss_root), encoding="utf-8")
    print(f"TEACHER_SCALE_CHAIR_AUDIT=WRITTEN rows={len(rows)} mounted={mounted} "
          f"checkpoint={dict(Counter(row['ckpt_status'] for row in rows))} "
          f"chair={dict(Counter(row['chair_status'] for row in rows))} "
          f"warnings={len(issues)} md={args.output_md}")


if __name__ == "__main__":
    main()
