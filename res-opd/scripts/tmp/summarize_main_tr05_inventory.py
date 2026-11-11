#!/usr/bin/env python3
"""Audit tr=0.5 main-table results against exact checkpoints and raw eval files."""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CHAIR_SCHEMA = "official_chair_caption_and_mention_v1"
CHAIR_PROMPT = "Describe the image."
DATASET = "train5000_test1000_original_sr1p0"
METHODS = ("Base", "Frozen RKL", "Low-res SFT", "Res-OPSD", "RandomMask", "EntropyMask")
METRICS = ("CHAIRi", "CHAIRs", "ObjF1", "POPE", "A-CHAIR", "A-HAL", "MMStar", "CV-Bench")
WORD_RE = re.compile(r"\b\w+(?:[-']\w+)*\b")
STEP_RE = re.compile(r"global_step_(\d+)$")


@dataclass(frozen=True)
class Spec:
    family: str
    size: str
    method: str
    experiment: str
    oss_name: str = ""

    @property
    def label(self):
        return f"{self.family}-{self.size}/{self.method}"

    @property
    def eval_root(self):
        tag = "internvl3" if self.family == "InternVL3" else "instruct"
        return ROOT / "res-opd/eval_results" / tag / "full"


def build_specs():
    specs = []
    for size in ("2B", "4B", "8B"):
        model = f"Qwen3VL-{size}-Instruct"
        specs.append(Spec("Qwen3VL", size, "Base", model))
        batch = "b16" if size == "8B" else "b32"
        prefix = f"Res-OPD-{model}-orig-sr1.0-tr0.5-"
        oss_prefix = "ResOPD_" + (f"Qwen3VL_{size}_Instruct_" if size != "2B" else "") + "orig_sr1.0_tr0.5_"
        settings = (
            ("Frozen RKL", f"a1.0-frozen-rkl-{batch}-rn4-full5k-e1",
             f"a1.0_frozen_rkl_{batch}_rn4_full5k-e1"),
            ("Low-res SFT", f"sft-lowres-{batch}-full5k-e1",
             f"sft_lowres_{batch}_full5k-e1"),
            ("Res-OPSD", f"a1.0-frozen-rkl-riskmask-nll-p30-{batch}-rn4-full5k-e1",
             f"a1.0_frozen_rkl_riskmask_nll_p30_{batch}_rn4_full5k-e1"),
            ("RandomMask", f"a1.0-frozen-rkl-randommask-p30-{batch}-rn4-full5k-e1",
             f"a1.0_frozen_rkl_randommask_p30_{batch}_rn4_full5k-e1"),
            ("EntropyMask", f"a1.0-frozen-rkl-entropymask-p30-{batch}-rn4-full5k-e1",
             f"a1.0_frozen_rkl_entropymask_p30_{batch}_rn4_full5k-e1"),
        )
        specs.extend(Spec("Qwen3VL", size, method, prefix + local, oss_prefix + oss)
                     for method, local, oss in settings)
    model = "InternVL3-2B-hf"
    specs.append(Spec("InternVL3", "2B", "Base", model))
    prefix = "Res-OPD-InternVL3-2B-hf-orig-sr1.0-tr0.5-"
    oss_prefix = "ResOPD_InternVL3_2B_hf_orig_sr1.0_tr0.5_"
    settings = (
        ("Frozen RKL", "a1.0-frozen-rkl-b32-rn4-full5k-e1", "a1.0_frozen_rkl_b32_rn4_full5k-e1"),
        ("Low-res SFT", "sft-lowres-b32-full5k-e1", "sft_lowres_b32_full5k-e1"),
        ("RandomMask", "a1.0-frozen-rkl-randommask-p30-b32-rn4-full5k-e1",
         "a1.0_frozen_rkl_randommask_p30_b32_rn4_full5k-e1"),
        ("EntropyMask", "a1.0-frozen-rkl-entropymask-p30-b32-rn4-full5k-e1",
         "a1.0_frozen_rkl_entropymask_p30_b32_rn4_full5k-e1"),
    )
    specs.extend(Spec("InternVL3", "2B", method, prefix + local, oss_prefix + oss)
                 for method, local, oss in settings)
    specs.append(Spec("InternVL3", "2B", "Res-OPSD",
                      "Res-OPD-InternVL3-2B-hf-tr05-fixedmb1-full5k-e1",
                      "ResOPD_InternVL3_2B_hf_tr05_fixedmb1_full5k-e1"))
    return specs


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                yield json.loads(raw)


def read_judge_rows(path):
    try:
        rows = read_json(path)
    except json.JSONDecodeError:
        rows = list(jsonl(path))
    return rows if isinstance(rows, list) else []


def chair_result(directory, expected_student_ratio=1.0):
    metrics_path = directory / "chair_metrics.json"
    rows_path = directory / "eval_results.jsonl"
    if not metrics_path.is_file() or not rows_path.is_file():
        return None
    metrics = read_json(metrics_path)
    if (metrics.get("metric_schema") != CHAIR_SCHEMA or
            metrics.get("eval_prompt") != CHAIR_PROMPT or
            metrics.get("total_results") != 1000 or
            metrics.get("total_captions") != 1000 or
            metrics.get("invalid_final_answer", 0) != 0):
        return None
    correct = int(metrics["total_correct_mentions"])
    hall = int(metrics["total_hallucinated_mentions"])
    total = int(metrics["total_object_mentions"])
    if (correct + hall != total or
            abs(hall / max(total, 1) - metrics["CHAIRi"]) > 1e-9 or
            abs(metrics["hallucinated_captions"] / 1000 - metrics["CHAIRs"]) > 1e-9):
        return None
    rows = list(jsonl(rows_path))
    if len(rows) != 1000 or len({row.get("image_id") for row in rows}) != 1000:
        return None
    if any(row.get("eval_prompt") not in (None, CHAIR_PROMPT) or
           float(row.get("eval_student_ratio", 1.0)) != expected_student_ratio for row in rows):
        return None
    texts = [str(row.get("generated_caption") or "").rsplit("</think>", 1)[-1].strip() for row in rows]
    if any(not value for value in texts):
        return None
    values = {key: metrics[key] for key in ("CHAIRi", "CHAIRs", "ObjF1")}
    values.update({
        "chars_per_image": sum(map(len, texts)) / 1000,
        "words_per_image": sum(len(WORD_RE.findall(text)) for text in texts) / 1000,
        "correct_mentions": correct,
        "hall_mentions": hall,
    })
    return values, metrics_path


def pope_result(directory):
    path = directory / "pope/pope_summary.json"
    if not path.is_file():
        return None
    data = read_json(path)
    splits = ("pope_adv", "pope_pop", "pope_random")
    if (not isinstance(data.get("macro_f1"), (int, float)) or
            any(split not in (data.get("benchmarks") or {}) for split in splits) or
            any(sum(1 for _ in jsonl(directory / "pope" / split / "eval_results.jsonl")) != 5223
                for split in splits)):
        return None
    return {"POPE": data["macro_f1"]}, path


def amber_result(directory):
    path = directory / "amber/amber_metrics.json"
    if not path.is_file():
        return None
    data = read_json(path)
    metrics = data.get("metrics") or {}
    if (data.get("official_returncode") != 0 or data.get("evaluation_type") != "a" or
            any(not isinstance(metrics.get(key), (int, float))
                for key in ("generative_task_chair", "generative_task_hal"))):
        return None
    return {"A-CHAIR": metrics["generative_task_chair"],
            "A-HAL": metrics["generative_task_hal"]}, path


def vision_result(directory, benchmark):
    paths = list((directory / benchmark / "judge").glob("*_answer.jsonl"))
    paths += list((directory / "vision_opd/judge" / benchmark).glob("*_answer.jsonl"))
    if len(paths) != 1:
        return None
    rows = read_judge_rows(paths[0])
    expected = 1500 if benchmark == "mmstar" else 2638
    if len(rows) != expected or any(str(row.get("judge", "")).strip().lower() not in ("yes", "no") for row in rows):
        return None
    value = sum(str(row["judge"]).strip().lower() == "yes" for row in rows) / expected
    source = paths[0]
    if benchmark == "cv-bench":
        breakdown = paths[0].parent.parent / "cvbench_breakdown.json"
        if breakdown.is_file():
            data = read_json(breakdown)
            if isinstance(data.get("accuracy"), (int, float)):
                value, source = data["accuracy"], breakdown
    return {"MMStar" if benchmark == "mmstar" else "CV-Bench": value}, source


def output_dirs(dataset):
    if not dataset.is_dir():
        return []
    children = [path for path in dataset.iterdir() if path.is_dir() and path.name != ".shards"]
    priority = ("three_tables_fill_official_v2", "official_v2", "chair_official_v2",
                "chair_describe_image_official")
    children.sort(key=lambda path: (priority.index(path.name) if path.name in priority else len(priority), path.name))
    return children + [dataset]


def collect_metrics(directories, issues, label):
    values, sources = {}, {}
    for directory in directories:
        for reader in (chair_result, pope_result, amber_result):
            try:
                result = reader(directory)
            except (OSError, ValueError, TypeError, KeyError) as error:
                issues.append(f"{label}: {directory}: {error}")
                continue
            if result:
                metrics, source = result
                for key, value in metrics.items():
                    if key in values and isinstance(value, (int, float)) and isinstance(values[key], (int, float)):
                        if abs(value - values[key]) > 1e-6:
                            issues.append(f"{label}: conflicting {key}: {sources[key]} vs {source}")
                    elif key not in values:
                        values[key], sources[key] = value, source
        for benchmark in ("mmstar", "cv-bench"):
            try:
                result = vision_result(directory, benchmark)
            except (OSError, ValueError, TypeError, KeyError) as error:
                issues.append(f"{label}: {directory}/{benchmark}: {error}")
                continue
            if result:
                metrics, source = result
                for key, value in metrics.items():
                    if key in values and abs(value - values[key]) > 1e-6:
                        issues.append(f"{label}: conflicting {key}: {sources[key]} vs {source}")
                    elif key not in values:
                        values[key], sources[key] = value, source
    return values, sources


def checkpoint_steps(spec, oss_root):
    parent = oss_root / spec.oss_name
    if not parent.is_dir():
        return []
    found = []
    for path in parent.iterdir():
        match = STEP_RE.fullmatch(path.name)
        if match and path.is_dir() and all((path / name).is_file() and (path / name).stat().st_size > 0
                                           for name in ("model.safetensors", "config.json")):
            found.append(int(match.group(1)))
    return sorted(found)


def evaluation_steps(spec):
    result = []
    for path in spec.eval_root.glob(f"{spec.experiment}_global_step_*"):
        match = re.fullmatch(rf"{re.escape(spec.experiment)}_global_step_(\d+)", path.name)
        if path.is_dir() and match:
            result.append(int(match.group(1)))
    return sorted(set(result))


def base_dirs(spec):
    if spec.family == "Qwen3VL":
        return output_dirs(spec.eval_root / spec.experiment / DATASET)
    legacy = ROOT / "res-opd/eval_results/internvl/full/InternVL3-2B-hf/base_chair_original_1000_8gpu"
    return output_dirs(spec.eval_root / spec.experiment / DATASET) + [legacy] + output_dirs(legacy)


def collect_specs(specs, oss_root, issues):
    mounted = oss_root.is_dir()
    rows = []
    for spec in specs:
        if spec.method == "Base":
            values, sources = collect_metrics(base_dirs(spec), issues, spec.label)
            rows.append({"spec": spec, "step": None, "checkpoint": "Base", "values": values,
                         "sources": sources, "dataset": spec.experiment,
                         "status": "齐全" if all(key in values for key in METRICS) else "待补推理"})
            continue
        ckpt_steps = checkpoint_steps(spec, oss_root) if mounted else []
        eval_steps = evaluation_steps(spec)
        steps = sorted(set(ckpt_steps) | set(eval_steps))
        if not steps:
            status = "待训练（规范名称下无checkpoint）" if mounted else "OSS未挂载，待核实checkpoint"
            rows.append({"spec": spec, "step": None, "checkpoint": status, "values": {},
                         "sources": {}, "dataset": "--", "status": status})
            continue
        for step in steps:
            dataset = (spec.eval_root / f"{spec.experiment}_global_step_{step}" / DATASET)
            values, sources = collect_metrics(output_dirs(dataset), issues, f"{spec.label}/s{step}")
            checkpoint = str(oss_root / spec.oss_name / f"global_step_{step}")
            if not mounted:
                status = "OSS未挂载，需核实checkpoint"
            elif step not in ckpt_steps:
                status = "仅发现旧推理目录，checkpoint待核实"
            elif all(key in values for key in METRICS):
                status = "齐全"
            else:
                status = "待补推理"
            rows.append({"spec": spec, "step": step, "checkpoint": checkpoint if step in ckpt_steps else status,
                         "values": values, "sources": sources, "dataset": str(dataset), "status": status})
    return rows, mounted


def external_rows(issues, eval_root=None, include_presentation_halc=False):
    root = eval_root or ROOT / "res-opd/eval_results"
    configs = []
    for pattern in ("external_baselines/**/*_config.json", "chair_external_*/*_config.json"):
        configs.extend(root.glob(pattern))
    rows = []
    seen = set()
    for config_path in sorted(set(configs)):
        relative = config_path.relative_to(root)
        presentation = any(part.startswith("presentation_") for part in relative.parts)
        if (any(part.startswith(".") for part in relative.parts) or
                (presentation and not include_presentation_halc) or
                config_path.parent in seen):
            continue
        try:
            config = read_json(config_path)
        except (OSError, ValueError) as error:
            issues.append(f"External config {config_path}: {error}")
            continue
        if ((config.get("presentation_only") or config.get("not_for_paper"))
                and not (presentation and include_presentation_halc)):
            continue
        model = config.get("model_name") or Path(config.get("model_path", "")).name
        if model not in {"Qwen3VL-2B-Instruct", "Qwen3VL-4B-Instruct", "Qwen3VL-8B-Instruct", "InternVL3-2B-hf"}:
            issues.append(f"External result needs exact model identity: {config_path}")
            continue
        method = config_path.name.split("_config.json", 1)[0].upper()
        if method not in ("VCD", "OPERA", "HALC"):
            continue
        values, sources = collect_metrics([config_path.parent], issues, f"{model}/{method}")
        family = "InternVL3" if model.startswith("InternVL") else "Qwen3VL"
        size = "2B" if family == "InternVL3" else model.split("-")[1]
        spec = Spec(family, size, method + " (external)", model)
        status = "齐全" if all(key in values for key in METRICS) else "待补推理"
        if presentation:
            status = "混合解码展示（8192+2048）" if status == "齐全" else "混合解码展示，待补推理"
        rows.append({"spec": spec, "step": None, "checkpoint": "base model", "values": values,
                     "sources": sources, "dataset": str(config_path.parent), "status": status,
                     "config": str(config_path)})
        seen.add(config_path.parent)
    for family, size, model in (("Qwen3VL", "2B", "Qwen3VL-2B-Instruct"),
                                ("InternVL3", "2B", "InternVL3-2B-hf")):
        for method in ("VCD", "OPERA", "HALC"):
            if not any(row["spec"].family == family and row["spec"].size == size and
                       row["spec"].method == method + " (external)" for row in rows):
                rows.append({"spec": Spec(family, size, method + " (external)", model),
                             "step": None, "checkpoint": "base model", "values": {}, "sources": {},
                             "dataset": "--", "status": "待推理/适配"})
    return rows


def fmt(value, percent=False):
    if value is None or isinstance(value, bool):
        return "--"
    return f"{value * 100 if percent else value:.2f}"


def render(rows, externals, issues, mounted, oss_root, include_presentation_halc=False):
    chair_note = (
        "本次显式纳入 HALC 的展示性混合 CHAIR：旧 8192-token 行与 2048-token 续跑行合并；"
        "对应行会标为“混合解码展示”，不可与统一解码上限结果混作正式比较。"
        if include_presentation_halc else
        "每个指标均有独立来源，不把旧 CHAIR 与新 CHAIR 混用。"
    )
    lines = ["# 主实验 tr=0.50 全 step 结果与缺口", "",
             "仅使用同一 checkpoint 的原图 student 推理结果；所有已有 step 均列出，不按指标挑选 checkpoint。",
             "Checkpoint 以实验脚本生成的规范名称精确匹配；未找到仅说明该名称下没有完整模型，不排除历史别名。",
             "CHAIR 仅收录 `Describe the image.`、1000 张图、official caption/mention 口径；",
             "POPE 为三个 split 的 macro F1；AMBER 为官方 generative A-CHAIR/A-HAL；",
             "MMStar/CV-Bench 为完整 judge 输出。CHAIR/POPE/MMStar/CV-Bench 显示百分数，AMBER 保留点数。",
             "Words/img 和 Chars/img 来自最终 `generated_caption`，不是 tokenizer token 数；correct/hall mentions 均包含重复 mention。",
             "`--` 表示未找到可核实的结果，不是零。" + chair_note,
             "同一指标若有多个不同的有效结果，使用表中来源列所列的优先结果，同时在末尾报告冲突。", "",
             f"OSS 挂载检查：`{oss_root}` {'可用' if mounted else '不可用；不能据此断言缺少训练'}。", "",
             "## 逐 checkpoint 主表", "",
             "| ID | 家族 | 尺寸 | 方法 | Step | CHAIRi↓ | CHAIRs↓ | ObjF1↑ | Words/img | Chars/img | Correct mentions | Hall mentions | POPE F1↑ | A-CHAIR↓ | A-HAL↓ | MMStar↑ | CV-Bench↑ | 状态 |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    indexed = []
    for index, row in enumerate(rows + externals, 1):
        row_id = f"M{index:03d}"
        indexed.append((row_id, row))
        spec, values = row["spec"], row["values"]
        cells = [row_id, spec.family, spec.size, spec.method,
                 "--" if row["step"] is None else str(row["step"])]
        cells += [fmt(values.get(key), True) for key in METRICS[:3]]
        cells += [fmt(values.get("words_per_image")), fmt(values.get("chars_per_image")),
                  str(values.get("correct_mentions", "--")), str(values.get("hall_mentions", "--"))]
        cells += [fmt(values.get(key), key not in ("A-CHAIR", "A-HAL")) for key in METRICS[3:]]
        lines.append("| " + " | ".join(cells + [row["status"]]) + " |")
    lines += ["", "## 待补任务", ""]
    for status, title in (("待训练", "无 checkpoint：待训练"),
                          ("待补推理", "有 checkpoint 或 Base：待补推理"),
                          ("待推理/适配", "外部 baseline：待推理或适配"),
                          ("checkpoint待核实", "有旧推理目录但 checkpoint 未核实"),
                          ("OSS未挂载", "checkpoint 尚未核实")):
        selected = [(row_id, row) for row_id, row in indexed if status in row["status"]]
        lines += [f"### {title}", ""]
        if not selected:
            lines.append("无。")
        for row_id, row in selected:
            missing = ", ".join(key for key in METRICS if key not in row["values"])
            step = "--" if row["step"] is None else str(row["step"])
            lines.append(f"- {row_id} {row['spec'].label} step={step}: 缺 {missing}; checkpoint={row['checkpoint']}")
        lines.append("")
    lines += ["## 指标来源", "",
              "| ID | Checkpoint | CHAIR | POPE | AMBER | MMStar | CV-Bench |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row_id, row in indexed:
        sources = row["sources"]
        cells = [row_id, str(row["checkpoint"]),
                 str(sources.get("CHAIRi", "--")), str(sources.get("POPE", "--")),
                 str(sources.get("A-CHAIR", "--")), str(sources.get("MMStar", "--")),
                 str(sources.get("CV-Bench", "--"))]
        lines.append("| " + " | ".join(f"`{cell}`" if cell != "--" else cell for cell in cells) + " |")
    if issues:
        lines += ["", "## 冲突与读取警告", ""]
        lines += [f"- {issue}" for issue in sorted(set(issues))]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oss-root", type=Path, default=Path("/data/oss_bucket_0/yanlin/ckpt/OPD/v4"))
    parser.add_argument("--output-md", type=Path, default=Path("res-opd/plan/main_tr05_results_inventory_zh.md"))
    parser.add_argument("--include-presentation-halc", action="store_true",
                        help="Include exploratory HALC CHAIR rows mixed from legacy 8192-token and 2048-token continuation outputs")
    args = parser.parse_args()
    issues = []
    rows, mounted = collect_specs(build_specs(), args.oss_root, issues)
    externals = external_rows(issues, include_presentation_halc=args.include_presentation_halc)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render(rows, externals, issues, mounted, args.oss_root,
                                    args.include_presentation_halc), encoding="utf-8")
    print(f"MAIN_TR05_INVENTORY=WRITTEN rows={len(rows)} external={len(externals)} "
          f"oss_mounted={mounted} warnings={len(issues)} path={args.output_md}")


if __name__ == "__main__":
    main()
