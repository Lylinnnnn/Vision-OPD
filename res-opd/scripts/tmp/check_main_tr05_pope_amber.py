#!/usr/bin/env python3
"""Audit POPE and AMBER generation and scoring for selected tr=0.5 rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from summarize_main_tr05_inventory import DATASET, ROOT, build_specs
import audit_teacher_scale_tr05_chair as teacher_chair
import summarize_main_tr05_inventory as inventory
import summarize_teacher_scale_tr05 as teacher_scale


SPLITS = ("pope_adv", "pope_pop", "pope_random")
EXPECTED_POPE = 5223
EXPECTED_AMBER = 15220
VISION_EXPECTED = {"mmstar": 1500, "cv-bench": 2638}
BAD_PREFIXES = ("[ERROR]", "[API_ERROR]", "[FUTURE_ERROR]")
EXTERNAL_SUFFIX = {
    "HALC": "chair_external_halc_qwen3vl_adapted_official",
    "OPERA": "chair_external_opera_official",
}


@dataclass(frozen=True)
class Target:
    row_id: str
    size: str
    method: str
    step: int | None = None


TARGETS = (
    Target("Base-2B", "2B", "Base"),
    Target("M005", "2B", "Frozen RKL", 156),
    Target("M006", "2B", "Low-res SFT", 156),
    Target("M014", "2B", "RandomMask", 156),
    Target("M018", "2B", "EntropyMask", 156),
    Target("M088", "2B", "HALC"),
    Target("OPERA-2B", "2B", "OPERA"),
    Target("Base-4B", "4B", "Base"),
    Target("M021", "4B", "Frozen RKL", 100),
    Target("M024", "4B", "Low-res SFT", 156),
    Target("M020", "4B", "Frozen RKL", 50),
    Target("M025", "4B", "Res-OPSD", 50),
    Target("M028", "4B", "Res-OPSD", 156),
    Target("Base-8B", "8B", "Base"),
    Target("M032", "8B", "Frozen RKL", 50),
    Target("M041", "8B", "Low-res SFT", 150),
    Target("M039", "8B", "Low-res SFT", 50),
    Target("M055", "8B", "RandomMask", 150),
    Target("M064", "8B", "EntropyMask", 250),
)


@dataclass
class Inspection:
    state: str
    count: str
    values: tuple[str, ...]
    source: Path | None = None
    detail: str = ""


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def records(path, key):
    if not path.is_file():
        return set(), 0, 0
    ids, bad, total = set(), 0, 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            identity = row.get(key)
            if identity is None and key == "sample_uid":
                identity = row.get("id") or row.get("question_id")
            if identity is None:
                bad += 1
            else:
                ids.add(str(identity))
            answer = str(row.get("model_answer") or "").strip()
            if not answer or answer.startswith(BAD_PREFIXES):
                bad += 1
    return ids, bad, total


def pope_metrics(directory):
    path = directory / "pope/pope_summary.json"
    if not path.is_file():
        return None
    data = read_json(path)
    splits = data.get("benchmarks") or {}
    values = [splits.get(split, {}).get("f1") for split in SPLITS]
    macro = data.get("macro_f1")
    sizes = [splits.get(split, {}).get("num_samples") for split in SPLITS]
    if (not isinstance(macro, (int, float)) or isinstance(macro, bool) or
            any(not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in values) or
            any(not 0 <= value <= 1 for value in values) or
            any(size not in (None, EXPECTED_POPE) for size in sizes) or
            abs(sum(values) / len(values) - macro) > 1e-6):
        return None
    return f"{macro * 100:.2f}", path


def amber_metrics(directory):
    path = directory / "amber/amber_metrics.json"
    if not path.is_file():
        return None
    data = read_json(path)
    metrics = data.get("metrics") or {}
    values = [metrics.get(name) for name in ("generative_task_chair", "generative_task_hal")]
    if (data.get("official_returncode") != 0 or data.get("evaluation_type") != "a" or
            any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values)):
        return None
    return tuple(f"{value:.2f}" for value in values), path


def amber_responses(directory):
    path = directory / "amber/amber_a_responses.json"
    if not path.is_file():
        return 0, 0
    rows = read_json(path)
    if not isinstance(rows, list):
        return 0, 1
    ids = set()
    bad = 0
    for row in rows:
        identity = row.get("id")
        if identity is None:
            bad += 1
        else:
            ids.add(str(identity))
        response = str(row.get("response") or "").strip()
        if not response or response.startswith(BAD_PREFIXES):
            bad += 1
    return len(ids), bad


def inspect_pope(directory, expected=EXPECTED_POPE):
    counts, bad = [], 0
    for split in SPLITS:
        ids, failures, total = records(directory / "pope" / split / "eval_results.jsonl", "sample_uid")
        counts.append(len(ids))
        bad += failures + total - len(ids)
    complete = all(count == expected for count in counts) and bad == 0
    try:
        metrics = pope_metrics(directory)
    except (OSError, ValueError, TypeError, KeyError):
        metrics = None
    count_text = "/".join(str(count) for count in counts)
    if complete and metrics:
        return Inspection("已推理有结果", count_text, (metrics[0],), metrics[1])
    if complete:
        return Inspection("已推理待计算", count_text, ("--",), directory,
                          "三个 split 原始结果齐全，但汇总缺失或无效")
    if any(counts):
        return Inspection("推理未完成", count_text, ("--",), directory,
                          f"bad_or_duplicate={bad}")
    if metrics:
        return Inspection("指标孤立需核验", count_text, (metrics[0],), metrics[1],
                          "有汇总，但没有可核验的原始推理")
    return Inspection("未推理", count_text, ("--",), directory)


def inspect_amber(directory, expected=EXPECTED_AMBER):
    ids, bad, total = records(directory / "amber/raw_results.jsonl", "id")
    try:
        responses, response_bad = amber_responses(directory)
    except (OSError, ValueError, TypeError, KeyError):
        responses, response_bad = 0, 1
    raw_complete = len(ids) == expected and total == expected and bad == 0
    responses_complete = responses == expected and response_bad == 0
    try:
        metrics = amber_metrics(directory)
    except (OSError, ValueError, TypeError, KeyError):
        metrics = None
    count_text = f"{len(ids)}/{responses}"
    if raw_complete and responses_complete and metrics:
        return Inspection("已推理有结果", count_text, metrics[0], metrics[1])
    if raw_complete and responses_complete:
        return Inspection("已推理待计算", count_text, ("--", "--"), directory,
                          "原始结果和官方响应齐全，但官方指标缺失或无效")
    if raw_complete:
        return Inspection("已推理待合并计算", count_text, ("--", "--"), directory,
                          "原始回答齐全，但官方响应未完成")
    if ids or responses:
        return Inspection("推理未完成", count_text, ("--", "--"), directory,
                          f"raw_bad_or_duplicate={bad + total - len(ids)}, response_bad={response_bad}")
    if metrics:
        return Inspection("指标孤立需核验", count_text, metrics[0], metrics[1],
                          "有官方指标，但没有可核验的原始推理")
    return Inspection("未推理", count_text, ("--", "--"), directory)


def shard_outputs(directory):
    for parent in (directory / ".shards", directory / ".aux_shards"):
        if not parent.is_dir():
            continue
        for run in parent.iterdir():
            if run.is_dir():
                outputs = sorted(run.glob("shard_*/out"))
                if outputs:
                    yield run, outputs


def inspect_shards(directory, benchmark, expected):
    best = None
    for run, outputs in shard_outputs(directory):
        if benchmark == "pope":
            counts = []
            bad = 0
            for split in SPLITS:
                ids = set()
                for output in outputs:
                    found, failures, total = records(output / "pope" / split / "eval_results.jsonl", "sample_uid")
                    ids.update(found)
                    bad += failures + total - len(found)
                counts.append(len(ids))
            count = sum(counts)
            text = "/".join(map(str, counts))
            complete = all(value == expected for value in counts) and bad == 0
            values = ("--",)
        else:
            ids = set()
            bad = 0
            for output in outputs:
                found, failures, total = records(output / "amber/raw_results.jsonl", "id")
                ids.update(found)
                bad += failures + total - len(found)
            count = len(ids)
            text = str(count)
            complete = count == expected and bad == 0
            values = ("--", "--")
        state = "已推理待合并计算" if complete else "推理未完成"
        candidate = Inspection(state, text, values, run, "仅发现 shard 原始结果")
        if count and (best is None or count > best[0]):
            best = count, candidate
    return best[1] if best else None


def inspect_all(directories, benchmark, expected):
    reader = inspect_pope if benchmark == "pope" else inspect_amber
    priority = {"已推理有结果": 5, "已推理待计算": 4, "已推理待合并计算": 3,
                "推理未完成": 2, "指标孤立需核验": 1, "未推理": 0}
    candidates = []
    for directory in directories:
        try:
            result = reader(directory, expected)
            candidates.append(result)
            if result.state not in ("已推理有结果", "已推理待计算"):
                shard = inspect_shards(directory, benchmark, expected)
                if shard:
                    candidates.append(shard)
        except (OSError, ValueError, TypeError, KeyError) as error:
            candidates.append(Inspection("读取失败", "--", ("--",) if benchmark == "pope" else ("--", "--"),
                                         directory, str(error)))
    return max(candidates, key=lambda item: priority.get(item.state, -1), default=Inspection(
        "未推理", "0/0/0" if benchmark == "pope" else "0/0",
        ("--",) if benchmark == "pope" else ("--", "--")))


def vision_rows(path):
    try:
        rows = read_json(path)
    except json.JSONDecodeError:
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    if isinstance(rows, dict):
        return [rows]
    return rows if isinstance(rows, list) else []


def vision_files(directory, benchmark, stage):
    return sorted(set((directory / benchmark / stage).glob("*_answer.jsonl")) |
                  set((directory / "vision_opd" / stage / benchmark).glob("*_answer.jsonl")))


def inspect_vision(directory, benchmark, expected):
    answers = vision_files(directory, benchmark, "model_answer")
    judges = vision_files(directory, benchmark, "judge")
    if len(answers) > 1 or len(judges) > 1:
        return Inspection("结果冲突需核验", "--/--", ("--",), directory,
                          "同一目录下有多个原始回答或 judge 文件")
    raw_rows = vision_rows(answers[0]) if answers else []
    judge_rows = vision_rows(judges[0]) if judges else []
    raw_ids = [next((f"{key}:{row[key]}" for key in ("sample_uid", "uid", "index", "question_id", "id")
                    if row.get(key) is not None), None) for row in raw_rows if isinstance(row, dict)]
    raw_good = len(raw_rows) == expected and all(
        isinstance(row, dict) and
        str(row.get("model_answer") or "").strip() and
        not str(row["model_answer"]).startswith(BAD_PREFIXES)
        for row in raw_rows
    ) and (not all(raw_ids) or len(set(raw_ids)) == expected)
    judge_good = len(judge_rows) == expected and all(
        isinstance(row, dict) and str(row.get("judge", "")).strip().lower() in ("yes", "no")
        for row in judge_rows
    )
    count = f"{len(raw_rows)}/{len(judge_rows)}"
    if raw_good and judge_good:
        accuracy = sum(str(row["judge"]).strip().lower() == "yes" for row in judge_rows) / expected
        return Inspection("已推理有结果", count, (f"{accuracy * 100:.2f}",), judges[0])
    if raw_good:
        return Inspection("已推理待计算", count, ("--",), answers[0], "原始回答齐全，judge 缺失或无效")
    if raw_rows or judge_rows:
        return Inspection("推理未完成" if raw_rows else "指标孤立需核验", count, ("--",),
                          answers[0] if answers else judges[0], "原始回答或 judge 未齐全")
    return Inspection("未推理", count, ("--",), directory)


def internvl_base_dirs(eval_root):
    canonical = eval_root / "internvl3/full/InternVL3-2B-hf" / DATASET
    legacy = eval_root / "internvl/full/InternVL3-2B-hf/base_chair_original_1000_8gpu"
    return (output_dirs(canonical) or [canonical]) + (output_dirs(legacy) or [legacy])


def inspect_vision_all(directories, benchmark, expected):
    priority = {"已推理有结果": 6, "已推理待计算": 5, "已推理待合并计算": 4,
                "推理未完成": 3,
                "指标孤立需核验": 2, "结果冲突需核验": 1, "未推理": 0}
    candidates = []
    for directory in directories:
        try:
            result = inspect_vision(directory, benchmark, expected)
            candidates.append(result)
            if result.state not in ("已推理有结果", "已推理待计算"):
                for run, outputs in shard_outputs(directory):
                    rows = []
                    for output in outputs:
                        for path in vision_files(output, benchmark, "model_answer"):
                            rows.extend(vision_rows(path))
                    if rows:
                        state = "已推理待合并计算" if len(rows) == expected else "推理未完成"
                        candidates.append(Inspection(state, f"{len(rows)}/0", ("--",), run,
                                                     "仅发现 shard 原始回答"))
        except (OSError, ValueError, TypeError, KeyError) as error:
            candidates.append(Inspection("读取失败", "--/--", ("--",), directory, str(error)))
    return max(candidates, key=lambda item: priority.get(item.state, -1), default=Inspection(
        "未推理", "0/0", ("--",)))


def output_dirs(dataset):
    if not dataset.is_dir():
        return []
    preferred = ("three_tables_fill_official_v2", "official_v2", "chair_official_v2")
    children = [path for path in dataset.iterdir() if path.is_dir() and not path.name.startswith(".")]
    children.sort(key=lambda path: (preferred.index(path.name) if path.name in preferred else len(preferred), path.name))
    return children + [dataset]


def target_dirs(target, eval_root):
    model = f"Qwen3VL-{target.size}-Instruct"
    if target.method in EXTERNAL_SUFFIX:
        dataset = (eval_root / "external_baselines/full" / model / DATASET)
        canonical = dataset / EXTERNAL_SUFFIX[target.method]
        paths = [canonical]
        legacy = eval_root / EXTERNAL_SUFFIX[target.method]
        config = legacy / f"{target.method.lower()}_config.json"
        if config.is_file():
            try:
                data = read_json(config)
                identity = data.get("model_name") or Path(data.get("model_path", "")).name
                if identity == model:
                    paths.append(legacy)
            except (OSError, ValueError, TypeError, KeyError):
                pass
        return paths
    specs = {(spec.size, spec.method): spec for spec in build_specs() if spec.family == "Qwen3VL"}
    spec = specs[(target.size, target.method)]
    experiment = spec.experiment if target.step is None else f"{spec.experiment}_global_step_{target.step}"
    dataset = spec.eval_root / experiment / DATASET
    paths = output_dirs(dataset) or [dataset]
    if target.method == "Base":
        legacy = eval_root / model / DATASET
        paths += output_dirs(legacy) or [legacy]
    return paths


def checkpoint(target, oss_root):
    if target.step is None:
        return "Base 模型"
    specs = {(spec.size, spec.method): spec for spec in build_specs() if spec.family == "Qwen3VL"}
    spec = specs[(target.size, target.method)]
    path = oss_root / spec.oss_name / f"global_step_{target.step}"
    if not oss_root.is_dir():
        return "OSS 未挂载"
    if all((path / name).is_file() and (path / name).stat().st_size > 0
           for name in ("config.json", "model.safetensors")):
        return "完整"
    return f"未找到完整 checkpoint: {path}"


def collect_teacher_scale_rows(eval_root, oss_root):
    rows = []
    issues = []
    for student, teacher, view, spec in teacher_scale.table_specs():
        if spec.method == "Base":
            continue
        oss_parent = oss_root / spec.oss_name
        eval_parent = eval_root / "instruct/full"
        checkpoint_steps = {int(path.name.split("_")[-1])
                            for path in teacher_scale.checkpoint_dirs(spec, oss_root)}
        eval_steps = set()
        if eval_parent.is_dir():
            for path in eval_parent.glob(f"{spec.experiment}_global_step_*"):
                suffix = path.name.removeprefix(f"{spec.experiment}_global_step_")
                if path.is_dir() and suffix.isdigit():
                    eval_steps.add(int(suffix))
        for step in sorted(checkpoint_steps | eval_steps) or [None]:
            ckpt = oss_parent / f"global_step_{step}" if step is not None else oss_parent
            if not oss_root.is_dir():
                ckpt_state = "OSS未挂载，待核实"
            elif step is None:
                ckpt_state = "无完整step，待训练/核实"
            elif all((ckpt / name).is_file() and (ckpt / name).stat().st_size > 0
                     for name in ("config.json", "model.safetensors")):
                ckpt_state = "完整"
            else:
                ckpt_state = "checkpoint不完整/缺失"
            dataset = eval_parent / f"{spec.experiment}_global_step_{step}" / DATASET
            directories = output_dirs(dataset) if step is not None else []
            chair_state, chair_values, chair_evidence = teacher_chair.inspect_chair(
                directories, dataset, issues, f"{student}/{teacher}/{view}/s{step}")
            pope = inspect_all(directories, "pope", EXPECTED_POPE)
            amber = inspect_all(directories, "amber", EXPECTED_AMBER)
            mmstar = inspect_vision_all(directories, "mmstar", VISION_EXPECTED["mmstar"])
            cvbench = inspect_vision_all(directories, "cv-bench", VISION_EXPECTED["cv-bench"])
            rows.append((student, teacher, view, step, ckpt_state, ckpt,
                         chair_state, chair_values, chair_evidence, pope, amber, mmstar, cvbench))
    return rows, issues


def render_teacher_scale(rows, issues):
    lines = ["", "## Fixed-teacher 表缺口（两个 Base 除外）", "",
             "只匹配表中的 7 组 uniform Frozen RKL：压缩视角 tr=0.50、原图视角 tr=1.0；逐 step 核查。",
             "Resp Len 是最终回答的字符数/图，不是 token 数；Correct/Hall 为含重复对象的 mention 总数。",
             "`--` 表示未取得合规结果，不能用旧 tr=0.75 数字填补。", "",
             "| Student | Teacher | View | Step | CKPT | CHAIR 状态 | CHAIRi | CHAIRs | ObjF1 | Resp Len (chars/img) | Correct | Hall |",
             "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def value(values, key, percent=False):
        found = values.get(key)
        return "--" if found is None else f"{found * 100:.2f}" if percent else f"{found:.2f}" if isinstance(found, float) else str(found)

    for student, teacher, view, step, ckpt_state, _, chair_state, values, _, *_ in rows:
        cells = (student, teacher, view, str(step) if step is not None else "--", ckpt_state,
                 chair_state, value(values, "CHAIRi", True), value(values, "CHAIRs", True),
                 value(values, "ObjF1", True), value(values, "chars_per_image"),
                 value(values, "correct_mentions"), value(values, "hall_mentions"))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "| Student | Teacher | View | Step | POPE 状态 | F1 (%) | AMBER 状态 | A-HAL | MMStar 状态 | Acc. (%) | CV-Bench 状态 | Acc. (%) |",
              "| --- | --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | --- | ---: |"]
    for student, teacher, view, step, _, _, _, _, _, pope, amber, mmstar, cvbench in rows:
        cells = (student, teacher, view, str(step) if step is not None else "--",
                 pope.state, pope.values[0], amber.state, amber.values[1],
                 mmstar.state, mmstar.values[0], cvbench.state, cvbench.values[0])
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "### Fixed-teacher 证据路径", ""]
    for student, teacher, view, step, _, ckpt, _, _, chair_evidence, pope, amber, mmstar, cvbench in rows:
        label = f"{student}/{teacher}/{view}/s{step if step is not None else '--'}"
        lines.append(f"- {label}: ckpt=`{ckpt}`; CHAIR=`{chair_evidence}`; POPE=`{pope.source or '--'}`; "
                     f"AMBER=`{amber.source or '--'}`; MMStar=`{mmstar.source or '--'}`; "
                     f"CV-Bench=`{cvbench.source or '--'}`")
    if issues:
        lines += ["", "### 读取警告", ""]
        lines += [f"- {issue}" for issue in sorted(set(issues))]
    return "\n".join(lines) + "\n"


def render(rows, vision_results=(), teacher_rows=(), teacher_issues=()):
    lines = ["# tr=0.50 主表指定行：POPE / AMBER 状态核查", "",
             "仅检查原图 student 评测；Base 和 external 使用原始 Base 模型。",
             "POPE 原始结果需三个 split 各 5223 条；AMBER(a) 需 15220 条原始回答与官方响应。",
             "Raw 列是去重后条数；错误回答或重复行不会被判为完整。",
             "状态区分未推理、推理未完成、已推理待合并计算、已推理待计算、已推理有结果；指标孤立不算完成。",
             "M 编号仅作对照，实际按模型、方法、step 的规范实验名定位。", "",
             "| ID | 模型 | 方法 | Step | CKPT | POPE | Raw adv/pop/rand | F1 (%) | AMBER | Raw/official | A-CHAIR | A-HAL |",
             "| --- | --- | --- | ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: |"]
    for target, ckpt, pope, amber in rows:
        step = "--" if target.step is None else str(target.step)
        lines.append("| " + " | ".join((target.row_id, f"Qwen3VL-{target.size}", target.method,
                                       step, ckpt, pope.state, pope.count, pope.values[0],
                                       amber.state, amber.count, *amber.values)) + " |")
    lines += ["", "## 证据路径", ""]
    for target, _, pope, amber in rows:
        lines.append(f"- {target.row_id} POPE: `{pope.source or '--'}`; {pope.detail or pope.state}; "
                     f"AMBER: `{amber.source or '--'}`; {amber.detail or amber.state}")
    lines += ["", "## InternVL3-2B Base：MMStar / CV-Bench", "",
              "Base 使用原图；Raw/Judge 为各自完整 benchmark 的回答与判分条数。", "",
              "| Benchmark | 状态 | Raw/Judge | Acc. (%) | 证据路径 |",
              "| --- | --- | ---: | ---: | --- |"]
    for benchmark, result in vision_results:
        label = "MMStar" if benchmark == "mmstar" else "CV-Bench"
        lines.append(f"| {label} | {result.state} | {result.count} | {result.values[0]} | "
                     f"`{result.source or '--'}` |")
    return "\n".join(lines) + "\n" + render_teacher_scale(teacher_rows, teacher_issues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=ROOT / "res-opd/eval_results")
    parser.add_argument("--oss-root", type=Path, default=Path("/data/oss_bucket_0/yanlin/ckpt/OPD/v4"))
    parser.add_argument("--output-md", type=Path,
                        default=ROOT / "res-opd/plan/main_tr05_pope_amber_status.md")
    args = parser.parse_args()
    rows = []
    counts = Counter()
    for target in TARGETS:
        directories = target_dirs(target, args.eval_root)
        pope = inspect_all(directories, "pope", EXPECTED_POPE)
        amber = inspect_all(directories, "amber", EXPECTED_AMBER)
        ckpt = checkpoint(target, args.oss_root)
        rows.append((target, ckpt, pope, amber))
        counts[("POPE", pope.state)] += 1
        counts[("AMBER", amber.state)] += 1
        print(f"{target.row_id:9} {target.size:2} {target.method:12} step={target.step or '--':>3} "
              f"POPE={pope.state}({pope.count}) AMBER={amber.state}({amber.count})")
    vision_results = []
    for benchmark, expected in VISION_EXPECTED.items():
        result = inspect_vision_all(internvl_base_dirs(args.eval_root), benchmark, expected)
        vision_results.append((benchmark, result))
        print(f"InternVL3-2B Base {benchmark}: {result.state} ({result.count}); Acc={result.values[0]}")
    teacher_rows, teacher_issues = collect_teacher_scale_rows(args.eval_root, args.oss_root)
    print(f"Fixed-teacher: {len(teacher_rows)} step rows across 7 non-Base configurations; "
          f"warnings={len(teacher_issues)}")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render(rows, vision_results, teacher_rows, teacher_issues), encoding="utf-8")
    print(f"WROTE {args.output_md}")
    for (benchmark, state), count in sorted(counts.items()):
        print(f"{benchmark}: {state}={count}")


if __name__ == "__main__":
    main()
