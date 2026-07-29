#!/usr/bin/env python3
"""Resolve and verify the exact model sources for the 45 paper CHAIR runs."""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


BASE_MODEL_RE = re.compile(r"^Qwen3VL-(2B|4B|8B)-Instruct$")
STEP_RE = re.compile(r"^(?P<experiment>.+)_global_step_(?P<step>\d+)$")
RATIO_RE = re.compile(r"_original_sr(?P<ratio>[0-9]+(?:p[0-9]+)?)$")


def experiment_to_oss_name(experiment_name: str) -> str:
    """Mirror get_oss_name_from_exp_name in eval_batch_from_oss.sh."""
    if experiment_name.startswith("ResOPD_"):
        return experiment_name
    if experiment_name.startswith("Res-OPD-Qwen3VL-2B-Instruct-"):
        suffix = experiment_name.removeprefix("Res-OPD-Qwen3VL-2B-Instruct-")
    elif experiment_name.startswith("Res-OPD-"):
        suffix = experiment_name.removeprefix("Res-OPD-")
    else:
        suffix = experiment_name

    epoch_match = re.match(r"^(.+)-(e[0-9]+)$", suffix)
    epoch_tag = ""
    if epoch_match:
        suffix, epoch = epoch_match.groups()
        epoch_tag = f"-{epoch}"
    return f"ResOPD_{suffix.replace('-', '_')}{epoch_tag}"


def ratio_from_dataset_tag(dataset_tag: str) -> float:
    match = RATIO_RE.search(dataset_tag)
    if not match:
        raise ValueError(f"Cannot parse student ratio from dataset directory: {dataset_tag}")
    return float(match.group("ratio").replace("p", "."))


def selected_entries(summary_path: Path, model_root: Path, oss_base: str):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = data.get("paper_rows")
    if not isinstance(rows, list):
        raise ValueError(f"Missing paper_rows list in {summary_path}")
    if len(rows) != 45:
        raise ValueError(
            f"Expected the existing45 summary to contain exactly 45 rows, found {len(rows)}"
        )

    entries = []
    for index, row in enumerate(rows, start=1):
        label = row.get("label")
        selected = row.get("selected_eval_results")
        if not label or not selected:
            raise ValueError(f"Row {index} is missing label or selected_eval_results")
        result_path = Path(selected)
        parts = result_path.parts
        try:
            full_index = parts.index("full")
            experiment_dir = parts[full_index + 1]
            dataset_dir = parts[full_index + 2]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Cannot parse selected result path for {label}: {selected}") from exc

        ratio = ratio_from_dataset_tag(dataset_dir)
        base_match = BASE_MODEL_RE.fullmatch(experiment_dir)
        if base_match:
            model_path = model_root / experiment_dir
            entries.append(
                {
                    "index": index,
                    "label": label,
                    "source_type": "local_base",
                    "model_path": str(model_path),
                    "student_ratio": ratio,
                    "selected_eval_results": selected,
                }
            )
            continue

        step_match = STEP_RE.fullmatch(experiment_dir)
        if not step_match:
            raise ValueError(
                f"Trained result directory lacks an exact global_step suffix for {label}: "
                f"{experiment_dir}"
            )
        experiment_name = step_match.group("experiment")
        step = f"global_step_{step_match.group('step')}"
        oss_name = experiment_to_oss_name(experiment_name)
        oss_step_uri = f"{oss_base.rstrip('/')}/{oss_name}/{step}"
        entries.append(
            {
                "index": index,
                "label": label,
                "source_type": "oss_checkpoint",
                "experiment_name": experiment_name,
                "oss_name": oss_name,
                "step": step,
                "oss_step_uri": oss_step_uri,
                "student_ratio": ratio,
                "selected_eval_results": selected,
            }
        )

    source_keys = []
    for entry in entries:
        if entry["source_type"] == "local_base":
            source_keys.append((entry["model_path"], entry["student_ratio"]))
        else:
            source_keys.append((entry["oss_step_uri"], entry["student_ratio"]))
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("The 45-row selection contains duplicate model-source/eval-ratio targets")
    return entries


def command_ok(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    detail = (result.stderr or result.stdout).strip().splitlines()
    return result.returncode == 0, detail[-1] if detail else ""


def verify_entries(entries, ossutil: str):
    failures = []
    for entry in entries:
        if entry["source_type"] == "local_base":
            model_path = Path(entry["model_path"])
            checks = {
                "model_dir": model_path.is_dir(),
                "config_json": (model_path / "config.json").is_file(),
            }
        else:
            uri = entry["oss_step_uri"]
            model_ok, model_detail = command_ok([ossutil, "stat", f"{uri}/model.safetensors"])
            config_ok, config_detail = command_ok([ossutil, "stat", f"{uri}/config.json"])
            checks = {
                "model_safetensors": model_ok,
                "config_json": config_ok,
            }
            if not model_ok:
                entry["model_error"] = model_detail
            if not config_ok:
                entry["config_error"] = config_detail
        entry["checks"] = checks
        entry["verified"] = all(checks.values())
        status = "OK" if entry["verified"] else "FAIL"
        source = entry.get("oss_step_uri", entry.get("model_path"))
        print(f"[{entry['index']:02d}/45] {status:4s} {entry['label']} -> {source}")
        if not entry["verified"]:
            failures.append(entry)
    return failures


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-summary",
        default="res-opd/paper/tmp/paper_chair_official_existing45.json",
    )
    parser.add_argument(
        "--model-root",
        default="/home/liuyanlin.lyl/notebook/model/qwen",
    )
    parser.add_argument(
        "--oss-base",
        default="oss://industry-algo/yanlin/ckpt/OPD/v4",
    )
    parser.add_argument("--ossutil", default="ossutil")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary_path = Path(args.selection_summary)
    entries = selected_entries(summary_path, Path(args.model_root), args.oss_base)
    failures = []
    if not args.skip_verify:
        if shutil.which(args.ossutil) is None:
            raise RuntimeError(f"ossutil executable not found: {args.ossutil}")
        failures = verify_entries(entries, args.ossutil)

    payload = {
        "selection_summary": str(summary_path),
        "num_entries": len(entries),
        "num_local_base": sum(e["source_type"] == "local_base" for e in entries),
        "num_oss_checkpoint": sum(e["source_type"] == "oss_checkpoint" for e in entries),
        "num_failures": len(failures),
        "entries": entries,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved preflight inventory to {output_path}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
