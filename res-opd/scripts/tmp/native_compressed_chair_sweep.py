#!/usr/bin/env python3

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import summarize_paper_chair_accounting as accounting


SOURCE_LABELS = {
    ("2b", "0.25"): "2b/riskmask/tr025/p30/s50",
    ("2b", "0.5"): "2b/riskmask/tr05/p30/s50",
    ("2b", "0.75"): "2b/riskmask/tr075/p30/main_s50",
    ("8b", "0.25"): "8b/riskmask/tr025/p30/s200",
    ("8b", "0.5"): "8b/riskmask/tr05/p30/s50",
    ("8b", "0.75"): "8b/riskmask/tr075/p30/main_s100",
}
RATIOS = ("0.25", "0.5", "0.75")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover all native-compressed CHAIR checkpoints and summarize official results."
    )
    parser.add_argument("action", choices=("prepare", "summarize"))
    parser.add_argument(
        "--source-manifest",
        default="res-opd/eval/manifests/paper_chair_describe45_sources.txt",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="/data/oss_bucket_0/yanlin/ckpt/OPD/v4",
    )
    parser.add_argument(
        "--model-root", default="/home/liuyanlin.lyl/notebook/model/qwen"
    )
    parser.add_argument(
        "--manifest-out",
        default="res-opd/paper/tmp/native_compressed_chair_all_steps_sources.txt",
    )
    parser.add_argument(
        "--inventory-md",
        default="res-opd/paper/tmp/native_compressed_chair_checkpoint_inventory.md",
    )
    parser.add_argument(
        "--eval-root", default="res-opd/eval_results/instruct/full"
    )
    parser.add_argument("--output-suffix", default="chair_native_sweep_official_v2")
    parser.add_argument("--label-regex", default="")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument(
        "--fallback-output-suffix",
        action="append",
        default=["chair_official_v2", "chair_describe_image_official"],
    )
    parser.add_argument(
        "--output-md", default="res-opd/paper/tmp/native_compressed_chair_all_steps.md"
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def lookup_sources(source_manifest):
    sources = accounting.load_manifest(source_manifest)
    by_label = {entry["label"]: entry for entry in sources}
    if len(by_label) != len(sources):
        raise ValueError("Source manifest has duplicate labels")
    selected = {}
    for key, label in SOURCE_LABELS.items():
        entry = by_label[label]
        scale, ratio = key
        if (
            entry["source_type"] != "oss_checkpoint"
            or entry["student_ratio"] != "1.0"
            or f"Qwen3VL-{scale.upper()}-Instruct" not in entry["primary"]
            or f"-tr{ratio}-" not in entry["primary"]
            or "-riskmask-nll-p30-" not in entry["primary"]
            or not entry["oss_name"]
        ):
            raise ValueError(f"Source does not match expected experiment: {label}")
        selected[key] = entry
    return selected


def discover_steps(checkpoint_root, source):
    experiment_root = checkpoint_root / source["oss_name"]
    complete, incomplete = [], []
    for path in experiment_root.glob("global_step_*"):
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        if not path.is_dir() or not match:
            continue
        missing = []
        for filename in ("model.safetensors", "config.json"):
            file = path / filename
            try:
                if not file.is_file() or file.stat().st_size == 0:
                    missing.append(filename)
            except OSError:
                missing.append(filename)
        if missing:
            incomplete.append((int(match.group(1)), ", ".join(missing)))
        else:
            complete.append(int(match.group(1)))
    return sorted(set(complete)), sorted(incomplete)


def make_entries(sources, checkpoint_root, model_root):
    entries, inventory = [], []
    for scale in ("2b", "8b"):
        model_path = model_root / f"Qwen3VL-{scale.upper()}-Instruct"
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"Base model is missing: {model_path}/config.json")
        entries.append(
            (f"{scale}/base/original", "local_base", str(model_path), "", "", "1.0")
        )
        for ratio in RATIOS:
            tag = ratio.replace(".", "") if ratio != "0.5" else "05"
            entries.append(
                (
                    f"{scale}/native_base/sr{tag}",
                    "local_base", str(model_path), "", "", ratio,
                )
            )
            source = sources[(scale, ratio)]
            complete, incomplete = discover_steps(checkpoint_root, source)
            inventory.append((scale, ratio, source, complete, incomplete))
            if not complete:
                raise FileNotFoundError(
                    f"No complete checkpoints for {scale} tr={ratio}: "
                    f"{checkpoint_root / source['oss_name']}"
                )
            for step in complete:
                entries.append(
                    (
                        f"{scale}/riskmask/tr{tag}/p30/s{step}",
                        "oss_checkpoint", source["primary"], source["oss_name"],
                        f"global_step_{step}", "1.0",
                    )
                )
    return entries, inventory


def prepare(args, root):
    sources = lookup_sources(accounting.resolve_path(args.source_manifest, root))
    checkpoint_root = Path(args.checkpoint_root)
    model_root = Path(args.model_root)
    entries, inventory = make_entries(sources, checkpoint_root, model_root)
    manifest = accounting.resolve_path(args.manifest_out, root)
    inventory_md = accounting.resolve_path(args.inventory_md, root)
    lines = ["# index|label|source_type|model_path_or_experiment_name|oss_name|step|student_ratio"]
    lines.extend(
        "|".join((str(index), *entry))
        for index, entry in enumerate(entries, 1)
    )
    report = [
        "# Native Compressed-View CHAIR Checkpoint Inventory", "",
        "Only exact OSS experiment directories from the paper source manifest were scanned. "
        "Native Base and original Base use the same untrained checkpoint with different "
        "inference image ratios. Incomplete checkpoint directories are not evaluated.", "",
        "| Scale | Teacher ratio | Paper's old step | All complete steps | Incomplete steps | Exact OSS experiment |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for scale, ratio, source, complete, incomplete in inventory:
        old_step = source["step"].removeprefix("global_step_")
        bad = ", ".join(f"{step} ({reason})" for step, reason in incomplete) or "--"
        report.append(
            f"| {scale.upper()} | {ratio} | {old_step} | "
            f"{', '.join(map(str, complete))} | {bad} | "
            f"`{checkpoint_root / source['oss_name']}` |"
        )
    report.extend(["", f"Manifest: `{manifest}`", ""])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    inventory_md.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    inventory_md.write_text("\n".join(report), encoding="utf-8")
    print(f"Discovered {len(entries)} entries, including {len(entries) - 8} checkpoints")
    print(f"Manifest: {manifest}\nInventory: {inventory_md}")


def describe_entry(entry):
    label = entry["label"]
    if "/base/original" in label:
        return "Original Base", "1.0"
    if "/native_base/" in label:
        return "Native Base", entry["student_ratio"]
    match = re.search(r"/tr(025|05|075)/", label)
    if not match:
        raise ValueError(f"Unexpected riskmask label: {label}")
    return "Res-OPSD (RiskMask)", {"025": "0.25", "05": "0.5", "075": "0.75"}[match.group(1)]


def summary_lines(rows, missing, manifest, suffixes, label_regex):
    lines = [
        "# Native Compressed-View CHAIR: All Available Steps", "",
        "All rows use the 1,000-image COCO CHAIR set, `Describe the image.`, and "
        "the official caption/mention schema. No checkpoint was selected by its score. "
        "CHAIRi counts all object mentions; ObjPrec/ObjRecall/ObjF1 use unique "
        "canonical object categories per caption. Results are not comparable to the "
        "paper's older CHAIR numbers until the paper is updated.", "",
        f"Manifest: `{manifest}`; result suffixes, in priority order: "
        f"{', '.join(f'`{suffix}`' for suffix in suffixes)}. "
        f"Validated: **{len(rows)}**; missing/invalid: **{len(missing)}**.", "",
    ]
    if label_regex:
        lines.extend([f"Label filter: `{label_regex}`.", ""])
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scale"]].append(row)
    for scale in ("2b", "8b"):
        lines.extend([f"## {scale.upper()}", ""])
        lines.extend([
            "| Method | Ratio | Step | CHAIRi↓ | CHAIRs↓ | ObjPrec↑ | ObjRecall↑ | ObjF1↑ | RepRate↓ | Hall captions | Source |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in grouped[scale]:
            source_id = f"S{row['index']:03d}"
            lines.append(
                f"| {row['display_method']} | {row['ratio']} | {row['checkpoint_step']} | "
                f"{row['CHAIRi']:.4f} | {row['CHAIRs']:.4f} | "
                f"{row['ObjPrec']:.4f} | {row['ObjRecall']:.4f} | "
                f"{row['ObjF1']:.4f} | {row['RepRate']:.4f} | "
                f"{row['hallucinated_captions']} | {source_id} |"
            )
        lines.extend([
            "", "### Response and mention accounting", "",
            "| Method | Ratio | Step | Tok/img | All mentions | Correct mentions | Hall mentions | Correct/img | Hall/img | Hall/100 tok | Unique GT | Unique pred | Unique correct | Unique hall |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in grouped[scale]:
            lines.append(
                f"| {row['display_method']} | {row['ratio']} | {row['checkpoint_step']} | "
                f"{row['response_tokens_mean']:.2f} | {row['official_pred_mentions']} | "
                f"{row['official_correct_mentions']} | {row['official_hallucinated_mentions']} | "
                f"{row['official_correct_mentions_per_caption']:.3f} | "
                f"{row['official_hallucinated_mentions_per_caption']:.3f} | "
                f"{row['official_hallucinated_mentions_per_100_tokens']:.3f} | "
                f"{row['unique_gt_categories']} | {row['unique_pred_categories']} | "
                f"{row['unique_correct_categories']} | {row['unique_hallucinated_categories']} |"
            )
        lines.append("")
    lines.extend(["## Source Paths", ""])
    for row in rows:
        source_id = f"S{row['index']:03d}"
        lines.append(
            f"- {source_id}: `{row['output_dir']}`; "
            f"checkpoint: `{row['checkpoint_path']}`"
        )
    if missing:
        lines.extend(["", "## Missing or Invalid", ""])
        for entry, reason in missing:
            lines.append(f"- `{entry['label']}`: {reason.replace(chr(10), ' | ')}")
    lines.append("")
    return "\n".join(lines)


def summarize(args, root):
    manifest = accounting.resolve_path(args.manifest_out, root)
    eval_root = accounting.resolve_path(args.eval_root, root)
    entries = accounting.load_manifest(manifest)
    if args.label_regex:
        label_pattern = re.compile(args.label_regex)
        entries = [entry for entry in entries if label_pattern.search(entry["label"])]
    if not entries:
        raise ValueError("No manifest entries match the label filter")
    suffixes = list(dict.fromkeys(
        [args.output_suffix, *([] if args.no_fallback else args.fallback_output_suffix)]
    ))
    accounting_args = SimpleNamespace(
        expected_prompt=accounting.OFFICIAL_PROMPT,
        expected_samples=1000,
        length_mode="tokenizer",
        model_root=args.model_root,
        best_available_step=False,
        fallback_best_available_step=False,
    )
    rows, missing, tokenizer_cache = [], [], {}
    for entry in entries:
        try:
            row = accounting.summarize_entry(
                entry, eval_root, suffixes, accounting_args, tokenizer_cache
            )
            output_dir = Path(row["output_dir"])
            metrics = json.loads((output_dir / "chair_metrics.json").read_text(encoding="utf-8"))
            if metrics.get("ground_truth_protocol") != "instance_categories_union_reference_caption_objects":
                raise ValueError("Ground-truth protocol is not official")
            records = accounting.load_jsonl(output_dir / "eval_results.jsonl")
            expected_ratio = float(entry["student_ratio"])
            if any(
                record.get("eval_degradation_mode") != "original"
                or record.get("eval_prompt") != accounting.OFFICIAL_PROMPT
                or float(record.get("eval_student_ratio", -1)) != expected_ratio
                for record in records
            ):
                raise ValueError("Generation prompt, degradation, or student ratio does not match")
            row["RepRate"] = accounting.finite_float(metrics["RepRate"], "RepRate")
            row["display_method"], row["ratio"] = describe_entry(entry)
            row["checkpoint_path"] = (
                entry["primary"] if entry["source_type"] == "local_base" else
                str(Path(args.checkpoint_root) / entry["oss_name"] / entry["step"])
            )
            rows.append(row)
        except (OSError, ValueError, TypeError, KeyError) as error:
            missing.append((entry, str(error)))
    if args.strict and missing:
        raise RuntimeError(f"{len(missing)} entries are missing or invalid; refusing partial summary")
    output = accounting.resolve_path(args.output_md, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        summary_lines(rows, missing, manifest, suffixes, args.label_regex), encoding="utf-8"
    )
    print(f"Saved {output}; validated={len(rows)}, missing_or_invalid={len(missing)}")


def main():
    args = parse_args()
    root = accounting.repo_root()
    if args.action == "prepare":
        prepare(args, root)
    else:
        summarize(args, root)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
