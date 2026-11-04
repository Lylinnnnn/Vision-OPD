#!/usr/bin/env python3

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import summarize_paper_chair_accounting as accounting


SCALES = ("2b", "4b", "8b")
EXTRA_RUN = re.compile(
    r"^Res-OPD-Qwen3VL-(?P<scale>2B|4B|8B)-Instruct-orig-sr1\.0-tr0\.5-"
    r"(?P<setting>.+)-full5k-e1_global_step_(?P<step>\d+)$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="List every complete official tr=0.5 CHAIR checkpoint evaluation."
    )
    parser.add_argument(
        "--manifest",
        default="res-opd/eval/manifests/paper_chair_tr05_method_comparison_sources.txt",
    )
    parser.add_argument("--eval-root", default="res-opd/eval_results/instruct/full")
    parser.add_argument("--output-suffix", default="chair_official_v2")
    parser.add_argument("--fallback-output-suffix", action="append", default=[])
    parser.add_argument(
        "--model-root", default="/home/liuyanlin.lyl/notebook/model/qwen"
    )
    parser.add_argument(
        "--output-md", default="res-opd/plan/chair_official_v2_tr05_all_steps.md"
    )
    parser.add_argument("--expected-prompt", default=accounting.OFFICIAL_PROMPT)
    parser.add_argument("--expected-samples", type=int, default=1000)
    parser.add_argument(
        "--length-mode", choices=("tokenizer", "record", "words"), default="tokenizer"
    )
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def method_from_name(name):
    if "-sft-lowres-" in name:
        return "Low-res SFT"
    if "-riskmask-entropy-" in name:
        return "RiskMask (entropy)"
    for marker, label in (
        ("-riskmask-", "RiskMask"),
        ("-randommask-", "RandomMask"),
        ("-entropymask-", "EntropyMask"),
        ("-fixed-rkl-", "Fixed RKL"),
        ("-frozen-rkl-", "Frozen RKL"),
    ):
        if marker in name:
            return label
    return "Other"


def step_entry(source, step):
    entry = dict(source)
    entry["step"] = f"global_step_{step}"
    entry["label"] = re.sub(r"/s\d+$", f"/s{step}", source["label"])
    return entry


def discover_entries(manifest, eval_root, include_extras=True):
    sources = accounting.load_manifest(manifest)
    entries = []
    issues = []
    known = {}
    for source in sources:
        if source["source_type"] == "local_base":
            entries.append((source, "manifest"))
            continue
        if source["source_type"] != "oss_checkpoint":
            raise ValueError(f"Unsupported source type: {source['source_type']}")
        if source["student_ratio"] != "1.0" or "-sr1.0-tr0.5-" not in source["primary"]:
            raise ValueError(f"Manifest includes a non-tr0.5 original-view run: {source['label']}")
        previous = known.setdefault(source["primary"], source)
        if previous["oss_name"] != source["oss_name"]:
            raise ValueError(f"Conflicting OSS names for {source['primary']}")

    for name, source in known.items():
        pattern = re.compile(rf"^{re.escape(name)}_global_step_(\d+)$")
        found = 0
        for path in eval_root.glob(f"{name}_global_step_*"):
            match = pattern.fullmatch(path.name)
            if path.is_dir() and match:
                entries.append((step_entry(source, int(match.group(1))), "manifest"))
                found += 1
        if not found:
            issues.append((name, "no matching checkpoint evaluation directories"))

    if include_extras:
        for path in eval_root.iterdir():
            if not path.is_dir():
                continue
            match = EXTRA_RUN.fullmatch(path.name)
            if not match:
                continue
            step = int(match.group("step"))
            name = path.name.rsplit("_global_step_", 1)[0]
            if name in known:
                continue
            scale = match.group("scale").lower()
            method = method_from_name(name)
            entries.append(
                (
                    {
                        "index": 0,
                        "label": f"{scale}/{method}/tr05/s{step}",
                        "source_type": "oss_checkpoint",
                        "primary": name,
                        "oss_name": "",
                        "step": f"global_step_{step}",
                        "student_ratio": "1.0",
                    },
                    "discovered, not in manifest",
                )
            )
    return entries, issues


def summarize_entries(entries, eval_root, args):
    suffixes = list(dict.fromkeys([args.output_suffix, *args.fallback_output_suffix]))
    summarize_args = SimpleNamespace(
        expected_prompt=args.expected_prompt,
        expected_samples=args.expected_samples,
        length_mode=args.length_mode,
        model_root=args.model_root,
        best_available_step=False,
        fallback_best_available_step=False,
    )
    rows = []
    issues = []
    tokenizer_cache = {}
    for entry, source_kind in entries:
        try:
            row = accounting.summarize_entry(
                entry, eval_root, suffixes, summarize_args, tokenizer_cache
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            issues.append((f"{entry['primary']}:{entry['step'] or 'base'}", str(error)))
            continue
        row["experiment"] = (
            Path(entry["primary"]).name
            if entry["source_type"] == "local_base"
            else entry["primary"]
        )
        row["source_kind"] = source_kind
        row["oss_name"] = entry["oss_name"]
        row["method"] = (
            "Base" if entry["source_type"] == "local_base"
            else method_from_name(entry["primary"])
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            SCALES.index(row["scale"]),
            row["method"] != "Base",
            row["method"],
            row["experiment"],
            int(row["checkpoint_step"]) if row["checkpoint_step"].isdigit() else -1,
        )
    )
    return rows, issues


def markdown_report(rows, issues, args, manifest):
    lines = [
        "# tr=0.50: All Available Official CHAIR Checkpoint Results",
        "",
        "Only original-view inference (`sr=1.0`) from Qwen3VL-Instruct full5k-e1 runs is included. "
        "Each row is one complete 1,000-caption evaluation using `Describe the image.` "
        "and the official caption/mention CHAIR schema. No checkpoint is selected by score.",
        "",
        f"Manifest: `{manifest}`. Extra runs with exact `tr0.5` names are marked "
        "as discovered; their OSS provenance is not inferred. Old-prompt and non-official "
        "metrics are excluded. Missing evaluations do not imply missing checkpoints.",
        "",
        f"Validated rows: **{len(rows)}**; excluded or missing: **{len(issues)}**. "
        "Correct/hallucinated mention totals include repeats; ObjPrec/ObjRecall/ObjF1 "
        "use unique object categories per caption.",
        "",
    ]
    sources = []
    for scale in SCALES:
        scale_rows = [row for row in rows if row["scale"] == scale]
        if not scale_rows:
            continue
        lines.extend([f"## {scale.upper()}", ""])
        grouped = defaultdict(list)
        for row in scale_rows:
            grouped[(row["method"], row["experiment"])].append(row)
        for (method, experiment), group in grouped.items():
            lines.extend(
                [
                    f"### {method}: `{experiment}`",
                    "",
                    "| Step | CHAIRi↓ | CHAIRs↓ | ObjPrec↑ | ObjRecall↑ | ObjF1↑ | "
                    "Tok/img | Correct mentions | Hall mentions | Hall/100 tok | Source |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for row in group:
                source_id = f"S{len(sources) + 1:03d}"
                sources.append((source_id, row))
                lines.append(
                    f"| {row['checkpoint_step']} | {row['CHAIRi']:.4f} | "
                    f"{row['CHAIRs']:.4f} | {row['ObjPrec']:.4f} | "
                    f"{row['ObjRecall']:.4f} | {row['ObjF1']:.4f} | "
                    f"{row['response_tokens_mean']:.2f} | "
                    f"{row['official_correct_mentions']} | "
                    f"{row['official_hallucinated_mentions']} | "
                    f"{row['official_hallucinated_mentions_per_100_tokens']:.3f} | "
                    f"{source_id} |"
                )
            lines.append("")
    lines.extend(["## Source Paths", ""])
    for source_id, row in sources:
        lines.append(
            f"- {source_id}: `{row['output_dir']}` "
            f"(source: {row['source_kind']}; OSS experiment: "
            f"`{row['oss_name'] or 'not verified'}`)"
        )
    if issues:
        lines.extend(["", "## Excluded or Missing", ""])
        for name, reason in issues:
            lines.append(f"- `{name}`: {reason.replace(chr(10), ' | ')}")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    root = accounting.repo_root()
    manifest = accounting.resolve_path(args.manifest, root)
    eval_root = accounting.resolve_path(args.eval_root, root)
    output = accounting.resolve_path(args.output_md, root)
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Evaluation root does not exist: {eval_root}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {output}")
    entries, discovery_issues = discover_entries(
        manifest, eval_root, include_extras=not args.manifest_only
    )
    rows, validation_issues = summarize_entries(entries, eval_root, args)
    issues = discovery_issues + validation_issues
    if not rows:
        raise ValueError("No complete official CHAIR outputs matched the requested scope")
    if args.strict and issues:
        raise ValueError(f"Strict mode: {len(issues)} missing or invalid results")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown_report(rows, issues, args, manifest), encoding="utf-8")
    counts = Counter(row["scale"] for row in rows)
    print(f"Saved {output}")
    print(f"Validated rows: {len(rows)}; by scale: {dict(counts)}; excluded/missing: {len(issues)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
