#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import summarize_paper_chair_accounting as accounting


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="res-opd/eval/manifests/paper_chair_tr05_method_comparison_sources.txt",
    )
    parser.add_argument("--indices", default="1-14")
    parser.add_argument("--output-suffix", default="chair_official_v2")
    parser.add_argument("--fallback-output-suffix", action="append", default=[])
    parser.add_argument("--eval-root", default="res-opd/eval_results/instruct/full")
    parser.add_argument(
        "--model-root",
        default="/home/liuyanlin.lyl/notebook/model/qwen",
    )
    parser.add_argument(
        "--output-prefix",
        default="res-opd/paper/tmp/chair_official_v2_checkpoint_sensitivity",
    )
    parser.add_argument("--expected-prompt", default=accounting.OFFICIAL_PROMPT)
    parser.add_argument("--expected-samples", type=int, default=1000)
    parser.add_argument(
        "--length-mode",
        choices=("tokenizer", "record", "words"),
        default="tokenizer",
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def selection_rank(row):
    return (
        row["CHAIRi"],
        -row["ObjF1"],
        row["CHAIRs"],
        int(row["checkpoint_step"]) if row["checkpoint_step"].isdigit() else -1,
    )


def endpoint_payload(row):
    fields = (
        "scale",
        "method",
        "label",
        "checkpoint_step",
        "response_tokens_mean",
        "official_correct_mentions",
        "official_hallucinated_mentions",
        "official_correct_mentions_per_caption",
        "official_hallucinated_mentions_per_caption",
        "official_hallucinated_mentions_per_100_tokens",
        "CHAIRi",
        "CHAIRs",
        "ObjPrec",
        "ObjRecall",
        "ObjF1",
        "output_dir",
    )
    return {field: row[field] for field in fields}


def candidate_entries(entry, eval_root, suffixes):
    if entry["source_type"] == "local_base":
        return [entry]
    candidates = accounting.fallback_step_outputs(entry, eval_root, suffixes)
    entries = []
    for candidate in candidates:
        candidate_entry = dict(entry)
        candidate_entry["step"] = f"global_step_{candidate['step']}"
        entries.append(candidate_entry)
    return entries


def markdown_report(methods, stress_tests, skipped):
    lines = [
        "# Official CHAIR Checkpoint Sensitivity",
        "",
        (
            "All endpoints use the official CHAIR caption/mention schema and the "
            "same `Describe the image.` prompt. Best and worst endpoints are selected "
            "by CHAIRi, with ObjF1, CHAIRs, and step used only as tie-breakers."
        ),
        "",
        "## Per-method checkpoint ranges",
        "",
        (
            "| Scale | Method | Steps | Best step | Best CHAIRi | Best CHAIRs | "
            "Best ObjF1 | Worst step | Worst CHAIRi | Worst CHAIRs | Worst ObjF1 |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in methods:
        best = item["best"]
        worst = item["worst"]
        lines.append(
            f"| {item['scale'].upper()} | {item['method']} | {item['candidate_count']} | "
            f"{best['checkpoint_step']} | {best['CHAIRi']:.4f} | {best['CHAIRs']:.4f} | "
            f"{best['ObjF1']:.4f} | {worst['checkpoint_step']} | "
            f"{worst['CHAIRi']:.4f} | {worst['CHAIRs']:.4f} | "
            f"{worst['ObjF1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Asymmetric stress-test diagnostic",
            "",
            (
                "This block intentionally contrasts the best RiskMask checkpoint with "
                "the worst Frozen RKL checkpoint. It is not a fair method comparison "
                "and must not be presented as a standard benchmark result."
            ),
            "",
            (
                "| Scale | Endpoint | Step | Tok/img | Correct mentions | Hall mentions | "
                "Hall/100 tok | CHAIRi | CHAIRs | ObjRecall | ObjF1 |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in stress_tests:
        for endpoint_name in ("riskmask_best", "frozen_rkl_worst"):
            row = item[endpoint_name]
            label = "RiskMask best" if endpoint_name == "riskmask_best" else "Frozen RKL worst"
            lines.append(
                f"| {item['scale'].upper()} | {label} | {row['checkpoint_step']} | "
                f"{row['response_tokens_mean']:.2f} | "
                f"{row['official_correct_mentions']} | "
                f"{row['official_hallucinated_mentions']} | "
                f"{row['official_hallucinated_mentions_per_100_tokens']:.3f} | "
                f"{row['CHAIRi']:.4f} | {row['CHAIRs']:.4f} | "
                f"{row['ObjRecall']:.4f} | {row['ObjF1']:.4f} |"
            )

    if skipped:
        lines.extend(["", "## Missing outputs", ""])
        for item in skipped:
            lines.append(f"- `{item['label']}`: {item['reason']}")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    root = accounting.repo_root()
    manifest_path = accounting.resolve_path(args.manifest, root)
    eval_root = accounting.resolve_path(args.eval_root, root)
    output_prefix = accounting.resolve_path(args.output_prefix, root)
    requested = accounting.parse_indices(args.indices)
    entries = [
        entry
        for entry in accounting.load_manifest(manifest_path)
        if entry["index"] in requested
    ]
    missing_indices = requested - {entry["index"] for entry in entries}
    if missing_indices:
        raise ValueError(f"Indices absent from manifest: {sorted(missing_indices)}")

    suffixes = list(dict.fromkeys([args.output_suffix, *args.fallback_output_suffix]))
    summarize_args = SimpleNamespace(
        expected_prompt=args.expected_prompt,
        expected_samples=args.expected_samples,
        length_mode=args.length_mode,
        model_root=args.model_root,
        best_available_step=False,
        fallback_best_available_step=False,
    )
    tokenizer_cache = {}
    methods = []
    all_candidates = []
    skipped = []

    for entry in entries:
        rows = []
        for candidate_entry in candidate_entries(entry, eval_root, suffixes):
            try:
                row = accounting.summarize_entry(
                    candidate_entry,
                    eval_root,
                    suffixes,
                    summarize_args,
                    tokenizer_cache,
                )
            except (FileNotFoundError, ValueError) as error:
                skipped.append(
                    {
                        "label": f"{entry['label']}:{candidate_entry['step']}",
                        "reason": str(error),
                    }
                )
                continue
            rows.append(row)
            all_candidates.append(row)

        if not rows:
            message = "no compatible official checkpoint output"
            if not args.allow_missing:
                raise FileNotFoundError(f"{entry['label']}: {message}")
            skipped.append({"label": entry["label"], "reason": message})
            continue

        best = min(rows, key=selection_rank)
        worst = max(rows, key=selection_rank)
        methods.append(
            {
                "scale": best["scale"],
                "method": best["method"],
                "label": entry["label"],
                "candidate_count": len(rows),
                "best": endpoint_payload(best),
                "worst": endpoint_payload(worst),
            }
        )

    by_scale_method = {
        (item["scale"], item["method"]): item
        for item in methods
    }
    stress_tests = []
    for scale in ("2b", "4b", "8b"):
        risk = by_scale_method.get((scale, "RiskMask"))
        rkl = by_scale_method.get((scale, "Frozen RKL"))
        if risk and rkl:
            stress_tests.append(
                {
                    "scale": scale,
                    "riskmask_best": risk["best"],
                    "frozen_rkl_worst": rkl["worst"],
                }
            )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    payload = {
        "report_type": "checkpoint_sensitivity_with_asymmetric_stress_test",
        "warning": (
            "RiskMask-best versus Frozen-RKL-worst is an asymmetric stress test, "
            "not a fair benchmark comparison."
        ),
        "manifest": str(manifest_path),
        "output_suffixes": suffixes,
        "expected_prompt": args.expected_prompt,
        "expected_samples": args.expected_samples,
        "methods": methods,
        "asymmetric_stress_tests": stress_tests,
        "skipped": skipped,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(methods, stress_tests, skipped), encoding="utf-8")

    csv_fields = (
        "scale",
        "method",
        "label",
        "checkpoint_step",
        "selection",
        "response_tokens_mean",
        "official_correct_mentions",
        "official_hallucinated_mentions",
        "official_hallucinated_mentions_per_100_tokens",
        "CHAIRi",
        "CHAIRs",
        "ObjPrec",
        "ObjRecall",
        "ObjF1",
        "output_dir",
    )
    endpoint_keys = {
        (item["label"], item["best"]["checkpoint_step"]): "best"
        for item in methods
    }
    for item in methods:
        key = (item["label"], item["worst"]["checkpoint_step"])
        endpoint_keys[key] = (
            "best,worst" if endpoint_keys.get(key) == "best" else "worst"
        )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in all_candidates:
            record = {field: row.get(field, "") for field in csv_fields}
            record["selection"] = endpoint_keys.get(
                (row["label"], row["checkpoint_step"]),
                "intermediate",
            )
            writer.writerow(record)

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
