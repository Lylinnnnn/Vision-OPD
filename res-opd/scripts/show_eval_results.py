#!/usr/bin/env python3
"""
Display CHAIR + POPE evaluation results in a unified table.

Usage:
    python scripts/show_eval_results.py [--base-dir eval_results/latest/full]

Scans all experiment directories under base-dir, reads chair_metrics.json
and pope_summary.json, and prints a formatted comparison table.
"""

import argparse
import json
import os
import sys


def find_experiments(base_dir):
    """Find all experiment result directories containing chair_metrics.json."""
    experiments = []
    for root, dirs, files in os.walk(base_dir):
        if "chair_metrics.json" in files:
            rel_path = os.path.relpath(root, base_dir)
            experiments.append((rel_path, root))
    experiments.sort(key=lambda x: x[0])
    return experiments


def load_chair(result_dir):
    """Load CHAIR metrics from chair_metrics.json."""
    path = os.path.join(result_dir, "chair_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_pope(result_dir):
    """Load POPE metrics from pope_summary.json."""
    pope_path = os.path.join(result_dir, "pope", "pope_summary.json")
    if not os.path.exists(pope_path):
        return None
    with open(pope_path) as f:
        data = json.load(f)
    benchmarks = data.get("benchmarks", {})
    # Compute macro precision and recall across subsets
    precisions = []
    recalls = []
    for subset_name, subset_data in benchmarks.items():
        if isinstance(subset_data, dict) and "precision" in subset_data:
            precisions.append(subset_data["precision"])
            recalls.append(subset_data["recall"])
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    return {
        "macro_accuracy": data.get("macro_accuracy", 0.0),
        "macro_f1": data.get("macro_f1", 0.0),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


def extract_short_name(exp_path):
    """Extract a concise experiment name from the result directory path.

    E.g. 'Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.5-a1.0-frozen-rkl-full5k-e1_global_step_39/train5000_test1000_original_sr1p0'
      -> 'orig-sr1.0-tr0.5-a1.0-frozen-rkl-full5k-e1 (step39)'
    """
    parts = exp_path.split(os.sep)
    exp_dir = parts[0] if parts else exp_path
    # Remove common prefix
    short = exp_dir.replace("Res-OPD-Qwen3VL-2B-Instruct-", "")
    # Extract step tag if present
    step_tag = ""
    if "_global_step_" in short:
        step_part = short.split("_global_step_")[-1]
        step_tag = f" (step{step_part})"
        short = short.split("_global_step_")[0]
    return short + step_tag


def format_table(experiments, base_dir):
    """Format and print the results table."""
    # Pre-compute display names to determine column width
    display_names = []
    for exp_name, result_dir in experiments:
        chair = load_chair(result_dir)
        if chair is not None:
            display_names.append(extract_short_name(exp_name))
    max_name_len = max(len(n) for n in display_names) if display_names else 30
    name_col = max(max_name_len + 1, 30)

    header = (
        f"{'Experiment':<{name_col}} "
        f"{'CHAIRi':>7} {'CHAIRs':>7} "
        f"{'ObjPrec':>8} {'ObjRec':>8} {'ObjF1':>7} "
        f"{'Mentioned':>9} {'Halluc':>7} {'Correct':>8} "
        f"{'POPE_Acc':>9} {'POPE_P':>8} {'POPE_R':>8} {'POPE_F1':>8}"
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    name_idx = 0
    for exp_name, result_dir in experiments:
        chair = load_chair(result_dir)
        pope = load_pope(result_dir)

        if chair is None:
            continue

        display_name = display_names[name_idx]
        name_idx += 1

        chair_i = chair.get("CHAIRi", 0.0)
        chair_s = chair.get("CHAIRs", 0.0)
        obj_prec = chair.get("ObjPrec", 0.0)
        obj_recall = chair.get("ObjRecall", 0.0)
        obj_f1 = chair.get("ObjF1", 0.0)
        mentioned = chair.get("total_mentioned", 0)
        hallucinated = chair.get("total_hallucinated", 0)
        correct = chair.get("total_correct", 0)

        pope_acc = pope["macro_accuracy"] if pope else 0.0
        pope_p = pope["macro_precision"] if pope else 0.0
        pope_r = pope["macro_recall"] if pope else 0.0
        pope_f1 = pope["macro_f1"] if pope else 0.0

        print(
            f"{display_name:<{name_col}} "
            f"{chair_i:>7.4f} {chair_s:>7.4f} "
            f"{obj_prec:>8.4f} {obj_recall:>8.4f} {obj_f1:>7.4f} "
            f"{mentioned:>9d} {hallucinated:>7d} {correct:>8d} "
            f"{pope_acc:>9.4f} {pope_p:>8.4f} {pope_r:>8.4f} {pope_f1:>8.4f}"
        )

    print(separator)


def main():
    parser = argparse.ArgumentParser(description="Display CHAIR + POPE eval results")
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory containing experiment results (default: auto-detect)",
    )
    args = parser.parse_args()

    if args.base_dir:
        base_dir = args.base_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        res_opd_root = os.path.dirname(script_dir)
        base_dir = os.path.join(res_opd_root, "eval_results", "latest", "full")

    if not os.path.isdir(base_dir):
        print(f"Error: directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    experiments = find_experiments(base_dir)
    if not experiments:
        print(f"No experiment results found in {base_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n📊 Evaluation Results ({len(experiments)} experiments)")
    print(f"   Base dir: {base_dir}\n")
    format_table(experiments, base_dir)
    print()


if __name__ == "__main__":
    main()
