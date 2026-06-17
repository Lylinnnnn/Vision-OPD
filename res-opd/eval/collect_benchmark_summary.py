#!/usr/bin/env python3
"""Collect compact Res-OPD validation metrics into one summary file."""

import argparse
import csv
import json
from pathlib import Path


POPE_BENCHMARKS = {"pope", "pope_adv", "pope_pop", "pope_random"}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_correct(item: dict) -> bool:
    return str(item.get("judge", "")).strip().lower() == "yes"


def extract_yes_no(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    if text.startswith("answer"):
        text = text.split("answer", 1)[1].lstrip(":").strip()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    first_sentence = text.split(".", 1)[0].replace(",", "")
    words = first_sentence.split()
    if "no" in words or "not" in words:
        return "no"
    if "yes" in words:
        return "yes"
    return ""


def summarize_judge_file(path: Path, benchmark: str) -> dict:
    data = load_json(path)
    total = len(data)
    correct = sum(1 for item in data if is_correct(item))
    summary = {
        "benchmark": benchmark,
        "accuracy": correct / total if total else 0.0,
        "num_samples": total,
        "source_path": str(path),
    }

    if benchmark in POPE_BENCHMARKS:
        tp = fp = tn = fn = pred_yes = 0
        for item in data:
            gt = str(item.get("response", "")).strip().lower()
            pred = extract_yes_no(item.get("extracted_answer", item.get("model_answer", "")))
            pred_yes += int(pred == "yes")
            if gt == "yes" and pred == "yes":
                tp += 1
            elif gt == "no" and pred == "yes":
                fp += 1
            elif gt == "no" and pred == "no":
                tn += 1
            elif gt == "yes" and pred == "no":
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        summary.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_yes_ratio": pred_yes / total if total else 0.0,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    return summary


def collect_res_opd_metrics(output_dir: Path) -> dict:
    metrics = {}
    chair_path = output_dir / "chair_metrics.json"
    if chair_path.exists():
        chair = load_json(chair_path)
        metrics["chair"] = {
            "benchmark": "chair",
            "CHAIRi": chair.get("CHAIRi"),
            "CHAIRs": chair.get("CHAIRs"),
            "ObjPrec": chair.get("ObjPrec"),
            "ObjRecall": chair.get("ObjRecall"),
            "ObjF1": chair.get("ObjF1"),
            "RepRate": chair.get("RepRate"),
            "num_samples": chair.get("num_samples"),
            "source_path": str(chair_path),
        }

    pope_summary_path = output_dir / "pope" / "pope_summary.json"
    if pope_summary_path.exists():
        pope = load_json(pope_summary_path)
        split_metrics = pope.get("benchmarks", {})
        metrics["pope"] = {
            "benchmark": "pope",
            "accuracy": pope.get("macro_accuracy"),
            "f1": pope.get("macro_f1"),
            "splits": split_metrics,
            "source_path": str(pope_summary_path),
        }
        if split_metrics:
            metrics["pope"]["recall"] = sum(m.get("recall", 0.0) for m in split_metrics.values()) / len(split_metrics)
            metrics["pope"]["predicted_yes_ratio"] = (
                sum(m.get("predicted_yes_ratio", 0.0) for m in split_metrics.values()) / len(split_metrics)
            )

    mme_path = output_dir / "mme" / "mme_metrics.json"
    if mme_path.exists():
        mme = load_json(mme_path)
        metrics["mme"] = {
            "benchmark": "mme",
            "total_score": mme.get("total_score"),
            "perception_score": mme.get("perception_score"),
            "cognition_score": mme.get("cognition_score"),
            "unknown_score": mme.get("unknown_score"),
            "num_samples": mme.get("num_samples"),
            "source_path": str(mme_path),
        }

    amber_path = output_dir / "amber" / "amber_metrics.json"
    if amber_path.exists():
        amber = load_json(amber_path)
        amber_metrics = amber.get("metrics", {})
        metrics["amber"] = {
            "benchmark": "amber",
            "evaluation_type": amber.get("evaluation_type"),
            "official_returncode": amber.get("official_returncode"),
            "source_path": str(amber_path),
        }
        for key, value in amber_metrics.items():
            metrics["amber"][key] = value

    return metrics


def collect_vision_opd_metrics(output_dir: Path) -> dict:
    metrics = {}
    judge_root = output_dir / "vision_opd" / "judge"
    if not judge_root.exists():
        return metrics
    for judge_file in sorted(judge_root.glob("*/*_answer.jsonl")):
        benchmark = judge_file.parent.name
        metrics[f"vision_opd/{benchmark}"] = summarize_judge_file(judge_file, benchmark)
    return metrics


def flatten_for_csv(metrics: dict) -> list[dict]:
    rows = []
    for name, values in metrics.items():
        row = {"name": name}
        for key, value in values.items():
            if isinstance(value, (str, int, float)) or value is None:
                row[key] = value
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect compact benchmark metrics")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    metrics.update(collect_res_opd_metrics(output_dir))
    metrics.update(collect_vision_opd_metrics(output_dir))

    summary = {
        "output_dir": str(output_dir),
        "metrics": metrics,
    }

    json_path = output_dir / "compact_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    rows = flatten_for_csv(metrics)
    csv_path = output_dir / "compact_summary.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved compact summary to: {json_path}")
    if rows:
        print(f"Saved compact CSV to: {csv_path}")


if __name__ == "__main__":
    main()
