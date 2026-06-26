#!/usr/bin/env python3
"""Compare two eval_results.jsonl files at caption/object level.

This is for separating output-behavior changes from KL probe changes. It
compares base model captions against a trained checkpoint's captions on the
same test set, using the official CHAIR synonym mapping and GT objects.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter


PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_OPD_ROOT = os.path.dirname(PROBE_DIR)
EVAL_DIR = os.path.join(RES_OPD_ROOT, "eval")
sys.path.insert(0, EVAL_DIR)

from robust_chair_analysis import (  # noqa: E402
    build_double_word_dict,
    caption_to_words,
    parse_official_synonyms,
)
from score_opd_eval_trace import (  # noqa: E402
    as_int,
    load_jsonl,
    load_test_index,
    normalize_generation_record,
)


DEFAULT_BASE_RESULTS = (
    "/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/"
    "res-opd/eval_results/latest/full/Qwen3VL-2B-Instruct/"
    "train5000_test1000_original_sr1p0/eval_results.jsonl"
)
DEFAULT_TR10_RESULTS = (
    "/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/"
    "res-opd/eval_results/latest/full/"
    "Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr1.0-a1.0-frozen-rkl-full5k-e1_global_step_39/"
    "train5000_test1000_original_sr1p0/eval_results.jsonl"
)
DEFAULT_TEST_JSON = os.path.join(RES_OPD_ROOT, "data", "test_1000.json")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROBE_DIR,
    "results",
    "same_image_rkl_signal",
    "eval_compare_base_vs_tr10_rkl",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare base vs trained eval_results captions.")
    parser.add_argument("--base-results", default=DEFAULT_BASE_RESULTS)
    parser.add_argument("--other-results", default=DEFAULT_TR10_RESULTS)
    parser.add_argument("--base-name", default="base")
    parser.add_argument("--other-name", default="tr1.0_rkl")
    parser.add_argument("--test-json", default=DEFAULT_TEST_JSON)
    parser.add_argument("--caption-field", default="auto")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--top-cases", type=int, default=30)
    return parser.parse_args()


def safe_div(num, den):
    return num / den if den else 0.0


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def truncate(text, limit=220):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def load_records(path, test_index, caption_field):
    records = {}
    for raw in load_jsonl(path):
        record = normalize_generation_record(raw, test_index, caption_field)
        image_id = as_int(record.get("image_id"))
        if image_id is not None:
            records[image_id] = record
    return records


def extract_objects(record, mscoco_objects, inverse_synonym_dict, double_word_dict):
    caption = record.get("generated_caption") or ""
    _, node_words = caption_to_words(caption, mscoco_objects, inverse_synonym_dict, double_word_dict)
    mentioned = set(node_words)
    gt_objects = set(record.get("gt_objects") or [])
    correct = mentioned & gt_objects
    hallucinated = mentioned - gt_objects if gt_objects else set()
    words = caption.split()
    return {
        "caption": caption,
        "mentioned": mentioned,
        "correct": correct,
        "hallucinated": hallucinated,
        "gt_objects": gt_objects,
        "num_words": len(words),
        "num_chars": len(caption),
    }


def summarize_run(parsed_by_image):
    totals = Counter()
    halluc_counter = Counter()
    correct_counter = Counter()
    for parsed in parsed_by_image.values():
        totals["records"] += 1
        totals["mentioned"] += len(parsed["mentioned"])
        totals["hallucinated"] += len(parsed["hallucinated"])
        totals["correct"] += len(parsed["correct"])
        totals["gt"] += len(parsed["gt_objects"])
        totals["words"] += parsed["num_words"]
        totals["chars"] += parsed["num_chars"]
        if parsed["hallucinated"]:
            totals["images_with_hallucination"] += 1
        halluc_counter.update(parsed["hallucinated"])
        correct_counter.update(parsed["correct"])

    precision = safe_div(totals["correct"], totals["mentioned"])
    recall = safe_div(totals["correct"], totals["gt"])
    return {
        "records": totals["records"],
        "mentioned": totals["mentioned"],
        "hallucinated": totals["hallucinated"],
        "correct": totals["correct"],
        "gt": totals["gt"],
        "chair_i": safe_div(totals["hallucinated"], totals["mentioned"]),
        "chair_s_proxy": safe_div(totals["images_with_hallucination"], totals["records"]),
        "obj_precision": precision,
        "obj_recall": recall,
        "obj_f1": safe_div(2 * precision * recall, precision + recall),
        "avg_words": safe_div(totals["words"], totals["records"]),
        "avg_chars": safe_div(totals["chars"], totals["records"]),
        "top_hallucinated_objects": halluc_counter.most_common(30),
        "top_correct_objects": correct_counter.most_common(30),
    }


def case_row(image_id, base, other):
    base_h = base["hallucinated"]
    other_h = other["hallucinated"]
    base_c = base["correct"]
    other_c = other["correct"]
    return {
        "image_id": image_id,
        "base_mentioned": sorted(base["mentioned"]),
        "other_mentioned": sorted(other["mentioned"]),
        "removed_hallucinated": sorted(base_h - other_h),
        "added_hallucinated": sorted(other_h - base_h),
        "removed_correct": sorted(base_c - other_c),
        "added_correct": sorted(other_c - base_c),
        "halluc_delta": len(other_h) - len(base_h),
        "correct_delta": len(other_c) - len(base_c),
        "mentioned_delta": len(other["mentioned"]) - len(base["mentioned"]),
        "word_delta": other["num_words"] - base["num_words"],
        "base_caption": base["caption"],
        "other_caption": other["caption"],
    }


def compare(parsed_base, parsed_other):
    common_ids = sorted(set(parsed_base) & set(parsed_other))
    rows = [case_row(image_id, parsed_base[image_id], parsed_other[image_id]) for image_id in common_ids]

    counters = {
        "removed_hallucinated": Counter(),
        "added_hallucinated": Counter(),
        "removed_correct": Counter(),
        "added_correct": Counter(),
    }
    totals = Counter()
    for row in rows:
        for key in counters:
            counters[key].update(row[key])
        totals["halluc_delta"] += row["halluc_delta"]
        totals["correct_delta"] += row["correct_delta"]
        totals["mentioned_delta"] += row["mentioned_delta"]
        totals["word_delta"] += row["word_delta"]
        if row["base_caption"].strip() == row["other_caption"].strip():
            totals["same_caption"] += 1
        if row["halluc_delta"] < 0:
            totals["images_hallucination_reduced"] += 1
        elif row["halluc_delta"] > 0:
            totals["images_hallucination_increased"] += 1
        else:
            totals["images_hallucination_same"] += 1
        if row["correct_delta"] < 0:
            totals["images_correct_reduced"] += 1
        elif row["correct_delta"] > 0:
            totals["images_correct_increased"] += 1
        else:
            totals["images_correct_same"] += 1
        if row["removed_hallucinated"] and not row["removed_correct"]:
            totals["halluc_removed_without_correct_loss"] += 1
        if row["removed_hallucinated"] and row["removed_correct"]:
            totals["halluc_removed_with_correct_loss"] += 1
        if row["added_hallucinated"]:
            totals["halluc_added_images"] += 1

    n = len(rows)
    return {
        "common_images": n,
        "base_only_images": len(set(parsed_base) - set(parsed_other)),
        "other_only_images": len(set(parsed_other) - set(parsed_base)),
        "same_caption": totals["same_caption"],
        "same_caption_frac": safe_div(totals["same_caption"], n),
        "delta_totals": {
            "hallucinated": totals["halluc_delta"],
            "correct": totals["correct_delta"],
            "mentioned": totals["mentioned_delta"],
            "words": totals["word_delta"],
        },
        "delta_means": {
            "hallucinated": safe_div(totals["halluc_delta"], n),
            "correct": safe_div(totals["correct_delta"], n),
            "mentioned": safe_div(totals["mentioned_delta"], n),
            "words": safe_div(totals["word_delta"], n),
        },
        "image_counts": dict(totals),
        "object_deltas": {key: counter.most_common(50) for key, counter in counters.items()},
        "cases": rows,
    }


def top_cases(rows, key, reverse=True, limit=30):
    return sorted(rows, key=lambda row: row[key], reverse=reverse)[:limit]


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def list_cell(values, limit=8):
    values = list(values or [])
    if len(values) > limit:
        return ", ".join(values[:limit]) + f", ...(+{len(values) - limit})"
    return ", ".join(values)


def write_markdown(summary, path, top_cases_limit):
    base_name = summary["base_name"]
    other_name = summary["other_name"]
    compare_summary = summary["comparison"]
    base = summary["base_summary"]
    other = summary["other_summary"]
    lines = [
        "# Eval Result Object Comparison",
        "",
        "## Inputs",
        "",
        f"- base: `{summary['base_results']}`",
        f"- other: `{summary['other_results']}`",
        f"- common_images: {compare_summary['common_images']}",
        f"- same_caption_frac: {fmt(compare_summary['same_caption_frac'])}",
        "",
        "## Aggregate Metrics",
        "",
        markdown_table(
            ["run", "records", "CHAIRi", "CHAIRs proxy", "ObjPrec", "ObjRec", "ObjF1", "mentioned", "halluc", "correct", "avg words"],
            [
                [
                    base_name,
                    base["records"],
                    fmt(base["chair_i"]),
                    fmt(base["chair_s_proxy"]),
                    fmt(base["obj_precision"]),
                    fmt(base["obj_recall"]),
                    fmt(base["obj_f1"]),
                    base["mentioned"],
                    base["hallucinated"],
                    base["correct"],
                    fmt(base["avg_words"], 2),
                ],
                [
                    other_name,
                    other["records"],
                    fmt(other["chair_i"]),
                    fmt(other["chair_s_proxy"]),
                    fmt(other["obj_precision"]),
                    fmt(other["obj_recall"]),
                    fmt(other["obj_f1"]),
                    other["mentioned"],
                    other["hallucinated"],
                    other["correct"],
                    fmt(other["avg_words"], 2),
                ],
            ],
        ),
        "",
        "## Paired Deltas",
        "",
        markdown_table(
            ["delta", "total", "mean/image"],
            [
                ["hallucinated", compare_summary["delta_totals"]["hallucinated"], fmt(compare_summary["delta_means"]["hallucinated"])],
                ["correct", compare_summary["delta_totals"]["correct"], fmt(compare_summary["delta_means"]["correct"])],
                ["mentioned", compare_summary["delta_totals"]["mentioned"], fmt(compare_summary["delta_means"]["mentioned"])],
                ["words", compare_summary["delta_totals"]["words"], fmt(compare_summary["delta_means"]["words"])],
            ],
        ),
        "",
        "## Image-Level Counts",
        "",
        markdown_table(
            ["bucket", "count"],
            sorted(compare_summary["image_counts"].items()),
        ),
        "",
        "## Object-Level Changes",
        "",
    ]

    for key, title in [
        ("removed_hallucinated", f"Hallucinated objects removed by {other_name}"),
        ("added_hallucinated", f"Hallucinated objects added by {other_name}"),
        ("removed_correct", f"Correct objects lost by {other_name}"),
        ("added_correct", f"Correct objects gained by {other_name}"),
    ]:
        lines.extend([
            f"### {title}",
            "",
            markdown_table(["object", "count"], compare_summary["object_deltas"][key][:20]),
            "",
        ])

    case_sections = [
        ("Hallucination Reduced Most", top_cases(compare_summary["cases"], "halluc_delta", reverse=False, limit=top_cases_limit)),
        ("Hallucination Increased Most", top_cases(compare_summary["cases"], "halluc_delta", reverse=True, limit=top_cases_limit)),
        ("Correct Objects Lost Most", top_cases(compare_summary["cases"], "correct_delta", reverse=False, limit=top_cases_limit)),
        ("Correct Objects Gained Most", top_cases(compare_summary["cases"], "correct_delta", reverse=True, limit=top_cases_limit)),
    ]
    for title, rows in case_sections:
        lines.extend([
            f"## {title}",
            "",
            markdown_table(
                [
                    "image",
                    "halluc_delta",
                    "correct_delta",
                    "removed_halluc",
                    "added_halluc",
                    "removed_correct",
                    "added_correct",
                    f"{base_name} caption",
                    f"{other_name} caption",
                ],
                [
                    [
                        row["image_id"],
                        row["halluc_delta"],
                        row["correct_delta"],
                        list_cell(row["removed_hallucinated"]),
                        list_cell(row["added_hallucinated"]),
                        list_cell(row["removed_correct"]),
                        list_cell(row["added_correct"]),
                        truncate(row["base_caption"]),
                        truncate(row["other_caption"]),
                    ]
                    for row in rows
                ],
            ),
            "",
        ])

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    args = parse_args()
    output_json = args.output_json or os.path.join(DEFAULT_OUTPUT_DIR, "eval_result_object_comparison.json")
    output_md = args.output_md or os.path.join(DEFAULT_OUTPUT_DIR, "eval_result_object_comparison.md")

    test_index = load_test_index(args.test_json)
    mscoco_objects, inverse_synonym_dict = parse_official_synonyms()
    double_word_dict = build_double_word_dict()

    base_records = load_records(args.base_results, test_index, args.caption_field)
    other_records = load_records(args.other_results, test_index, args.caption_field)
    if not base_records:
        raise SystemExit(f"No records loaded from --base-results: {args.base_results}")
    if not other_records:
        raise SystemExit(f"No records loaded from --other-results: {args.other_results}")

    parsed_base = {
        image_id: extract_objects(record, mscoco_objects, inverse_synonym_dict, double_word_dict)
        for image_id, record in base_records.items()
    }
    parsed_other = {
        image_id: extract_objects(record, mscoco_objects, inverse_synonym_dict, double_word_dict)
        for image_id, record in other_records.items()
    }
    comparison = compare(parsed_base, parsed_other)
    summary = {
        "base_name": args.base_name,
        "other_name": args.other_name,
        "base_results": args.base_results,
        "other_results": args.other_results,
        "test_json": args.test_json,
        "base_summary": summarize_run(parsed_base),
        "other_summary": summarize_run(parsed_other),
        "comparison": comparison,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_markdown(summary, output_md, args.top_cases)
    print(f"Saved comparison JSON: {output_json}")
    print(f"Saved comparison Markdown: {output_md}")


if __name__ == "__main__":
    main()
