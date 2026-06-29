#!/usr/bin/env python3
"""Decompose CHAIR-style object changes into artifact/uncertainty buckets.

This offline diagnostic compares a base eval_results.jsonl with one or more
trained eval_results.jsonl files. It asks whether hallucination changes are
mostly ordinary visual object changes, or concentrated in text/logo/graphic,
color-as-object, and uncertain/small/background mentions.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict


PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_OPD_ROOT = os.path.dirname(PROBE_DIR)
EVAL_DIR = os.path.join(RES_OPD_ROOT, "eval")
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, PROBE_DIR)

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
from probe_same_image_rkl_signal import build_canonical_synonym_map  # noqa: E402


DEFAULT_BASE_RESULTS = (
    "/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/"
    "res-opd/eval_results/latest/full/Qwen3VL-2B-Instruct/"
    "train5000_test1000_original_sr1p0/eval_results.jsonl"
)
DEFAULT_TEST_JSON = os.path.join(RES_OPD_ROOT, "data", "test_1000.json")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROBE_DIR,
    "results",
    "caption_object_decomposition",
    "full5k_base_vs_variants",
)
DEFAULT_RUNS = [
    (
        "tr10_rkl",
        "/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/latest/full/"
        "Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr1.0-a1.0-frozen-rkl-full5k-e1_global_step_39/"
        "train5000_test1000_original_sr1p0/eval_results.jsonl",
    ),
    (
        "tr075_rkl",
        "/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/latest/full/"
        "Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-full5k-e1_global_step_39/"
        "train5000_test1000_original_sr1p0/eval_results.jsonl",
    ),
    (
        "tr075_rkl_sw",
        "/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/latest/full/"
        "Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-sw-full5k-e1_global_step_39/"
        "train5000_test1000_original_sr1p0/eval_results.jsonl",
    ),
    (
        "tr075_rkl_sw075",
        "/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/latest/full/"
        "Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-sw075-full5k-e1_global_step_39/"
        "train5000_test1000_original_sr1p0/eval_results.jsonl",
    ),
]

TEXT_OR_GRAPHIC_RE = re.compile(
    r"\b("
    r"sign|text|word|words|read|reads|says|written|letter|letters|number|numbers|"
    r"logo|brand|emblem|icon|symbol|label|sticker|banner|poster|advertisement|ad|"
    r"screen|display|monitor|license plate|graphic|image|picture|photo|photograph|"
    r"drawing|illustration|cartoon|painting|mural|print|printed|pattern|design|"
    r"face on|depicts|depicted|shaped like"
    r")\b",
    re.IGNORECASE,
)
UNCERTAIN_SMALL_RE = re.compile(
    r"\b("
    r"possibly|probably|likely|appears|appear|seems|seem|might|may be|could be|"
    r"partially|small|tiny|distant|far|background|foreground edge|faint|blurred|"
    r"blurry|out of focus|obscured|partly hidden|silhouette|dark shape|hard to see|"
    r"not fully visible|barely visible|indistinct"
    r")\b",
    re.IGNORECASE,
)
COLOR_ATTR_RE = re.compile(
    r"\b("
    r"orange[- ]colored|bright orange|orange color|orange hue|orange stripe|orange stripes|"
    r"orange accent|orange accents|orange trim|orange paint|orange painted|orange jacket|"
    r"orange shirt|orange turn signal|orange cone|orange ribbon|orange background|"
    r"orange roof|orange panel|orange light|orange lights|orange object"
    r")\b",
    re.IGNORECASE,
)
FRUIT_RE = re.compile(r"\b(fruit|fruits|slice|slices|peeled|peel|juice|citrus|basket|bowl of oranges)\b", re.IGNORECASE)

COCO_GROUPS = {
    "person": {"person"},
    "vehicle": {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"},
    "outdoor_sign": {"traffic light", "fire hydrant", "stop sign", "parking meter", "bench"},
    "animal": {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"},
    "accessory": {"backpack", "umbrella", "handbag", "tie", "suitcase"},
    "sports": {
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
    },
    "kitchen_tableware": {"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"},
    "food": {
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
    },
    "furniture": {"chair", "couch", "potted plant", "bed", "dining table", "toilet"},
    "electronics_appliance": {
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
    },
    "indoor_other": {"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"},
}
OBJECT_TO_GROUP = {
    obj: group
    for group, objects in COCO_GROUPS.items()
    for obj in objects
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decompose base-vs-model CHAIR object changes into interpretable mention categories."
    )
    parser.add_argument("--base-results", default=DEFAULT_BASE_RESULTS)
    parser.add_argument("--base-name", default="base")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec as name=/path/to/eval_results.jsonl. Can be passed multiple times.",
    )
    parser.add_argument(
        "--runs-json",
        default=None,
        help="Optional JSON list/dict of runs. List items need name/results; dict maps name to path.",
    )
    parser.add_argument("--test-json", default=DEFAULT_TEST_JSON)
    parser.add_argument("--caption-field", default="auto")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument("--changes-jsonl", default=None)
    parser.add_argument("--top-examples", type=int, default=25)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any default/provided run is missing. By default missing runs are skipped.",
    )
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


def compact_text(text, limit=220):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def bounded_context(text, start, end, window=90):
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right].replace("\n", " ")
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet += "..."
    return snippet


def spans_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def load_records(path, test_index, caption_field):
    records = {}
    for raw in load_jsonl(path):
        record = normalize_generation_record(raw, test_index, caption_field)
        image_id = as_int(record.get("image_id"))
        if image_id is not None and record.get("generated_caption"):
            records[image_id] = record
    return records


def extract_chair_objects(caption, mscoco_objects, inverse_synonym_dict, double_word_dict):
    _, node_words = caption_to_words(caption, mscoco_objects, inverse_synonym_dict, double_word_dict)
    return set(node_words)


def regex_mentions(caption, canonical_synonym_map, allowed_objects):
    lowered = caption.lower()
    candidates = []
    for canonical, terms in canonical_synonym_map.items():
        if canonical not in allowed_objects:
            continue
        for term in terms:
            if term:
                candidates.append((canonical, term))
    candidates.sort(key=lambda item: (-len(item[1]), item[0], item[1]))

    selected_spans_by_object = defaultdict(list)
    mentions_by_object = defaultdict(list)
    for canonical, term in candidates:
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        pattern = r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
        for match in re.finditer(pattern, lowered):
            char_span = (match.start(), match.end())
            if any(spans_overlap(char_span, existing) for existing in selected_spans_by_object[canonical]):
                continue
            selected_spans_by_object[canonical].append(char_span)
            context = bounded_context(caption, match.start(), match.end())
            mentions_by_object[canonical].append(
                {
                    "source_term": term,
                    "mention_text": caption[match.start():match.end()],
                    "context": context,
                    "char_start": match.start(),
                    "char_end": match.end(),
                }
            )
    return mentions_by_object


def is_uppercase_mention(mention_text):
    text = re.sub(r"[^A-Za-z0-9]+", "", str(mention_text or ""))
    return len(text) > 1 and text.upper() == text and any(ch.isalpha() for ch in text)


def classify_mention(obj, mention):
    context = mention.get("context") or ""
    mention_text = mention.get("mention_text") or ""
    text_or_graphic = bool(TEXT_OR_GRAPHIC_RE.search(context)) or is_uppercase_mention(mention_text)
    uncertain_small = bool(UNCERTAIN_SMALL_RE.search(context))
    color_attribute = (
        obj == "orange"
        and bool(COLOR_ATTR_RE.search(context))
        and not bool(FRUIT_RE.search(context))
    )
    return {
        "text_or_graphic": text_or_graphic,
        "uncertain_small_background": uncertain_small,
        "color_attribute": color_attribute,
        "uppercase_mention": is_uppercase_mention(mention_text),
    }


def primary_category(flags):
    if flags.get("color_attribute"):
        return "color_attribute_as_object"
    if flags.get("text_or_graphic"):
        return "text_logo_graphic"
    if flags.get("uncertain_small_background"):
        return "uncertain_small_background"
    return "ordinary_visual_object"


def object_group(obj):
    return OBJECT_TO_GROUP.get(obj, "other")


def object_rows_for_record(record, mscoco_objects, inverse_synonym_dict, double_word_dict, canonical_synonym_map):
    caption = record.get("generated_caption") or ""
    mentioned = extract_chair_objects(caption, mscoco_objects, inverse_synonym_dict, double_word_dict)
    gt_objects = set(record.get("gt_objects") or [])
    mentions_by_object = regex_mentions(caption, canonical_synonym_map, mentioned)

    rows = {}
    for obj in sorted(mentioned):
        mentions = mentions_by_object.get(obj) or []
        flags = {
            "text_or_graphic": False,
            "uncertain_small_background": False,
            "color_attribute": False,
            "uppercase_mention": False,
        }
        for mention in mentions:
            current = classify_mention(obj, mention)
            for key, value in current.items():
                flags[key] = flags[key] or value

        example_mention = mentions[0] if mentions else {}
        rows[obj] = {
            "object": obj,
            "label": "correct" if obj in gt_objects else "hallucinated",
            "primary_category": primary_category(flags),
            "object_group": object_group(obj),
            "flags": flags,
            "mention_count": len(mentions),
            "source_terms": sorted({m.get("source_term") for m in mentions if m.get("source_term")}),
            "mention_texts": sorted({m.get("mention_text") for m in mentions if m.get("mention_text")}),
            "example_context": example_mention.get("context", ""),
            "caption": caption,
            "gt_objects": sorted(gt_objects),
        }
    return rows


def summarize_run(object_rows_by_image):
    totals = Counter()
    object_counter = Counter()
    category_counter = Counter()
    group_counter = Counter()
    for rows in object_rows_by_image.values():
        totals["records"] += 1
        gt_objects = set()
        for row in rows.values():
            gt_objects.update(row.get("gt_objects") or [])
        totals["gt"] += len(gt_objects)
        totals["mentioned"] += len(rows)
        for row in rows.values():
            label = row["label"]
            totals[label] += 1
            object_counter[(label, row["object"])] += 1
            category_counter[(label, row["primary_category"])] += 1
            group_counter[(label, row["object_group"])] += 1
        if any(row["label"] == "hallucinated" for row in rows.values()):
            totals["images_with_hallucination"] += 1
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
        "top_hallucinated_objects": [
            [obj, count]
            for (label, obj), count in object_counter.most_common()
            if label == "hallucinated"
        ][:30],
        "top_correct_objects": [
            [obj, count]
            for (label, obj), count in object_counter.most_common()
            if label == "correct"
        ][:30],
        "category_counts": counter_to_nested(category_counter),
        "group_counts": counter_to_nested(group_counter),
    }


def counter_to_nested(counter):
    out = defaultdict(dict)
    for (label, key), count in counter.items():
        out[label][key] = count
    return {label: dict(values) for label, values in out.items()}


def status_bucket(status, label):
    return f"{status}_{label}"


def add_change_row(rows, run_name, image_id, status, source_run, source_row):
    row = {
        "run": run_name,
        "image_id": image_id,
        "status": status,
        "bucket": status_bucket(status, source_row["label"]),
        "source_run": source_run,
    }
    for key in (
        "object",
        "label",
        "primary_category",
        "object_group",
        "flags",
        "mention_count",
        "source_terms",
        "mention_texts",
        "example_context",
    ):
        row[key] = source_row.get(key)
    rows.append(row)


def summarize_change_rows(rows):
    total = len(rows)
    by_status = Counter(row["status"] for row in rows)
    by_bucket = Counter(row["bucket"] for row in rows)
    by_category = nested_change_summary(rows, "primary_category")
    by_group = nested_change_summary(rows, "object_group")
    flag_summary = {}
    for flag in ("text_or_graphic", "color_attribute", "uncertain_small_background", "uppercase_mention"):
        flagged = [row for row in rows if (row.get("flags") or {}).get(flag)]
        flag_summary[flag] = basic_change_summary(flagged, total)
    return {
        "total_change_rows": total,
        "by_status": dict(by_status),
        "by_bucket": dict(by_bucket),
        "by_primary_category": by_category,
        "by_object_group": by_group,
        "by_flag": flag_summary,
        "net": {
            "hallucinated_delta": by_bucket.get("added_hallucinated", 0) - by_bucket.get("removed_hallucinated", 0),
            "correct_delta": by_bucket.get("added_correct", 0) - by_bucket.get("removed_correct", 0),
            "mentioned_delta": (
                by_status.get("added", 0)
                - by_status.get("removed", 0)
            ),
        },
    }


def basic_change_summary(rows, denominator=None):
    denominator = len(rows) if denominator is None else denominator
    by_bucket = Counter(row["bucket"] for row in rows)
    by_status = Counter(row["status"] for row in rows)
    return {
        "count": len(rows),
        "share": safe_div(len(rows), denominator),
        "removed_hallucinated": by_bucket.get("removed_hallucinated", 0),
        "added_hallucinated": by_bucket.get("added_hallucinated", 0),
        "removed_correct": by_bucket.get("removed_correct", 0),
        "added_correct": by_bucket.get("added_correct", 0),
        "kept_hallucinated": by_bucket.get("kept_hallucinated", 0),
        "kept_correct": by_bucket.get("kept_correct", 0),
        "net_hallucinated_delta": by_bucket.get("added_hallucinated", 0) - by_bucket.get("removed_hallucinated", 0),
        "net_correct_delta": by_bucket.get("added_correct", 0) - by_bucket.get("removed_correct", 0),
        "removed": by_status.get("removed", 0),
        "added": by_status.get("added", 0),
        "kept": by_status.get("kept", 0),
    }


def nested_change_summary(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get(field) or "unknown"].append(row)
    out = {}
    total_removed_hallucinated = sum(1 for row in rows if row["bucket"] == "removed_hallucinated")
    total_removed_correct = sum(1 for row in rows if row["bucket"] == "removed_correct")
    for key, group_rows in sorted(grouped.items()):
        summary = basic_change_summary(group_rows, len(rows))
        summary["share_of_removed_hallucinated"] = safe_div(
            summary["removed_hallucinated"],
            total_removed_hallucinated,
        )
        summary["share_of_removed_correct"] = safe_div(summary["removed_correct"], total_removed_correct)
        out[key] = summary
    return out


def compare_runs(run_name, base_objects, other_objects, base_name, other_name):
    common_ids = sorted(set(base_objects) & set(other_objects))
    changes = []
    image_rows = []
    for image_id in common_ids:
        base_rows = base_objects[image_id]
        other_rows = other_objects[image_id]
        base_set = set(base_rows)
        other_set = set(other_rows)
        removed = sorted(base_set - other_set)
        added = sorted(other_set - base_set)
        kept = sorted(base_set & other_set)
        for obj in removed:
            add_change_row(changes, run_name, image_id, "removed", base_name, base_rows[obj])
        for obj in added:
            add_change_row(changes, run_name, image_id, "added", other_name, other_rows[obj])
        for obj in kept:
            add_change_row(changes, run_name, image_id, "kept", base_name, base_rows[obj])
        image_rows.append(
            {
                "run": run_name,
                "image_id": image_id,
                "removed": len(removed),
                "added": len(added),
                "kept": len(kept),
                "removed_hallucinated": sorted(
                    obj for obj in removed if base_rows[obj]["label"] == "hallucinated"
                ),
                "removed_correct": sorted(obj for obj in removed if base_rows[obj]["label"] == "correct"),
                "added_hallucinated": sorted(
                    obj for obj in added if other_rows[obj]["label"] == "hallucinated"
                ),
                "added_correct": sorted(obj for obj in added if other_rows[obj]["label"] == "correct"),
            }
        )
    return changes, image_rows


def parse_run_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Run spec must be name=/path/to/eval_results.jsonl, got: {spec}")
    name, path = spec.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Invalid run spec: {spec}")
    return name, path


def collect_run_specs(args):
    specs = []
    if args.runs_json:
        with open(args.runs_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            specs.extend((str(name), str(path)) for name, path in data.items())
        elif isinstance(data, list):
            for item in data:
                specs.append((str(item["name"]), str(item["results"])))
        else:
            raise SystemExit("--runs-json must be a dict or list")
    for spec in args.run:
        specs.append(parse_run_spec(spec))
    if not specs:
        specs = list(DEFAULT_RUNS)

    seen = set()
    unique = []
    for name, path in specs:
        if name in seen:
            raise SystemExit(f"Duplicate run name: {name}")
        seen.add(name)
        unique.append((name, path))
    return unique


def prepare_runs(args):
    warnings = []
    if not os.path.exists(args.base_results):
        raise SystemExit(f"Base eval_results missing: {args.base_results}")
    runs = []
    for name, path in collect_run_specs(args):
        if not os.path.exists(path):
            message = f"Run '{name}' eval_results missing: {path}"
            if args.strict:
                raise SystemExit(message)
            warnings.append(message)
            continue
        runs.append((name, path))
    if not runs:
        raise SystemExit("No model runs available after filtering missing eval_results.")
    return runs, warnings


def build_object_index(records, mscoco_objects, inverse_synonym_dict, double_word_dict, canonical_synonym_map):
    return {
        image_id: object_rows_for_record(
            record,
            mscoco_objects,
            inverse_synonym_dict,
            double_word_dict,
            canonical_synonym_map,
        )
        for image_id, record in records.items()
    }


def top_examples(rows, bucket=None, category=None, limit=25):
    selected = rows
    if bucket:
        selected = [row for row in selected if row["bucket"] == bucket]
    if category:
        selected = [row for row in selected if row["primary_category"] == category]
    selected = sorted(
        selected,
        key=lambda row: (
            row["run"],
            row["bucket"],
            row["primary_category"],
            row["object"],
            row["image_id"],
        ),
    )
    return [
        {
            "run": row["run"],
            "image_id": row["image_id"],
            "bucket": row["bucket"],
            "object": row["object"],
            "category": row["primary_category"],
            "group": row["object_group"],
            "terms": row.get("mention_texts"),
            "context": compact_text(row.get("example_context"), 260),
        }
        for row in selected[:limit]
    ]


def analyze(args):
    runs, warnings = prepare_runs(args)
    test_index = load_test_index(args.test_json)
    mscoco_objects, inverse_synonym_dict = parse_official_synonyms()
    double_word_dict = build_double_word_dict()
    canonical_synonym_map = build_canonical_synonym_map(inverse_synonym_dict)

    base_records = load_records(args.base_results, test_index, args.caption_field)
    if not base_records:
        raise SystemExit(f"No base records loaded from {args.base_results}")
    base_objects = build_object_index(
        base_records,
        mscoco_objects,
        inverse_synonym_dict,
        double_word_dict,
        canonical_synonym_map,
    )

    summary = {
        "config": {
            "base_name": args.base_name,
            "base_results": args.base_results,
            "test_json": args.test_json,
            "caption_field": args.caption_field,
            "runs": [{"name": name, "results": path} for name, path in runs],
        },
        "warnings": warnings,
        "base_summary": summarize_run(base_objects),
        "runs": {},
    }
    all_changes = []
    all_image_rows = []
    for name, path in runs:
        other_records = load_records(path, test_index, args.caption_field)
        if not other_records:
            if args.strict:
                raise SystemExit(f"No records loaded for {name}: {path}")
            summary["warnings"].append(f"No records loaded for {name}: {path}")
            continue
        other_objects = build_object_index(
            other_records,
            mscoco_objects,
            inverse_synonym_dict,
            double_word_dict,
            canonical_synonym_map,
        )
        changes, image_rows = compare_runs(name, base_objects, other_objects, args.base_name, name)
        all_changes.extend(changes)
        all_image_rows.extend(image_rows)
        other_summary = summarize_run(other_objects)
        change_summary = summarize_change_rows(changes)
        summary["runs"][name] = {
            "results": path,
            "common_images": len(set(base_objects) & set(other_objects)),
            "base_only_images": len(set(base_objects) - set(other_objects)),
            "other_only_images": len(set(other_objects) - set(base_objects)),
            "run_summary": other_summary,
            "change_summary": change_summary,
            "examples": {
                "removed_hallucinated_text_logo_graphic": top_examples(
                    changes, "removed_hallucinated", "text_logo_graphic", args.top_examples
                ),
                "removed_hallucinated_uncertain_small": top_examples(
                    changes, "removed_hallucinated", "uncertain_small_background", args.top_examples
                ),
                "removed_correct_text_logo_graphic": top_examples(
                    changes, "removed_correct", "text_logo_graphic", args.top_examples
                ),
                "removed_correct_uncertain_small": top_examples(
                    changes, "removed_correct", "uncertain_small_background", args.top_examples
                ),
            },
        }
    return summary, all_changes, all_image_rows


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        clean = [str(value).replace("\n", " ").replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(clean) + " |")
    return "\n".join(lines)


def category_rows(change_summary):
    rows = []
    for category, stats in sorted(
        change_summary["by_primary_category"].items(),
        key=lambda item: item[1].get("removed_hallucinated", 0),
        reverse=True,
    ):
        rows.append(
            [
                category,
                stats["removed_hallucinated"],
                fmt(stats["share_of_removed_hallucinated"]),
                stats["added_hallucinated"],
                stats["net_hallucinated_delta"],
                stats["removed_correct"],
                fmt(stats["share_of_removed_correct"]),
                stats["added_correct"],
                stats["net_correct_delta"],
            ]
        )
    return rows


def flag_rows(change_summary):
    rows = []
    for flag, stats in change_summary["by_flag"].items():
        rows.append(
            [
                flag,
                stats["removed_hallucinated"],
                stats["added_hallucinated"],
                stats["net_hallucinated_delta"],
                stats["removed_correct"],
                stats["added_correct"],
                stats["net_correct_delta"],
            ]
        )
    return rows


def write_markdown(path, summary):
    base = summary["base_summary"]
    lines = [
        "# Caption Object Change Decomposition",
        "",
        "## Inputs",
        "",
        f"- base: `{summary['config']['base_results']}`",
        f"- test_json: `{summary['config']['test_json']}`",
    ]
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
        ]
    )
    metric_rows = [
        [
            summary["config"]["base_name"],
            base["records"],
            fmt(base["chair_i"]),
            fmt(base["chair_s_proxy"]),
            fmt(base["obj_precision"]),
            fmt(base["obj_recall"]),
            fmt(base["obj_f1"]),
            base["mentioned"],
            base["hallucinated"],
            base["correct"],
        ]
    ]
    for name, run in summary["runs"].items():
        current = run["run_summary"]
        metric_rows.append(
            [
                name,
                current["records"],
                fmt(current["chair_i"]),
                fmt(current["chair_s_proxy"]),
                fmt(current["obj_precision"]),
                fmt(current["obj_recall"]),
                fmt(current["obj_f1"]),
                current["mentioned"],
                current["hallucinated"],
                current["correct"],
            ]
        )
    lines.append(
        markdown_table(
            ["run", "records", "CHAIRi", "CHAIRs proxy", "ObjPrec", "ObjRec", "ObjF1", "mentioned", "halluc", "correct"],
            metric_rows,
        )
    )
    lines.extend(["", "## Paired Change Totals", ""])
    delta_rows = []
    for name, run in summary["runs"].items():
        stats = run["change_summary"]
        buckets = stats["by_bucket"]
        delta_rows.append(
            [
                name,
                buckets.get("removed_hallucinated", 0),
                buckets.get("added_hallucinated", 0),
                stats["net"]["hallucinated_delta"],
                buckets.get("removed_correct", 0),
                buckets.get("added_correct", 0),
                stats["net"]["correct_delta"],
                stats["net"]["mentioned_delta"],
            ]
        )
    lines.append(
        markdown_table(
            [
                "run",
                "removed halluc",
                "added halluc",
                "net halluc",
                "removed correct",
                "added correct",
                "net correct",
                "net mentioned",
            ],
            delta_rows,
        )
    )

    for name, run in summary["runs"].items():
        lines.extend(["", f"## {name}", "", "### Category Decomposition", ""])
        lines.append(
            markdown_table(
                [
                    "category",
                    "removed halluc",
                    "share removed halluc",
                    "added halluc",
                    "net halluc",
                    "removed correct",
                    "share removed correct",
                    "added correct",
                    "net correct",
                ],
                category_rows(run["change_summary"]),
            )
        )
        lines.extend(["", "### Flag Decomposition", ""])
        lines.append(
            markdown_table(
                [
                    "flag",
                    "removed halluc",
                    "added halluc",
                    "net halluc",
                    "removed correct",
                    "added correct",
                    "net correct",
                ],
                flag_rows(run["change_summary"]),
            )
        )
        for title, key in [
            ("Removed Hallucinated: Text/Logo/Graphic", "removed_hallucinated_text_logo_graphic"),
            ("Removed Hallucinated: Uncertain/Small/Background", "removed_hallucinated_uncertain_small"),
            ("Removed Correct: Text/Logo/Graphic", "removed_correct_text_logo_graphic"),
            ("Removed Correct: Uncertain/Small/Background", "removed_correct_uncertain_small"),
        ]:
            examples = run.get("examples", {}).get(key, [])
            if not examples:
                continue
            lines.extend(["", f"### {title}", ""])
            lines.append(
                markdown_table(
                    ["image", "object", "category", "group", "terms", "context"],
                    [
                        [
                            ex["image_id"],
                            ex["object"],
                            ex["category"],
                            ex["group"],
                            ", ".join(ex.get("terms") or []),
                            ex["context"],
                        ]
                        for ex in examples
                    ],
                )
            )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    args = parse_args()
    summary_json = args.summary_json or os.path.join(args.output_dir, "caption_object_decomposition_summary.json")
    summary_md = args.summary_md or os.path.join(args.output_dir, "caption_object_decomposition_summary.md")
    changes_jsonl = args.changes_jsonl or os.path.join(args.output_dir, "caption_object_changes.jsonl")
    image_jsonl = os.path.join(args.output_dir, "caption_object_image_changes.jsonl")

    summary, changes, image_rows = analyze(args)
    write_json(summary_json, summary)
    write_markdown(summary_md, summary)
    write_jsonl(changes_jsonl, changes)
    write_jsonl(image_jsonl, image_rows)
    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote summary MD:   {summary_md}")
    print(f"Wrote changes JSONL:{changes_jsonl}")
    print(f"Wrote image JSONL:  {image_jsonl}")


if __name__ == "__main__":
    main()
