#!/usr/bin/env python3
"""Rank qualitative CHAIR cases across paper experiments.

Each experiment is compared with the original-image base model of the same
model size.  Object extraction follows the repository's official CHAIR parser.
The script is intentionally a first-stage miner: candidates still require
visual inspection because COCO annotations can be incomplete.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


RES_OPD_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = RES_OPD_DIR / "eval"
sys.path.insert(0, str(EVAL_DIR))

from robust_chair_analysis import (  # noqa: E402
    build_double_word_dict,
    caption_to_words,
    parse_official_synonyms,
)


DEFAULT_RESULTS_SUBDIR = "chair_describe_image_official"
DEFAULT_EXPECTED_PROMPT = "Describe the image."
TIER_A = "tier_a_official_chairs_clean_flip"
TIER_B = "tier_b_official_chairi_mention_reduction"
TIER_C = "tier_c_official_chairi_ratio_improvement"
TIER_PRIORITY = {TIER_A: 3, TIER_B: 2, TIER_C: 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare every paper experiment with its same-size base model and "
            "rank visually reviewable qualitative cases."
        )
    )
    parser.add_argument(
        "--preflight-json",
        required=True,
        help="paper_chair_describe45_preflight.json generated on the server.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--results-subdir",
        default=DEFAULT_RESULTS_SUBDIR,
        help=(
            "Prompt-isolated result directory below each selected result dir "
            f"(default: {DEFAULT_RESULTS_SUBDIR})."
        ),
    )
    parser.add_argument(
        "--expected-prompt",
        default=DEFAULT_EXPECTED_PROMPT,
        help="Require every compared row to use this exact evaluation prompt.",
    )
    parser.add_argument(
        "--top-per-experiment",
        type=int,
        default=20,
        help="Candidates retained for each unique experiment.",
    )
    parser.add_argument(
        "--global-limit",
        type=int,
        default=120,
        help="Maximum candidates in the diverse global shortlist.",
    )
    parser.add_argument(
        "--tier-limit",
        type=int,
        default=1000,
        help="Maximum candidates written to each tier-specific JSONL.",
    )
    parser.add_argument(
        "--max-per-experiment",
        type=int,
        default=5,
        help="Per-experiment cap in the diverse global shortlist.",
    )
    parser.add_argument(
        "--min-common-samples",
        type=int,
        default=900,
        help="Fail when an experiment and its base have fewer aligned samples.",
    )
    parser.add_argument(
        "--min-correct-mention-retention",
        type=float,
        default=0.80,
        help=(
            "Quality guard: retain at least this fraction of Base correct object "
            "mentions (default: 0.80)."
        ),
    )
    parser.add_argument(
        "--min-word-retention",
        type=float,
        default=0.60,
        help=(
            "Quality guard: retain at least this fraction of Base caption words "
            "(default: 0.60)."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Warn and skip missing result files instead of failing.",
    )
    args = parser.parse_args()
    for name in ("min_correct_mention_retention", "min_word_retention"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    return args


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_image_id(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text.isdigit():
        return str(int(text))
    return text


def read_eval_results(path: Path, expected_prompt: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    bad_rows: list[str] = []
    prompts: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

            image_id = row.get("image_id", row.get("id"))
            caption = row.get("generated_caption", row.get("model_answer", ""))
            prompt = str(row.get("eval_prompt", row.get("query", ""))).strip()
            if image_id is None:
                bad_rows.append(f"line {line_number}: missing image_id")
                continue
            if not isinstance(caption, str) or not caption.strip():
                bad_rows.append(f"line {line_number}: empty generated caption")
                continue
            if caption.startswith("[ERROR]"):
                bad_rows.append(f"line {line_number}: generation error")
                continue
            if prompt:
                prompts.add(prompt)

            key = normalize_image_id(image_id)
            if key in records:
                raise ValueError(f"{path}: duplicate image_id={key}")
            records[key] = row

    if bad_rows:
        preview = "; ".join(bad_rows[:5])
        raise ValueError(f"{path}: {len(bad_rows)} invalid rows ({preview})")
    if expected_prompt and prompts != {expected_prompt}:
        raise ValueError(
            f"{path}: prompt mismatch; expected {expected_prompt!r}, found {sorted(prompts)!r}"
        )
    if not records:
        raise ValueError(f"{path}: no valid records")
    return records


def model_size_from_label(label: str) -> str:
    match = re.match(r"^(2b|4b|8b)/", label.lower())
    if not match:
        raise ValueError(f"Cannot infer model size from label: {label!r}")
    return match.group(1)


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return value.strip("_") or "experiment"


def describe_results_path(entry: dict[str, Any], results_subdir: str) -> Path:
    selected = Path(entry["selected_eval_results"])
    if selected.parent.name == results_subdir:
        return selected
    return selected.parent / results_subdir / "eval_results.jsonl"


def canonical_gt_objects(
    row: dict[str, Any],
    mscoco_objects: list[str],
    inverse_synonym_dict: dict[str, str],
    double_word_dict: dict[str, str],
) -> set[str]:
    objects = {
        inverse_synonym_dict.get(str(obj).lower(), str(obj).lower())
        for obj in row.get("gt_objects", [])
    }
    for caption in row.get("gt_captions", []) or []:
        _, nodes = caption_to_words(
            caption, mscoco_objects, inverse_synonym_dict, double_word_dict
        )
        objects.update(nodes)
    return objects


def caption_stats(
    caption: str,
    gt_objects: set[str],
    mscoco_objects: list[str],
    inverse_synonym_dict: dict[str, str],
    double_word_dict: dict[str, str],
) -> dict[str, Any]:
    surface_words, node_words = caption_to_words(
        caption, mscoco_objects, inverse_synonym_dict, double_word_dict
    )
    mentions = [
        {
            "surface": surface,
            "canonical": canonical,
            "grounded": canonical in gt_objects,
        }
        for surface, canonical in zip(surface_words, node_words)
    ]
    mentioned = set(node_words)
    correct = mentioned & gt_objects
    hallucinated = mentioned - gt_objects
    precision = len(correct) / len(mentioned) if mentioned else 0.0
    recall = len(correct) / len(gt_objects) if gt_objects else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    hallucinated_mentions = sum(not item["grounded"] for item in mentions)
    correct_mentions = len(mentions) - hallucinated_mentions
    chair_i = hallucinated_mentions / len(mentions) if mentions else 0.0
    return {
        "word_count": len(re.findall(r"[A-Za-z0-9]+", caption)),
        "mentions": mentions,
        "mentioned_objects": sorted(mentioned),
        "correct_objects": sorted(correct),
        "hallucinated_objects": sorted(hallucinated),
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": f1,
        "chair_i": chair_i,
        "chair_s": int(hallucinated_mentions > 0),
        "has_hallucination": bool(hallucinated_mentions),
        "object_mention_count": len(mentions),
        "correct_mention_count": correct_mentions,
        "hallucinated_mention_count": hallucinated_mentions,
    }


def official_case_score(
    base_stats: dict[str, Any],
    model_stats: dict[str, Any],
) -> float:
    """Diagnostic score composed only of official CHAIR quantities."""
    chair_i_reduction = base_stats["chair_i"] - model_stats["chair_i"]
    hallucinated_mentions_removed = (
        base_stats["hallucinated_mention_count"]
        - model_stats["hallucinated_mention_count"]
    )
    clean_flip = int(base_stats["chair_s"] == 1 and model_stats["chair_s"] == 0)
    return (
        100.0 * chair_i_reduction
        + 5.0 * hallucinated_mentions_removed
        + 10.0 * clean_flip
    )


def tier_sort_key(case: dict[str, Any]) -> tuple[Any, ...]:
    """Return a tier-aware ranking key.

    Tier membership always dominates. Ranking inside a tier uses only official
    CHAIRi and hallucinated-mention quantities.
    """
    return (
        TIER_PRIORITY.get(case["primary_tier"], 0),
        -case["chair_i_delta"],
        -case["hallucinated_mention_delta"],
        case["base"]["chair_i"],
        case["official_candidate_score"],
    )


def build_case(
    experiment: dict[str, Any],
    base_entry: dict[str, Any],
    image_id: str,
    model_row: dict[str, Any],
    base_row: dict[str, Any],
    mscoco_objects: list[str],
    inverse_synonym_dict: dict[str, str],
    double_word_dict: dict[str, str],
    min_correct_mention_retention: float,
    min_word_retention: float,
) -> dict[str, Any]:
    gt_objects = canonical_gt_objects(
        base_row, mscoco_objects, inverse_synonym_dict, double_word_dict
    )
    model_caption = model_row.get(
        "generated_caption", model_row.get("model_answer", "")
    )
    base_caption = base_row.get("generated_caption", base_row.get("model_answer", ""))
    base_stats = caption_stats(
        base_caption,
        gt_objects,
        mscoco_objects,
        inverse_synonym_dict,
        double_word_dict,
    )
    model_stats = caption_stats(
        model_caption,
        gt_objects,
        mscoco_objects,
        inverse_synonym_dict,
        double_word_dict,
    )

    base_hall = set(base_stats["hallucinated_objects"])
    model_hall = set(model_stats["hallucinated_objects"])
    base_correct = set(base_stats["correct_objects"])
    model_correct = set(model_stats["correct_objects"])
    removed_hall = base_hall - model_hall
    new_hall = model_hall - base_hall
    lost_correct = base_correct - model_correct
    gained_correct = model_correct - base_correct
    object_f1_delta = model_stats["object_f1"] - base_stats["object_f1"]
    chair_i_delta = model_stats["chair_i"] - base_stats["chair_i"]
    hallucinated_mention_delta = (
        model_stats["hallucinated_mention_count"]
        - base_stats["hallucinated_mention_count"]
    )
    correct_mention_delta = (
        model_stats["correct_mention_count"] - base_stats["correct_mention_count"]
    )
    correct_mention_retention = (
        model_stats["correct_mention_count"] / base_stats["correct_mention_count"]
        if base_stats["correct_mention_count"] > 0
        else 1.0
    )
    word_retention = (
        model_stats["word_count"] / base_stats["word_count"]
        if base_stats["word_count"] > 0
        else 1.0
    )
    quality_guards_passed = (
        correct_mention_retention >= min_correct_mention_retention
        and word_retention >= min_word_retention
    )

    tags: list[str] = []
    if chair_i_delta < 0:
        tags.append("official_chair_i_improved")
    if hallucinated_mention_delta < 0:
        tags.append("official_hallucinated_mentions_reduced")
    if quality_guards_passed:
        tags.append("quality_guards_passed")

    # Assign one mutually exclusive tier from official CHAIR quantities only.
    # Correct-mention and word retention are admission guards, not rank signals.
    primary_tier = None
    if quality_guards_passed and (
        base_stats["chair_s"] == 1
        and model_stats["chair_s"] == 0
    ):
        primary_tier = TIER_A
    elif quality_guards_passed and (
        base_stats["chair_s"] == 1
        and model_stats["chair_s"] == 1
        and chair_i_delta < 0
        and hallucinated_mention_delta < 0
    ):
        primary_tier = TIER_B
    elif quality_guards_passed and (
        model_stats["chair_s"] <= base_stats["chair_s"]
        and chair_i_delta < 0
        and hallucinated_mention_delta == 0
    ):
        primary_tier = TIER_C
    if primary_tier:
        tags.append(primary_tier)

    image_path = (
        model_row.get("image_path")
        or base_row.get("image_path")
        or (model_row.get("images") or base_row.get("images") or [""])[0]
    )
    score = official_case_score(base_stats, model_stats)
    return {
        "experiment_label": experiment["label"],
        "experiment_aliases": experiment.get("aliases", [experiment["label"]]),
        "model_size": experiment["model_size"],
        "experiment_results": str(experiment["results_path"]),
        "base_label": base_entry["label"],
        "base_results": str(base_entry["results_path"]),
        "image_id": image_id,
        "image_path": image_path,
        "file_name": model_row.get("file_name", base_row.get("file_name")),
        "eval_prompt": model_row.get(
            "eval_prompt", model_row.get("query", "")
        ),
        "gt_objects": sorted(gt_objects),
        "gt_captions": base_row.get("gt_captions", []),
        "base_caption": base_caption,
        "model_caption": model_caption,
        "base": base_stats,
        "model": model_stats,
        "removed_hallucinated_objects": sorted(removed_hall),
        "new_hallucinated_objects": sorted(new_hall),
        "lost_correct_objects": sorted(lost_correct),
        "gained_correct_objects": sorted(gained_correct),
        "base_chair_s": base_stats["chair_s"],
        "model_chair_s": model_stats["chair_s"],
        "chair_s_transition": f"{base_stats['chair_s']}->{model_stats['chair_s']}",
        "object_f1_delta": object_f1_delta,
        "chair_i_delta": chair_i_delta,
        "correct_object_delta": len(model_correct) - len(base_correct),
        "hallucinated_object_delta": len(model_hall) - len(base_hall),
        "correct_mention_delta": correct_mention_delta,
        "hallucinated_mention_delta": hallucinated_mention_delta,
        "correct_mention_retention": correct_mention_retention,
        "word_retention": word_retention,
        "quality_guards": {
            "passed": quality_guards_passed,
            "min_correct_mention_retention": min_correct_mention_retention,
            "min_word_retention": min_word_retention,
        },
        "candidate_tags": tags,
        "primary_tier": primary_tier,
        "official_candidate_score": score,
    }


def prepare_entries(
    preflight: dict[str, Any],
    results_subdir: str,
    allow_missing: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    missing: list[dict[str, Any]] = []
    deduped: dict[tuple[str, str], dict[str, Any]] = {}

    for raw_entry in preflight.get("entries", []):
        label = raw_entry.get("label", "")
        model_size = model_size_from_label(label)
        results_path = describe_results_path(raw_entry, results_subdir)
        if not results_path.is_file():
            missing.append({"label": label, "expected_path": str(results_path)})
            continue
        key = (model_size, str(results_path.resolve()))
        if key not in deduped:
            deduped[key] = {
                **raw_entry,
                "model_size": model_size,
                "results_path": results_path,
                "aliases": [label],
            }
        elif label not in deduped[key]["aliases"]:
            deduped[key]["aliases"].append(label)

    if missing and not allow_missing:
        lines = "\n".join(
            f"  - {item['label']}: {item['expected_path']}" for item in missing
        )
        raise FileNotFoundError(
            f"{len(missing)} prompt-isolated result files are missing:\n{lines}"
        )
    return list(deduped.values()), missing


def select_diverse(
    candidates: list[dict[str, Any]],
    limit: int,
    max_per_experiment: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    experiment_counts: defaultdict[str, int] = defaultdict(int)
    for case in candidates:
        label = case["experiment_label"]
        if case["image_id"] in used_images:
            continue
        if experiment_counts[label] >= max_per_experiment:
            continue
        selected.append(case)
        used_images.add(case["image_id"])
        experiment_counts[label] += 1
        if len(selected) >= limit:
            break
    return selected


def markdown_caption(text: str, limit: int = 260) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def write_shortlist_markdown(
    path: Path,
    candidates: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    total_experiments: int,
) -> None:
    lines = [
        "# Qualitative CHAIR candidate shortlist",
        "",
        f"- Unique evaluated experiments: {total_experiments}",
        f"- Diverse candidates: {len(candidates)}",
        "- Tier A: official per-caption CHAIRs 1->0.",
        "- Tier B: CHAIRs remains 1, while official CHAIRi and "
        "hallucinated mentions both decrease.",
        "- Tier C: CHAIRs does not worsen and official CHAIRi decreases "
        "with unchanged hallucinated mentions.",
        "- Correct-mention and caption-length retention are admission guards only.",
        "- Ranking inside each tier uses official CHAIRi and hallucinated mentions only.",
        "- Every candidate must still be checked against the image; COCO annotations are incomplete.",
        "",
    ]
    if missing:
        lines.extend(
            [
                f"## Missing results ({len(missing)})",
                "",
                *[
                    f"- `{item['label']}`: `{item['expected_path']}`"
                    for item in missing
                ],
                "",
            ]
        )

    for index, case in enumerate(candidates, start=1):
        lines.extend(
            [
                f"## {index}. {case['experiment_label']} / image {case['image_id']}",
                "",
                f"- Primary tier: `{case['primary_tier']}`",
                f"- Official CHAIRs transition: `{case['chair_s_transition']}`",
                f"- Official CHAIRi: `{case['base']['chair_i']:.4f} -> "
                f"{case['model']['chair_i']:.4f}` "
                f"(`{case['chair_i_delta']:+.4f}`)",
                f"- Hallucinated mentions: "
                f"`{case['base']['hallucinated_mention_count']} -> "
                f"{case['model']['hallucinated_mention_count']}`",
                f"- Correct-mention retention: `{case['correct_mention_retention']:.3f}`",
                f"- Word retention: `{case['word_retention']:.3f}`",
                f"- Official score: `{case['official_candidate_score']:.3f}`",
                f"- Tags: `{', '.join(case['candidate_tags'])}`",
                f"- Image: `{case['image_path']}`",
                f"- Removed hallucinations: `{case['removed_hallucinated_objects']}`",
                f"- New hallucinations: `{case['new_hallucinated_objects']}`",
                f"- Lost correct objects: `{case['lost_correct_objects']}`",
                f"- Gained correct objects: `{case['gained_correct_objects']}`",
                f"- ObjF1 delta: `{case['object_f1_delta']:+.4f}`",
                f"- Base: {markdown_caption(case['base_caption'])}",
                f"- Model: {markdown_caption(case['model_caption'])}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    preflight_path = Path(args.preflight_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight = load_json(preflight_path)
    entries, missing = prepare_entries(
        preflight, args.results_subdir, args.allow_missing
    )
    base_entries: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry["label"].lower() == f"{entry['model_size']}/base/original":
            base_entries[entry["model_size"]] = entry

    required_sizes = sorted({entry["model_size"] for entry in entries})
    absent_bases = [size for size in required_sizes if size not in base_entries]
    if absent_bases:
        raise ValueError(f"Missing original-image base entries for: {absent_bases}")

    mscoco_objects, inverse_synonym_dict = parse_official_synonyms()
    double_word_dict = build_double_word_dict()
    records_cache: dict[str, dict[str, dict[str, Any]]] = {}

    def records_for(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
        key = str(entry["results_path"])
        if key not in records_cache:
            records_cache[key] = read_eval_results(
                entry["results_path"], args.expected_prompt
            )
        return records_cache[key]

    all_cases: list[dict[str, Any]] = []
    experiment_summaries: list[dict[str, Any]] = []
    compared_experiments = [
        entry
        for entry in entries
        if entry["label"].lower() != f"{entry['model_size']}/base/original"
    ]

    for experiment in compared_experiments:
        base_entry = base_entries[experiment["model_size"]]
        model_records = records_for(experiment)
        base_records = records_for(base_entry)
        common_ids = sorted(
            set(model_records) & set(base_records),
            key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
        )
        if len(common_ids) < args.min_common_samples:
            raise ValueError(
                f"{experiment['label']}: only {len(common_ids)} samples overlap "
                f"with {base_entry['label']}"
            )

        experiment_cases = [
            build_case(
                experiment,
                base_entry,
                image_id,
                model_records[image_id],
                base_records[image_id],
                mscoco_objects,
                inverse_synonym_dict,
                double_word_dict,
                args.min_correct_mention_retention,
                args.min_word_retention,
            )
            for image_id in common_ids
        ]
        experiment_cases.sort(key=tier_sort_key, reverse=True)
        all_cases.extend(experiment_cases)

        tier_a_count = sum(case["primary_tier"] == TIER_A for case in experiment_cases)
        tier_b_count = sum(case["primary_tier"] == TIER_B for case in experiment_cases)
        tier_c_count = sum(case["primary_tier"] == TIER_C for case in experiment_cases)
        experiment_summaries.append(
            {
                "label": experiment["label"],
                "aliases": experiment["aliases"],
                "model_size": experiment["model_size"],
                "results_path": str(experiment["results_path"]),
                "base_label": base_entry["label"],
                "aligned_samples": len(common_ids),
                "tier_a_official_chairs_clean_flip_count": tier_a_count,
                "tier_b_official_chairi_mention_reduction_count": tier_b_count,
                "tier_c_official_chairi_ratio_improvement_count": tier_c_count,
                "mean_official_chairi_delta": sum(
                    case["chair_i_delta"] for case in experiment_cases
                )
                / len(experiment_cases),
                "mean_hallucinated_mention_delta": sum(
                    case["hallucinated_mention_delta"] for case in experiment_cases
                )
                / len(experiment_cases),
            }
        )

    all_cases.sort(key=tier_sort_key, reverse=True)
    tier_a_cases = [case for case in all_cases if case["primary_tier"] == TIER_A]
    tier_b_cases = [case for case in all_cases if case["primary_tier"] == TIER_B]
    tier_c_cases = [case for case in all_cases if case["primary_tier"] == TIER_C]
    tiered_cases = tier_a_cases + tier_b_cases + tier_c_cases

    top_by_experiment: list[dict[str, Any]] = []
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in tiered_cases:
        grouped[case["experiment_label"]].append(case)
    for label in sorted(grouped):
        grouped[label].sort(key=tier_sort_key, reverse=True)
        top_by_experiment.extend(grouped[label][: args.top_per_experiment])

    diverse = select_diverse(
        tiered_cases,
        args.global_limit,
        args.max_per_experiment,
    )

    write_jsonl(
        output_dir / "tier_a_official_chairs_clean_flip.jsonl",
        tier_a_cases[: args.tier_limit],
    )
    write_jsonl(
        output_dir / "tier_b_official_chairi_mention_reduction.jsonl",
        tier_b_cases[: args.tier_limit],
    )
    write_jsonl(
        output_dir / "tier_c_official_chairi_ratio_improvement.jsonl",
        tier_c_cases[: args.tier_limit],
    )
    write_jsonl(output_dir / "top_cases_by_experiment.jsonl", top_by_experiment)
    write_jsonl(output_dir / "global_diverse_shortlist.jsonl", diverse)
    write_json(
        output_dir / "selection_summary.json",
        {
            "preflight_json": str(preflight_path),
            "results_subdir": args.results_subdir,
            "expected_prompt": args.expected_prompt,
            "quality_guards": {
                "min_correct_mention_retention": args.min_correct_mention_retention,
                "min_word_retention": args.min_word_retention,
            },
            "unique_result_entries": len(entries),
            "compared_experiments": len(compared_experiments),
            "missing_results": missing,
            "all_case_pairs": len(all_cases),
            "selection_policy": {
                "tier_a": (
                    "official per-caption CHAIRs 1->0"
                ),
                "tier_b": (
                    "CHAIRs remains 1, official per-caption CHAIRi decreases, "
                    "and official hallucinated-mention count decreases"
                ),
                "tier_c": (
                    "CHAIRs does not worsen, official per-caption CHAIRi decreases, "
                    "and official hallucinated-mention count is unchanged"
                ),
                "within_tier_tiebreakers": [
                    "official CHAIRi delta",
                    "official hallucinated-mention delta",
                    "Base official CHAIRi",
                ],
                "note": (
                    "Unique categories and ObjF1 are serialized for human "
                    "interpretation but do not determine tier or rank."
                ),
            },
            "tier_a_candidates": len(tier_a_cases),
            "tier_b_candidates": len(tier_b_cases),
            "tier_c_candidates": len(tier_c_cases),
            "diverse_shortlist": len(diverse),
            "experiment_summaries": sorted(
                experiment_summaries,
                key=lambda row: (
                    row["tier_a_official_chairs_clean_flip_count"],
                    row["tier_b_official_chairi_mention_reduction_count"],
                    row["tier_c_official_chairi_ratio_improvement_count"],
                ),
                reverse=True,
            ),
        },
    )
    write_shortlist_markdown(
        output_dir / "global_diverse_shortlist.md",
        diverse,
        missing,
        len(compared_experiments),
    )

    print(f"Compared experiments: {len(compared_experiments)}")
    print(f"All aligned case pairs: {len(all_cases)}")
    print(f"Tier A candidates: {len(tier_a_cases)}")
    print(f"Tier B candidates: {len(tier_b_cases)}")
    print(f"Tier C candidates: {len(tier_c_cases)}")
    print(f"Diverse shortlist: {len(diverse)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
