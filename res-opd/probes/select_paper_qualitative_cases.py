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
        "--strict-limit",
        type=int,
        default=1000,
        help="Maximum strict Pareto candidates written to JSONL.",
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
        "--allow-missing",
        action="store_true",
        help="Warn and skip missing result files instead of failing.",
    )
    return parser.parse_args()


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
        "has_hallucination": bool(hallucinated_mentions),
        "object_mention_count": len(mentions),
        "hallucinated_mention_count": hallucinated_mentions,
    }


def case_score(
    removed_hallucinations: set[str],
    new_hallucinations: set[str],
    lost_correct: set[str],
    gained_correct: set[str],
    base_stats: dict[str, Any],
    model_stats: dict[str, Any],
) -> float:
    score = 12.0 * len(removed_hallucinations)
    score -= 15.0 * len(new_hallucinations)
    score -= 7.0 * len(lost_correct)
    score += 5.0 * len(gained_correct)
    score += 10.0 * (model_stats["object_f1"] - base_stats["object_f1"])
    score += 4.0 * (
        base_stats["hallucinated_mention_count"]
        - model_stats["hallucinated_mention_count"]
    )
    if base_stats["has_hallucination"] and not model_stats["has_hallucination"]:
        score += 6.0

    # Avoid ranking a nearly empty caption as a strong qualitative success.
    base_words = max(base_stats["word_count"], 1)
    if model_stats["word_count"] / base_words < 0.55:
        score -= 8.0
    return score


def build_case(
    experiment: dict[str, Any],
    base_entry: dict[str, Any],
    image_id: str,
    model_row: dict[str, Any],
    base_row: dict[str, Any],
    mscoco_objects: list[str],
    inverse_synonym_dict: dict[str, str],
    double_word_dict: dict[str, str],
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

    tags: list[str] = []
    if removed_hall and not new_hall:
        tags.append("hallucination_reduced")
    if gained_correct and len(model_hall) <= len(base_hall):
        tags.append("grounded_coverage_improved")
    if removed_hall and not new_hall and not lost_correct:
        tags.append("strict_pareto_improvement")
    if (
        model_stats["object_f1"] > base_stats["object_f1"]
        and len(model_hall) <= len(base_hall)
    ):
        tags.append("balanced_f1_improvement")

    image_path = (
        model_row.get("image_path")
        or base_row.get("image_path")
        or (model_row.get("images") or base_row.get("images") or [""])[0]
    )
    score = case_score(
        removed_hall, new_hall, lost_correct, gained_correct, base_stats, model_stats
    )
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
        "object_f1_delta": model_stats["object_f1"] - base_stats["object_f1"],
        "chair_i_delta": model_stats["chair_i"] - base_stats["chair_i"],
        "correct_object_delta": len(model_correct) - len(base_correct),
        "hallucinated_object_delta": len(model_hall) - len(base_hall),
        "hallucinated_mention_delta": (
            model_stats["hallucinated_mention_count"]
            - base_stats["hallucinated_mention_count"]
        ),
        "candidate_tags": tags,
        "candidate_score": score,
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
        "- Ranking uses official CHAIR object parsing and same-size Base comparisons.",
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
                f"- Score: `{case['candidate_score']:.3f}`",
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
            )
            for image_id in common_ids
        ]
        experiment_cases.sort(
            key=lambda case: (
                case["candidate_score"],
                case["object_f1_delta"],
                -case["hallucinated_mention_delta"],
            ),
            reverse=True,
        )
        all_cases.extend(experiment_cases)

        strict_count = sum(
            "strict_pareto_improvement" in case["candidate_tags"]
            for case in experiment_cases
        )
        reduced_count = sum(
            "hallucination_reduced" in case["candidate_tags"]
            for case in experiment_cases
        )
        balanced_count = sum(
            "balanced_f1_improvement" in case["candidate_tags"]
            for case in experiment_cases
        )
        experiment_summaries.append(
            {
                "label": experiment["label"],
                "aliases": experiment["aliases"],
                "model_size": experiment["model_size"],
                "results_path": str(experiment["results_path"]),
                "base_label": base_entry["label"],
                "aligned_samples": len(common_ids),
                "strict_pareto_improvement_count": strict_count,
                "hallucination_reduced_count": reduced_count,
                "balanced_f1_improvement_count": balanced_count,
                "mean_object_f1_delta": sum(
                    case["object_f1_delta"] for case in experiment_cases
                )
                / len(experiment_cases),
                "mean_hallucinated_mention_delta": sum(
                    case["hallucinated_mention_delta"] for case in experiment_cases
                )
                / len(experiment_cases),
            }
        )

    all_cases.sort(
        key=lambda case: (
            case["candidate_score"],
            case["object_f1_delta"],
            -case["hallucinated_mention_delta"],
        ),
        reverse=True,
    )
    strict_cases = [
        case for case in all_cases if "strict_pareto_improvement" in case["candidate_tags"]
    ]
    balanced_cases = [
        case
        for case in all_cases
        if case["candidate_tags"]
        and (
            "hallucination_reduced" in case["candidate_tags"]
            or "grounded_coverage_improved" in case["candidate_tags"]
            or "balanced_f1_improvement" in case["candidate_tags"]
        )
    ]

    top_by_experiment: list[dict[str, Any]] = []
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in balanced_cases:
        grouped[case["experiment_label"]].append(case)
    for label in sorted(grouped):
        top_by_experiment.extend(grouped[label][: args.top_per_experiment])

    diverse = select_diverse(
        strict_cases + [case for case in balanced_cases if case not in strict_cases],
        args.global_limit,
        args.max_per_experiment,
    )

    write_jsonl(
        output_dir / "strict_good_candidates.jsonl",
        strict_cases[: args.strict_limit],
    )
    write_jsonl(output_dir / "top_cases_by_experiment.jsonl", top_by_experiment)
    write_jsonl(output_dir / "global_diverse_shortlist.jsonl", diverse)
    write_json(
        output_dir / "selection_summary.json",
        {
            "preflight_json": str(preflight_path),
            "results_subdir": args.results_subdir,
            "expected_prompt": args.expected_prompt,
            "unique_result_entries": len(entries),
            "compared_experiments": len(compared_experiments),
            "missing_results": missing,
            "all_case_pairs": len(all_cases),
            "strict_good_candidates": len(strict_cases),
            "balanced_candidates": len(balanced_cases),
            "diverse_shortlist": len(diverse),
            "experiment_summaries": sorted(
                experiment_summaries,
                key=lambda row: (
                    row["strict_pareto_improvement_count"],
                    row["balanced_f1_improvement_count"],
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
    print(f"Strict good candidates: {len(strict_cases)}")
    print(f"Diverse shortlist: {len(diverse)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
