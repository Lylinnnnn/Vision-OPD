#!/usr/bin/env python3
"""Select and package a broad image pool for qualitative manual review."""

from __future__ import annotations

import argparse
import json
import math
import re
import tarfile
from pathlib import Path
from typing import Any


IMAGE_ID_RE = re.compile(r"(?<!\d)(\d{1,12})(?=\.(?:jpg|jpeg|png)$)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-archive", type=Path, action="append", default=[])
    parser.add_argument("--exclude-image-dir", type=Path, action="append", default=[])
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--min-correct-retention", type=float, default=0.90)
    parser.add_argument("--min-word-retention", type=float, default=0.65)
    parser.add_argument("--max-word-retention", type=float, default=1.60)
    parser.add_argument("--max-model-words", type=int, default=800)
    parser.add_argument(
        "--package-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create review_images.tar.gz from each selected record's image_path.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Package available images instead of failing when any image is missing.",
    )
    return parser.parse_args()


def normalized_image_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("empty image id")
    return str(int(text))


def image_id_from_name(name: str) -> str | None:
    match = IMAGE_ID_RE.search(Path(name).name)
    return normalized_image_id(match.group(1)) if match else None


def load_excluded_ids(args: argparse.Namespace) -> set[str]:
    excluded: set[str] = set()
    for archive in args.exclude_archive:
        if not archive.is_file():
            raise FileNotFoundError(f"exclude archive not found: {archive}")
        with tarfile.open(archive, "r:*") as handle:
            for member in handle.getmembers():
                image_id = image_id_from_name(member.name)
                if image_id is not None:
                    excluded.add(image_id)

    for image_dir in args.exclude_image_dir:
        if not image_dir.is_dir():
            raise FileNotFoundError(f"exclude image directory not found: {image_dir}")
        for path in image_dir.iterdir():
            image_id = image_id_from_name(path.name)
            if image_id is not None:
                excluded.add(image_id)
    return excluded


def candidate_passes(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if int(row.get("base_chair_s", row.get("base", {}).get("chair_s", 0))) != 1:
        return False
    if int(row.get("model_chair_s", row.get("model", {}).get("chair_s", 1))) != 0:
        return False
    if row.get("new_hallucinated_objects"):
        return False
    if row.get("lost_correct_objects"):
        return False
    if float(row.get("correct_mention_retention", 0.0)) < args.min_correct_retention:
        return False

    word_retention = float(row.get("word_retention", 0.0))
    if not args.min_word_retention <= word_retention <= args.max_word_retention:
        return False
    if int(row.get("model", {}).get("word_count", 0)) > args.max_model_words:
        return False
    return bool(row.get("removed_hallucinated_objects"))


def candidate_score(row: dict[str, Any]) -> float:
    base = row.get("base", {})
    model = row.get("model", {})
    removed_objects = len(row.get("removed_hallucinated_objects", []))
    removed_mentions = max(
        0,
        int(base.get("hallucinated_mention_count", 0))
        - int(model.get("hallucinated_mention_count", 0)),
    )
    gained_objects = len(row.get("gained_correct_objects", []))
    correct_delta = int(row.get("correct_mention_delta", 0))
    f1_delta = float(row.get("object_f1_delta", 0.0))
    word_retention = max(float(row.get("word_retention", 1.0)), 1e-6)

    return (
        30.0 * removed_objects
        + 8.0 * removed_mentions
        + 15.0 * gained_objects
        + 2.0 * max(correct_delta, 0)
        + 20.0 * max(f1_delta, 0.0)
        - 8.0 * abs(math.log(word_retention))
    )


def load_best_candidates(
    path: Path, excluded_ids: set[str], args: argparse.Namespace
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"candidate JSONL not found: {path}")

    best_by_image: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc

            image_id = normalized_image_id(row.get("image_id", ""))
            if image_id in excluded_ids or not candidate_passes(row, args):
                continue

            row["_review_score"] = candidate_score(row)
            previous = best_by_image.get(image_id)
            if previous is None or row["_review_score"] > previous["_review_score"]:
                best_by_image[image_id] = row

    selected = sorted(
        best_by_image.values(),
        key=lambda row: (-row["_review_score"], int(normalized_image_id(row["image_id"]))),
    )
    return selected[: args.max_images]


def write_outputs(
    selected: list[dict[str, Any]], excluded_ids: set[str], args: argparse.Namespace
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "review_manifest.jsonl"
    ids_path = args.output_dir / "image_ids.txt"
    summary_path = args.output_dir / "review_manifest.md"

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with ids_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(normalized_image_id(row["image_id"]) + "\n")

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Qualitative review batch\n\n")
        handle.write(f"- Previously reviewed images excluded: {len(excluded_ids)}\n")
        handle.write(f"- Selected unique images: {len(selected)}\n")
        handle.write(
            "- Filter: CHAIRs 1->0, no new hallucinated object, no lost correct "
            "object, correct-mention retention and length guards passed.\n\n"
        )
        handle.write(
            "| Rank | Image ID | Size | Experiment | Prompt | Removed | Gained | Score |\n"
        )
        handle.write("| ---: | ---: | --- | --- | --- | --- | --- | ---: |\n")
        for rank, row in enumerate(selected, start=1):
            removed = ", ".join(row.get("removed_hallucinated_objects", [])) or "-"
            gained = ", ".join(row.get("gained_correct_objects", [])) or "-"
            handle.write(
                f"| {rank} | {normalized_image_id(row['image_id'])} | "
                f"{row.get('model_size', '-')} | {row.get('experiment_label', '-')} | "
                f"{row.get('prompt_variant', '-')} | {removed} | {gained} | "
                f"{row['_review_score']:.2f} |\n"
            )

    if not args.package_images:
        return

    missing: list[str] = []
    archive_path = args.output_dir / "review_images.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(manifest_path, arcname="_metadata/review_manifest.jsonl")
        archive.add(ids_path, arcname="_metadata/image_ids.txt")
        archive.add(summary_path, arcname="_metadata/review_manifest.md")
        for row in selected:
            image_path = Path(str(row.get("image_path", "")))
            if not image_path.is_file():
                missing.append(str(image_path))
                continue
            archive.add(image_path, arcname=row.get("file_name", image_path.name))

    if missing and not args.allow_missing_images:
        archive_path.unlink(missing_ok=True)
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        suffix = "\n  ..." if len(missing) > 20 else ""
        raise FileNotFoundError(
            f"{len(missing)} selected images are missing; archive was removed:\n"
            f"{preview}{suffix}"
        )
    if missing:
        print(f"WARNING: skipped {len(missing)} missing images")


def main() -> None:
    args = parse_args()
    excluded_ids = load_excluded_ids(args)
    selected = load_best_candidates(args.candidates, excluded_ids, args)
    if not selected:
        raise RuntimeError("no candidates passed the review-batch filters")
    write_outputs(selected, excluded_ids, args)
    print(f"Excluded previously reviewed images: {len(excluded_ids)}")
    print(f"Selected unique images: {len(selected)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
