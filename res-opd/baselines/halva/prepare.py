#!/usr/bin/env python3
"""Build matched COCO-5k phrase pairs for the HALVA adaptation."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_annotations(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("annotations"), list):
        raise ValueError(f"Invalid COCO annotations: {path}")
    return data


def choose_distractor(image_id, name, present, caption, categories):
    absent = sorted(
        candidate for candidate in categories
        if candidate not in present
        and not re.search(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)", caption, re.I)
    )
    alternatives = [candidate for candidate in absent if categories[candidate] == categories[name]]
    if not alternatives:
        alternatives = absent
    if not alternatives:
        return None
    digest = hashlib.sha256(f"{image_id}:{name}".encode()).digest()
    return alternatives[int.from_bytes(digest[:4], "big") % len(alternatives)]


def pair_record(question, caption, wrong, correct_phrase, rejected_phrase, strategy):
    return {
        "image_id": int(question["image_id"]),
        "image_path": question["image_path"],
        "prompt": question["question"],
        "chosen": caption,
        "rejected": wrong,
        "chosen_phrase": correct_phrase,
        "rejected_phrase": rejected_phrase,
        "source": "COCO train2017 captions and instances",
        "strategy": strategy,
    }


def choose_pair(question, captions, objects, categories):
    image_id = int(question["image_id"])
    present = objects.get(image_id, set())
    image_captions = captions.get(image_id, [])
    for caption in image_captions:
        for name in sorted(present, key=lambda value: (-len(value), value)):
            matches = list(re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", caption, re.I))
            if len(matches) != 1:
                continue
            distractor = choose_distractor(image_id, name, present, caption, categories)
            if distractor is None:
                continue
            match = matches[0]
            correct_phrase = caption[match.start():match.end()]
            wrong = caption[:match.start()] + distractor + caption[match.end():]
            return pair_record(question, caption, wrong, correct_phrase, distractor, "caption_replacement")

    if not present or not image_captions:
        return None
    caption = image_captions[0]
    for name in sorted(present, key=lambda value: (-len(value), value)):
        distractor = choose_distractor(image_id, name, present, caption, categories)
        if distractor is None:
            continue
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", caption, re.I):
            chosen = f"Visible object: {name}."
            rejected = f"Visible object: {distractor}."
            strategy = "annotated_object_clause_only"
        else:
            chosen = f"{caption.rstrip()} Visible object: {name}."
            rejected = f"{caption.rstrip()} Visible object: {distractor}."
            strategy = "annotated_object_clause_added"
        return pair_record(question, chosen, rejected, name, distractor, strategy)
    return None


def build(questions, captions_data, instances_data, max_samples=0, expected_pairs=0):
    target = max_samples or len(questions)
    if expected_pairs < 0 or expected_pairs > target or (max_samples and expected_pairs not in (0, max_samples)):
        raise ValueError("Expected pair count must be within the requested sample count")
    captions = defaultdict(list)
    for item in captions_data["annotations"]:
        captions[int(item["image_id"])].append(item["caption"].strip())
    category_by_id = {int(item["id"]): item for item in instances_data["categories"]}
    categories = {item["name"]: item["supercategory"] for item in category_by_id.values()}
    objects = defaultdict(set)
    for item in instances_data["annotations"]:
        objects[int(item["image_id"])].add(category_by_id[int(item["category_id"])]["name"])

    pairs = []
    seen = set()
    for question in questions:
        image_id = int(question["image_id"])
        if image_id in seen:
            raise ValueError(f"Duplicate training image: {image_id}")
        seen.add(image_id)
        if not Path(question["image_path"]).is_file():
            raise FileNotFoundError(question["image_path"])
        pair = choose_pair(question, captions, objects, categories)
        if pair is not None:
            pairs.append(pair)
            if max_samples and len(pairs) >= max_samples:
                break
    if len(pairs) != (expected_pairs or target) or (expected_pairs and expected_pairs < target):
        matched_ids = {pair["image_id"] for pair in pairs}
        unmatched = [int(row["image_id"]) for row in questions if int(row["image_id"]) not in matched_ids]
        no_instances = sum(not objects.get(image_id) for image_id in unmatched)
        if len(pairs) != (expected_pairs or target) or len(unmatched) != no_instances:
            raise ValueError(
                f"Only {len(pairs)}/{target} images yielded a valid phrase pair; "
                f"expected_pairs={expected_pairs or target}, unmatched={len(unmatched)}, "
                f"without_instances={no_instances}, example_ids={unmatched[:10]}"
            )
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--expected-questions", type=int, default=0)
    parser.add_argument("--expected-pairs", type=int, default=0)
    args = parser.parse_args()
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line]
    if args.expected_questions and len(questions) != args.expected_questions:
        raise ValueError(f"Expected {args.expected_questions} questions, got {len(questions)}")
    pairs = build(
        questions, load_annotations(args.captions), load_annotations(args.instances),
        args.max_samples, args.expected_pairs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
    strategies = dict(Counter(pair["strategy"] for pair in pairs))
    print(
        f"HALVA_PAIRS=PASS count={len(pairs)} requested={args.max_samples or len(questions)} "
        f"skipped={len(questions) - len(pairs) if not args.max_samples else 0} "
        f"strategies={strategies} output={args.output}"
    )


if __name__ == "__main__":
    main()
