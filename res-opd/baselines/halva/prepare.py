#!/usr/bin/env python3
"""Build matched COCO-5k phrase pairs for the HALVA adaptation."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def load_annotations(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("annotations"), list):
        raise ValueError(f"Invalid COCO annotations: {path}")
    return data


def choose_pair(question, captions, objects, categories):
    image_id = int(question["image_id"])
    present = objects.get(image_id, set())
    for caption in captions.get(image_id, []):
        for name in sorted(present, key=lambda value: (-len(value), value)):
            matches = list(re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", caption, re.I))
            if len(matches) != 1:
                continue
            absent = sorted(
                candidate for candidate in categories
                if candidate not in present
                and not re.search(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)", caption, re.I)
            )
            alternatives = [candidate for candidate in absent if categories[candidate] == categories[name]]
            if not alternatives:
                alternatives = absent
            if not alternatives:
                continue
            digest = hashlib.sha256(f"{image_id}:{name}".encode()).digest()
            distractor = alternatives[int.from_bytes(digest[:4], "big") % len(alternatives)]
            match = matches[0]
            correct_phrase = caption[match.start():match.end()]
            wrong = caption[:match.start()] + distractor + caption[match.end():]
            return {
                "image_id": image_id,
                "image_path": question["image_path"],
                "prompt": question["question"],
                "chosen": caption,
                "rejected": wrong,
                "chosen_phrase": correct_phrase,
                "rejected_phrase": distractor,
                "source": "COCO train2017 captions and instances; same-supercategory substitution",
            }
    return None


def build(questions, captions_data, instances_data, max_samples=0):
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
    if len(pairs) < (max_samples or len(questions)):
        raise ValueError(f"Only {len(pairs)}/{max_samples or len(questions)} images yielded a valid phrase pair")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line]
    pairs = build(questions, load_annotations(args.captions), load_annotations(args.instances), args.max_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"HALVA_PAIRS=PASS count={len(pairs)} output={args.output}")


if __name__ == "__main__":
    main()
