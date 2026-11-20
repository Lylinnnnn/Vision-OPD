#!/usr/bin/env python3
"""Build APASI-Base's co-occurrence graph and single-iteration injection plans."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re


TEMPLATES = (
    "There is a {}",
    "There are {}",
    "{} can also be seen",
    "{} can be seen",
    "You can see a {}",
    "There are multiple {}",
    "Several {} can be observed",
    "Some {} are present",
    "Among the items, there is a {}",
    "In the image, there is a {}",
    "On the right, there is a {}",
    "On the left, a {} is present",
    "In the center, you see a {}",
    "At the top, there is a {}",
    "At the bottom, a {} is visible",
    "In the background, a {} can be seen",
    "In the foreground, there is a {}",
    "To the side, a {} is located",
    "Near the edge, a {} appears",
    "Close to the center, a {} is seen",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def split_sentences(text: str) -> list[str]:
    ends = [match.end() for match in re.finditer(r"(?<!\d)\.(?!\d)", text)]
    starts = [0, *ends]
    parts = [text[start:end].strip() for start, end in zip(starts, [*ends, len(text)])]
    return [part for part in parts if part]


def weighted_indices(length: int, count: int, rng: random.Random) -> list[int]:
    remaining = list(range(1, length))  # The first sentence is never replaced.
    chosen = []
    for _ in range(count):
        if length > 2:
            scale = math.sqrt(length - 1) - 1
            weights = [(math.sqrt(index) - 1) / scale for index in remaining]
        else:
            weights = [1] * len(remaining)
        if not any(weights):
            weights = [1] * len(remaining)
        selected = rng.choices(remaining, weights=weights, k=1)[0]
        chosen.append(selected)
        remaining.remove(selected)
    return sorted(chosen)


def mentioned_in(text: str, node: str, synonyms: dict[str, list[str]]) -> bool:
    return any(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.I)
               for word in synonyms.get(node, []) if word)


def make_plan(row: dict, parsed: dict, cooccurrence: dict, synonyms: dict,
              injection_rate: float, seed: int) -> dict | None:
    sentences = split_sentences(row["caption"])
    count = round(injection_rate * len(sentences))
    if count < 1 or count > len(sentences) - 1:
        return None
    rng = random.Random(seed + int(row["image_id"]))
    selected = weighted_indices(len(sentences), count, rng)
    objects = set(parsed.get("node_objects", []))
    remaining_text = " ".join(sentence for index, sentence in enumerate(sentences)
                              if index not in selected)
    remaining = [node for node in sorted(objects) if mentioned_in(remaining_text, node, synonyms)]
    if not remaining:
        return None
    guides = {}
    for index in selected:
        rng.shuffle(remaining)
        candidates = []
        for node in remaining:
            candidates = [candidate for candidate in cooccurrence.get(node, {})
                          if candidate not in objects and synonyms.get(candidate)]
            if candidates:
                break
        if not candidates:
            candidates = [candidate for candidate in sorted(cooccurrence)
                          if candidate not in objects and synonyms.get(candidate)]
        if not candidates:
            return None
        target = rng.choice(candidates[:10])
        synonym = rng.choice(synonyms[target])
        guides[str(index)] = {"object": target, "text": rng.choice(TEMPLATES).format(synonym)}
    return {
        "image_id": row["image_id"], "image_path": row["image_path"],
        "prompt": row["prompt"], "chosen": row["caption"],
        "sentences": sentences, "guides": guides,
    }


def build_graph(preferred: list[dict], upstream: Path, work_dir: Path) -> dict:
    source = upstream / "llava/prepare_data/co_graph_construction.py"
    vocab = upstream / "playground/data/neg_data/LVIS_CATEGORIES.json"
    if not source.is_file() or not vocab.is_file():
        raise FileNotFoundError(f"APASI source or LVIS vocabulary missing: {source}, {vocab}")
    spec = importlib.util.spec_from_file_location("apasi_upstream_graph", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    caption_path = work_dir / "preferred_llava_format.json"
    processed_path = work_dir / "preferred_lvis_processed.json"
    caption_path.write_text(json.dumps([
        {"id": str(row["image_id"]), "image": row["image_path"],
         "conversations": [
             {"from": "human", "value": "<image>\n" + row["prompt"]},
             {"from": "gpt", "value": row["caption"]},
         ]}
        for row in preferred
    ], ensure_ascii=False), encoding="utf-8")
    module.ClosedSetNegConstructor(
        sym_file=str(vocab), cap_file=str(caption_path),
        processed_caption_file=str(processed_path), object_vocab_mode="lvis",
        re_process=True,
    )
    return json.loads(processed_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--preferred-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--injection-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.injection_rate < 1:
        parser.error("--injection-rate must be between zero and one")
    questions = read_jsonl(args.questions)
    expected = {str(row["image_id"]): row for row in questions}
    if len(expected) != len(questions):
        raise ValueError("Duplicate image IDs in questions")
    preferred = [row for path in sorted(args.preferred_dir.glob("shard_*.jsonl"))
                 for row in read_jsonl(path)]
    ids = [str(row["image_id"]) for row in preferred]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError(f"Preferred responses incomplete/duplicated: {len(ids)}/{len(expected)}")
    for row in preferred:
        question = expected[str(row["image_id"])]
        if row["image_path"] != question["image_path"] or row["prompt"] != question["question"]:
            raise ValueError(f"Mismatched training row: {row['image_id']}")
    preferred.sort(key=lambda row: int(row["image_id"]))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    plans_path = args.work_dir / "plans.jsonl"
    if plans_path.exists():
        raise FileExistsError(plans_path)
    graph = build_graph(preferred, args.upstream, args.work_dir)
    plans = []
    for row in preferred:
        parsed = graph["processed_captions"][str(row["image_id"])]
        plan = make_plan(row, parsed, graph["node_object_co_freq"],
                         graph["synonym_dict"], args.injection_rate, args.seed)
        if plan:
            plans.append(plan)
    if not plans:
        raise RuntimeError("No valid self-injection plans; inspect caption lengths and LVIS object tags")
    temporary = plans_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for plan in plans:
            stream.write(json.dumps(plan, ensure_ascii=False) + "\n")
    os.replace(temporary, plans_path)
    print(f"APASI_GRAPH=PASS preferred={len(preferred)} plans={len(plans)} "
          f"skipped={len(preferred) - len(plans)} rate={args.injection_rate} output={plans_path}")


if __name__ == "__main__":
    main()
