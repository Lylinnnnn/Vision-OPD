#!/usr/bin/env python3
"""Fail if an RLSF-V generation run silently skipped any requested samples."""

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(questions, answers, num_samples):
    expected = [str(row["image_id"]) for row in questions]
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate question image IDs")
    counts = Counter()
    answer_ids = set()
    for row in answers:
        qid = str(row["question_id"])
        answer_id = row["answer_id"]
        if answer_id in answer_ids:
            raise ValueError(f"Duplicate answer_id: {answer_id}")
        answer_ids.add(answer_id)
        counts[qid] += 1
        if not str(row.get("text", "")).strip():
            raise ValueError(f"Empty generated answer for {qid}")
    invalid = {qid: counts[qid] for qid in expected if counts[qid] != num_samples}
    unexpected = set(counts) - set(expected)
    if invalid or unexpected:
        raise ValueError(f"Incomplete generation: {len(invalid)} incomplete questions, "
                         f"{len(unexpected)} unexpected IDs; examples={list(invalid.items())[:5]}")
    return len(expected), len(answer_ids)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()
    if not args.logits.is_file() or args.logits.stat().st_size == 0:
        raise FileNotFoundError(args.logits)
    n_questions, n_answers = audit(
        read_jsonl(args.questions), read_jsonl(args.answers), args.num_samples
    )
    print(f"RLSF-V generation audit PASS: {n_questions} questions, {n_answers} answers")


if __name__ == "__main__":
    main()
