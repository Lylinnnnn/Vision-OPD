#!/usr/bin/env python3
"""Summarize AMBER generative response length and noun exposure.

This probe is intentionally lightweight. It reads AMBER raw_results.jsonl files
produced by res-opd/eval/eval_amber.py, keeps only generative samples, and
reports response length / noun-count statistics. If AMBER metadata is available,
it also computes exact-match proxies for the official generative CHAIR
denominator and numerator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROBE_DIR = Path(__file__).resolve().parent
RES_OPD_ROOT = PROBE_DIR.parent
EVAL_DIR = RES_OPD_ROOT / "eval"
sys.path.insert(0, str(EVAL_DIR))

from eval_amber import extract_final_response_text  # noqa: E402


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze AMBER generative outputs for length, noun count, and "
            "official-light noun exposure proxies."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "One or more LABEL=PATH entries. PATH can be raw_results.jsonl, "
            "an amber directory, or an eval dataset directory containing amber/."
        ),
    )
    parser.add_argument(
        "--amber-root",
        default="/home/liuyanlin.lyl/notebook/data/AMBER",
        help="AMBER root containing data/relation.json, annotations.json, safe_words.txt.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROBE_DIR / "results" / "amber_generative_stats"),
        help="Directory for summary.json, summary.md, and examples.jsonl.",
    )
    parser.add_argument(
        "--baseline-label",
        default=None,
        help="Optional baseline label for delta columns. Defaults to the first input.",
    )
    parser.add_argument("--top-examples", type=int, default=30)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_raw_results(path_text: str) -> Path:
    path = Path(path_text)
    candidates = []
    if path.is_file():
        candidates.append(path)
    else:
        candidates.extend(
            [
                path / "raw_results.jsonl",
                path / "amber" / "raw_results.jsonl",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find AMBER raw_results.jsonl under: {path_text}")


def parse_labeled_inputs(entries: list[str]) -> list[tuple[str, Path]]:
    out = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected LABEL=PATH input, got: {entry}")
        label, path_text = entry.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty label in input: {entry}")
        out.append((label, resolve_raw_results(path_text)))
    return out


class NounExtractor:
    def __init__(self) -> None:
        self.mode = "regex"
        self._wordnet = None
        try:
            import nltk  # type: ignore
            from nltk.stem import WordNetLemmatizer  # type: ignore

            self._nltk = nltk
            self._wordnet = WordNetLemmatizer()
            # Probe required resources once. If unavailable, fall back quietly.
            nltk.pos_tag(["test"])
            self.mode = "nltk"
        except Exception:
            self._nltk = None

    def words(self, text: str) -> list[str]:
        return [w.lower() for w in WORD_RE.findall(text or "")]

    def nouns(self, text: str) -> list[str]:
        if self.mode == "nltk":
            try:
                tokens = self._nltk.word_tokenize(text)  # type: ignore[union-attr]
                tagged = self._nltk.pos_tag(tokens)  # type: ignore[union-attr]
                nouns = []
                for word, pos in tagged:
                    if pos.startswith("NN") and WORD_RE.fullmatch(word):
                        word_lc = word.lower()
                        if self._wordnet is not None:
                            word_lc = self._wordnet.lemmatize(word_lc)
                        nouns.append(word_lc)
                return nouns
            except Exception:
                pass
        return self.words(text)


def load_amber_metadata(amber_root: Path) -> dict[str, Any]:
    data_dir = amber_root / "data"
    relation_path = data_dir / "relation.json"
    annotation_path = data_dir / "annotations.json"
    safe_words_path = data_dir / "safe_words.txt"
    if not (relation_path.exists() and annotation_path.exists() and safe_words_path.exists()):
        return {"available": False}

    relation = load_json(relation_path)
    annotations = load_json(annotation_path)
    safe_words = set()
    with safe_words_path.open("r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                safe_words.add(word)

    hallucination_words = set()
    for key, values in relation.items():
        hallucination_words.add(str(key).lower())
        for value in values:
            hallucination_words.add(str(value).lower())

    by_id = {}
    for idx, item in enumerate(annotations, start=1):
        if isinstance(item, dict):
            item_id = int(item.get("id", idx))
            by_id[item_id] = item

    return {
        "available": True,
        "relation": relation,
        "annotations_by_id": by_id,
        "safe_words": safe_words,
        "hallucination_words": hallucination_words,
    }


def lower_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(v).lower() for v in values]


def related_words(relation: dict[str, Any], words: list[str]) -> set[str]:
    out = set(words)
    for word in words:
        values = relation.get(word, [])
        for value in values:
            out.add(str(value).lower())
    return out


def official_light_stats(
    item_id: int,
    nouns: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not metadata.get("available"):
        return {}
    annotation = metadata["annotations_by_id"].get(item_id)
    if not annotation or annotation.get("type") != "generative":
        return {}

    relation = metadata["relation"]
    hallucination_words = metadata["hallucination_words"]
    global_safe_words = metadata["safe_words"]
    truth = lower_list(annotation.get("truth"))
    hallu = lower_list(annotation.get("hallu"))
    safe_set = related_words(relation, truth)
    hallu_set = related_words(relation, hallu)

    candidate_nouns = [noun for noun in nouns if noun in hallucination_words]
    unknown = []
    safe_hits = []
    hallu_hits = []
    for noun in candidate_nouns:
        if noun in global_safe_words:
            continue
        if noun in safe_set:
            safe_hits.append(noun)
        elif noun in hallu_set:
            hallu_hits.append(noun)
        else:
            unknown.append(noun)

    truth_covered = len({word for word in truth if word in candidate_nouns})
    hallu_covered = len({word for word in hallu if word in candidate_nouns})
    return {
        "candidate_noun_count": len(candidate_nouns),
        "unknown_candidate_count": len(unknown),
        "safe_hit_count": len(safe_hits),
        "hallu_hit_count": len(hallu_hits),
        "truth_count": len(truth),
        "truth_covered_exact": truth_covered,
        "hallu_count": len(hallu),
        "hallu_covered_exact": hallu_covered,
        "unknown_candidate_examples": sorted(Counter(unknown).items(), key=lambda x: (-x[1], x[0]))[:10],
        "candidate_examples": sorted(Counter(candidate_nouns).items(), key=lambda x: (-x[1], x[0]))[:10],
    }


def max_repeated_ngram(words: list[str], n: int = 5) -> tuple[str, int]:
    if len(words) < n:
        return "", 0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return "", 0
    gram, count = Counter(grams).most_common(1)[0]
    return gram, count


def is_generative(record: dict[str, Any], metadata: dict[str, Any]) -> bool:
    item_id = record.get("id")
    try:
        item_id_int = int(item_id)
    except (TypeError, ValueError):
        return not bool(record.get("is_discriminative"))
    if metadata.get("available"):
        annotation = metadata["annotations_by_id"].get(item_id_int)
        if annotation:
            return annotation.get("type") == "generative"
    return item_id_int < 1005 or not bool(record.get("is_discriminative"))


def quantile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.fmean(vals) if vals else None


def summarize_values(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(r[field]) for r in records if isinstance(r.get(field), (int, float))]
    return {
        "mean": mean(values),
        "median": quantile(values, 0.5),
        "p90": quantile(values, 0.9),
        "max": max(values) if values else None,
    }


def analyze_one(label: str, path: Path, metadata: dict[str, Any], extractor: NounExtractor) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    analyzed = []
    for record in rows:
        if not is_generative(record, metadata):
            continue
        item_id = int(record["id"])
        response = record.get("official_response") or extract_final_response_text(record.get("model_answer", ""))
        response = str(response or "")
        words = extractor.words(response)
        nouns = extractor.nouns(response)
        repeat5_text, repeat5_count = max_repeated_ngram(words, 5)
        raw_answer = str(record.get("model_answer", ""))
        has_think_close = "</think>" in raw_answer
        stats = {
            "label": label,
            "id": item_id,
            "image": record.get("image"),
            "response": response,
            "char_count": len(response),
            "word_count": len(words),
            "sentence_count": len([s for s in re.split(r"[.!?]+", response) if s.strip()]),
            "noun_count": len(nouns),
            "unique_noun_count": len(set(nouns)),
            "has_think_close": has_think_close,
            "missing_think_close": bool(raw_answer and not has_think_close and raw_answer.strip().lower().startswith(("so,", "let", "got it"))),
            "repeat5_max_count": repeat5_count,
            "repeat5_text": repeat5_text,
            "loop_like": repeat5_count >= 20,
        }
        stats.update(official_light_stats(item_id, nouns, metadata))
        analyzed.append(stats)

    summary = {
        "label": label,
        "source_path": str(path),
        "n": len(analyzed),
        "noun_extractor": extractor.mode,
        "metadata_available": bool(metadata.get("available")),
    }
    for field in (
        "char_count",
        "word_count",
        "sentence_count",
        "noun_count",
        "unique_noun_count",
        "repeat5_max_count",
        "candidate_noun_count",
        "unknown_candidate_count",
        "safe_hit_count",
        "hallu_hit_count",
    ):
        summary[field] = summarize_values(analyzed, field)
    summary["has_think_close_rate"] = (
        sum(1 for r in analyzed if r.get("has_think_close")) / len(analyzed) if analyzed else None
    )
    summary["missing_think_close_count"] = sum(1 for r in analyzed if r.get("missing_think_close"))
    summary["missing_think_close_rate"] = (
        summary["missing_think_close_count"] / len(analyzed) if analyzed else None
    )
    summary["loop_like_count"] = sum(1 for r in analyzed if r.get("loop_like"))
    summary["loop_like_rate"] = summary["loop_like_count"] / len(analyzed) if analyzed else None
    for numer, denom, out_name in (
        ("unknown_candidate_count", "candidate_noun_count", "unknown_candidate_rate"),
        ("truth_covered_exact", "truth_count", "truth_cover_exact_rate"),
        ("hallu_covered_exact", "hallu_count", "hallu_cover_exact_rate"),
    ):
        total_num = sum(int(r.get(numer, 0)) for r in analyzed)
        total_den = sum(int(r.get(denom, 0)) for r in analyzed)
        summary[out_name] = total_num / total_den if total_den else None
        summary[f"{numer}_sum"] = total_num
        summary[f"{denom}_sum"] = total_den
    return summary, analyzed


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def metric_mean(summary: dict[str, Any], metric: str) -> float | None:
    value = summary.get(metric)
    if isinstance(value, dict):
        return value.get("mean")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def write_markdown(path: Path, summaries: list[dict[str, Any]], baseline_label: str) -> None:
    by_label = {s["label"]: s for s in summaries}
    baseline = by_label.get(baseline_label)
    rows = []
    metrics = [
        ("word_count", "Words"),
        ("noun_count", "Nouns"),
        ("unique_noun_count", "Unique nouns"),
        ("missing_think_close_rate", "Missing think close"),
        ("loop_like_rate", "Loop-like"),
        ("repeat5_max_count", "Max repeated 5gram"),
        ("candidate_noun_count", "AMBER candidate nouns"),
        ("unknown_candidate_count", "Unknown candidate nouns"),
        ("unknown_candidate_rate", "Unknown/candidate"),
        ("truth_cover_exact_rate", "Truth cover exact"),
        ("hallu_cover_exact_rate", "Hallu cover exact"),
    ]
    header = ["Model", "N"]
    for _, name in metrics:
        header.append(name)
        header.append("Delta")
    rows.append("| " + " | ".join(header) + " |")
    rows.append("| " + " | ".join(["---"] * len(header)) + " |")
    for summary in summaries:
        row = [summary["label"], str(summary["n"])]
        for key, _ in metrics:
            value = metric_mean(summary, key)
            base_value = metric_mean(baseline, key) if baseline else None
            delta = value - base_value if value is not None and base_value is not None else None
            row.append(fmt(value, 3 if "rate" in key else 2))
            row.append(fmt(delta, 3 if "rate" in key else 2))
        rows.append("| " + " | ".join(row) + " |")

    lines = [
        "# AMBER Generative Response Stats",
        "",
        f"Baseline label: `{baseline_label}`",
        "",
        "All length/noun columns are means over generative AMBER samples.",
        "`AMBER candidate nouns` approximates the official generative CHAIR denominator by counting nouns that appear in AMBER's relation vocabulary.",
        "`Unknown candidate nouns` is an exact-match numerator proxy and does not use spaCy semantic similarity.",
        "",
        *rows,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    inputs = parse_labeled_inputs(args.inputs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_amber_metadata(Path(args.amber_root))
    extractor = NounExtractor()
    summaries = []
    all_records = []
    for label, path in inputs:
        summary, records = analyze_one(label, path, metadata, extractor)
        summaries.append(summary)
        all_records.extend(records)

    baseline_label = args.baseline_label or summaries[0]["label"]
    payload = {
        "baseline_label": baseline_label,
        "amber_root": args.amber_root,
        "metadata_available": bool(metadata.get("available")),
        "noun_extractor": extractor.mode,
        "summaries": summaries,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    write_markdown(output_dir / "summary.md", summaries, baseline_label)

    ranked = sorted(
        all_records,
        key=lambda r: (
            float(r.get("unknown_candidate_count", 0)),
            float(r.get("candidate_noun_count", 0)),
            float(r.get("noun_count", 0)),
            float(r.get("word_count", 0)),
        ),
        reverse=True,
    )
    with (output_dir / "examples.jsonl").open("w", encoding="utf-8") as f:
        for row in ranked[: args.top_examples]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote: {output_dir / 'summary.json'}")
    print(f"Wrote: {output_dir / 'summary.md'}")
    print(f"Wrote: {output_dir / 'examples.jsonl'}")


if __name__ == "__main__":
    main()
