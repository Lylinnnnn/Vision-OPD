#!/usr/bin/env python3
"""Validate four 2B CHAIR sources and compare their existing captions."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from robust_chair_analysis import (  # noqa: E402
    aggregate_from_arrays, build_double_word_dict, compute_per_sample, parse_official_synonyms,
)


def resolve(path):
    if path.is_file():
        return path
    candidates = [p for p in path.rglob("eval_results.jsonl")
                  if not any(part.startswith(".shards") for part in p.parts)
                  and p.parent.name in {"chair_official_v2", "official_v2"}]
    preferred = [p for p in candidates if p.parent.name == "chair_official_v2"]
    candidates = preferred or candidates
    if len(candidates) != 1:
        raise ValueError(f"Expected one complete CHAIR result under {path}; found: {candidates}")
    return candidates[0]


def load(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row.get("image_id") for row in rows]
    if len(rows) != 1000 or len(set(ids)) != 1000 or None in ids:
        raise ValueError(f"Expected 1000 unique image IDs: {path} ({len(rows)} rows)")
    for row in rows:
        caption = str(row.get("generated_caption") or "").strip()
        if not caption or caption.startswith("[ERROR]"):
            raise ValueError(f"Empty caption: {path}, image_id={row['image_id']}")
        if not row.get("gt_captions"):
            raise ValueError(f"Missing CHAIR reference captions: {path}, image_id={row['image_id']}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "random", "risk", "entropy"):
        parser.add_argument(f"--{name}-results", type=Path, required=True)
    parser.add_argument("--comparison-md", type=Path)
    args = parser.parse_args()
    expected = {"random": ("randommask", "global_step_156"), "risk": ("riskmask-nll", "global_step_150"), "entropy": ("entropymask", "global_step_156")}
    paths = {name: resolve(getattr(args, f"{name}_results")) for name in ("base", "random", "risk", "entropy")}
    for name, fragments in expected.items():
        path_text = str(paths[name]).lower()
        if any(fragment not in path_text for fragment in fragments):
            raise ValueError(f"{name} path must contain {fragments}: {path_text}")
    if "res-opd-" in str(paths["base"]).lower():
        raise ValueError(f"Base path appears to be a trained checkpoint: {paths['base']}")
    data = {name: load(path) for name, path in paths.items()}
    base_ids = {row["image_id"] for row in data["base"]}
    for name, rows in data.items():
        if {row["image_id"] for row in rows} != base_ids:
            raise ValueError(f"{name} image IDs differ from Base")
        prompts = {row.get("eval_prompt") for row in rows}
        ratios = {row.get("eval_student_ratio") for row in rows}
        modes = {row.get("eval_degradation_mode") for row in rows}
        if prompts != {"Describe the image."} or ratios != {1.0} or modes != {"original"}:
            raise ValueError(f"{name}: incompatible protocol prompt={prompts} ratio={ratios} mode={modes}")
        print(f"{name}: 1000/1000 original-view captions, protocol PASS: {paths[name]}")
    print(f"BASE_EVAL_RESULTS={paths['base'].resolve()}")
    if args.comparison_md:
        coco_words, inverse = parse_official_synonyms()
        double_words = build_double_word_dict()
        lines = ["# Four-model CHAIR comparison (2B, tr=0.50)", "",
                 "Existing captions only; no regeneration. Official CHAIR synonyms, all mentions, GT = instances + reference captions.", "",
                 "| Method | Step | CHAIRi | CHAIRs | ObjF1 | Response length (chars) | Correct mentions | Hall mentions |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, step in (("base", "--"), ("random", "156"), ("risk", "150"), ("entropy", "156")):
            rows = data[name]
            records = [dict(row, generated_text=row["generated_caption"]) for row in rows]
            per_sample = compute_per_sample(records, coco_words, inverse, double_words)
            metrics = aggregate_from_arrays(list(per_sample.values()), range(len(per_sample)))
            length = sum(len(row["generated_caption"]) for row in rows) / len(rows)
            label = {"base": "Base", "random": "RandomMask", "risk": "RiskMask (RKL x NLL)", "entropy": "EntropyMask"}[name]
            lines.append(f"| {label} | {step} | {metrics['CHAIRi']:.4f} | {metrics['CHAIRs']:.4f} | {metrics['ObjF1']:.4f} | {length:.1f} | {metrics['total_correct_mentions']} | {metrics['total_hallucinated_mentions']} |")
        lines.extend(["", "## Sources", ""])
        lines.extend(f"- {name}: `{path}`" for name, path in paths.items())
        args.comparison_md.parent.mkdir(parents=True, exist_ok=True)
        args.comparison_md.write_text("\n".join(lines) + "\n")
        print(f"COMPARISON_MD={args.comparison_md}")
    print("PREFLIGHT=PASS; selection analysis uses Base captions only; trained models are outcome controls")


if __name__ == "__main__":
    main()
