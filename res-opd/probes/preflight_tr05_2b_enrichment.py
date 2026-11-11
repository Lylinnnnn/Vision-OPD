#!/usr/bin/env python3
"""Validate the three evaluation sources before Base-caption forced scoring."""

import argparse
import json
from pathlib import Path


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
    for name in ("base", "random", "risk"):
        parser.add_argument(f"--{name}-results", type=Path, required=True)
    args = parser.parse_args()
    expected = {"random": ("randommask", "global_step_156"), "risk": ("riskmask-nll", "global_step_150")}
    for name, fragments in expected.items():
        path_text = str(getattr(args, f"{name}_results")).lower()
        if any(fragment not in path_text for fragment in fragments):
            raise ValueError(f"{name} path must contain {fragments}: {path_text}")
    if "res-opd-" in str(args.base_results).lower():
        raise ValueError(f"Base path appears to be a trained checkpoint: {args.base_results}")
    data = {name: load(getattr(args, f"{name}_results")) for name in ("base", "random", "risk")}
    base_ids = {row["image_id"] for row in data["base"]}
    for name, rows in data.items():
        if {row["image_id"] for row in rows} != base_ids:
            raise ValueError(f"{name} image IDs differ from Base")
        prompts = {row.get("eval_prompt") for row in rows}
        ratios = {row.get("eval_student_ratio") for row in rows}
        modes = {row.get("eval_degradation_mode") for row in rows}
        if prompts != {"Describe the image."} or ratios != {1.0} or modes != {"original"}:
            raise ValueError(f"{name}: incompatible protocol prompt={prompts} ratio={ratios} mode={modes}")
        print(f"{name}: 1000/1000 original-view captions, protocol PASS: {getattr(args, f'{name}_results')}")
    print("PREFLIGHT=PASS; selection analysis uses Base captions only; RandomMask/RiskMask are outcome controls")


if __name__ == "__main__":
    main()
