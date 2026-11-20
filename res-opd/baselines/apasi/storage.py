#!/usr/bin/env python3
"""Verified OSS storage for APASI-owned adapter and merged-model directories."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[2]
SYNC_SOURCE = Path(__file__).resolve().parents[1] / "rlsf_v/sync_oss.py"
MODELS = ("qwen2b", "qwen4b", "qwen8b", "internvl2b")


def sync_module():
    spec = importlib.util.spec_from_file_location("rlsf_v_sync_oss", SYNC_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def owned_target(root: Path, run_name: str, model: str, artifact: str,
                 stage: str = "") -> Path:
    root = root.absolute()
    if (root.name not in {"apasi_coco5k", "apasi_smoke"} or root.is_symlink()
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name)
            or model not in MODELS or artifact not in {"adapter", "merged"}):
        raise ValueError("Refusing to touch an unexpected APASI work root or artifact")
    run_dir = root / run_name
    directory = run_dir / model
    if run_dir.is_symlink() or directory.is_symlink():
        raise ValueError(f"Refusing to use symlink: {run_dir} or {directory}")
    if stage:
        if artifact != "merged" or not re.fullmatch(r"[A-Za-z0-9_-]+", stage):
            raise ValueError("Only merged models can use a safe staging name")
        target = directory / f".merged_stage_{stage}"
    else:
        target = directory / artifact
    if target.is_symlink():
        raise ValueError(f"Refusing to use symlink: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "restore", "cleanup"))
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--artifact", choices=("adapter", "merged"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/apasi_coco5k"))
    parser.add_argument("--stage", default="")
    parser.add_argument("--keep-local", action="store_true")
    args = parser.parse_args()
    target = owned_target(args.work_root, args.run_name, args.model,
                          args.artifact, args.stage)
    base = os.environ.get(
        "APASI_OSS_BASE", "oss://industry-algo/yanlin/ckpt/OPD/v4/APASI_COCO5K"
    ).rstrip("/")
    if not base.startswith("oss://"):
        raise ValueError("APASI_OSS_BASE must be an oss:// URL")
    uri = f"{base}/{args.run_name}/{args.model}/{args.artifact}"
    sync = sync_module()
    if args.action == "upload":
        if args.stage:
            parser.error("Upload the final artifact, not a temporary stage")
        if args.artifact == "adapter":
            sync.prune_training_checkpoints(target)
        sync.sync_tree(target, uri, args.artifact)
        if not args.keep_local:
            shutil.rmtree(target)
            print(f"APASI_LOCAL_CLEANUP=PASS path={target}")
    elif args.action == "restore":
        sync.restore_tree(uri, target, args.artifact)
    else:
        if not target.is_dir():
            raise FileNotFoundError(target)
        shutil.rmtree(target)
        print(f"APASI_LOCAL_CLEANUP=PASS path={target}")


if __name__ == "__main__":
    main()
