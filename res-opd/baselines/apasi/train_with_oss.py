#!/usr/bin/env python3
"""Run APASI DPO with a verified OSS upload after every epoch checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path


SYNC_SOURCE = Path(__file__).resolve().parents[1] / "rlsf_v/sync_oss.py"


def checkpoint_info(output_dir: Path, epoch: float, step: int, oss_prefix: str) -> tuple[Path, str]:
    if not math.isfinite(epoch) or epoch < 0 or step < 1:
        raise ValueError(f"Invalid epoch checkpoint: epoch={epoch}, step={step}")
    if not oss_prefix.startswith("oss://"):
        raise ValueError("Epoch checkpoint destination must be an oss:// URL")
    epoch_index = max(1, round(epoch))
    checkpoint = output_dir / f"checkpoint-{step}"
    destination = f"{oss_prefix.rstrip('/')}/epoch_{epoch_index:02d}_step_{step}"
    return checkpoint, destination


def upload_epoch(output_dir: Path, epoch: float, step: int, oss_prefix: str) -> str:
    checkpoint, destination = checkpoint_info(output_dir, epoch, step, oss_prefix)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Epoch checkpoint was not saved: {checkpoint}")
    spec = importlib.util.spec_from_file_location("apasi_sync_oss", SYNC_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.sync_tree(checkpoint, destination, "adapter")
    print(f"APASI_EPOCH_OSS_SYNC=PASS epoch={epoch:.4f} step={step} destination={destination}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--oss-prefix", required=True)
    args = parser.parse_args()

    import yaml
    from transformers import TrainerCallback
    from llamafactory.train.tuner import run_exp

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("stage") != "dpo" or config.get("save_strategy") != "epoch":
        raise ValueError("APASI requires DPO with save_strategy=epoch")
    output_dir = Path(config["output_dir"])
    synced = []

    class EpochOSSSyncCallback(TrainerCallback):
        def on_save(self, training_args, state, control, **kwargs):
            if training_args.should_save:
                synced.append(upload_epoch(
                    output_dir, float(state.epoch), int(state.global_step), args.oss_prefix
                ))
            return control

    run_exp(args=config, callbacks=[EpochOSSSyncCallback()])
    if int(os.environ.get("RANK", "0")) == 0:
        expected = 1 if config.get("max_steps", 0) else math.ceil(float(config["num_train_epochs"]))
        if len(synced) < expected:
            raise RuntimeError(f"Only {len(synced)}/{expected} epoch checkpoints reached OSS")
        print(f"APASI_EPOCHS_OSS=PASS checkpoints={len(synced)}", flush=True)


if __name__ == "__main__":
    main()
