#!/usr/bin/env python3
"""Check that InternVL low-res SFT examples retain aligned visual supervision."""

import argparse
import copy
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from omegaconf import OmegaConf

RES_OPD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RES_OPD_ROOT.parent))
sys.path.insert(0, str(RES_OPD_ROOT / "datasets"))
sys.path.insert(0, str(RES_OPD_ROOT / "pipelines" / "lowres_sft"))

from verl.utils.internvl_hf import assert_internvl_image_alignment, is_internvl_hf_processor
from verl.utils.tokenizer import hf_processor, hf_tokenizer
from res_opd_sft_dataset import ResOPDSFTDataset
from prepare_data import build_sft_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True, type=Path)
    args = parser.parse_args()

    tokenizer = hf_tokenizer(args.model_path, local_files_only=True)
    processor = hf_processor(args.model_path, local_files_only=True)
    if not is_internvl_hf_processor(processor):
        raise RuntimeError("Expected the HF InternVLProcessor")

    frame = pd.read_parquet(args.train_file)
    if len(frame) < 2:
        raise RuntimeError("Expected at least two training rows")
    indices = sorted({0, len(frame) // 2, len(frame) - 1})
    records = [build_sft_record(frame.iloc[index].to_dict(), "A visible object is in the image.") for index in indices]
    config = OmegaConf.create({
        "pad_mode": "no_padding",
        "max_length": 8192,
        "truncation": "right",
        "messages_key": "messages",
        "image_key": "images",
        "ignore_input_ids_mismatch": False,
    })

    with tempfile.TemporaryDirectory(prefix="internvl_sft_preflight_") as directory:
        path = Path(directory) / "samples.parquet"
        Dataset.from_list(records).to_parquet(str(path))
        dataset = ResOPDSFTDataset(str(path), tokenizer, config, processor=processor)
        for index in range(len(dataset)):
            sample = dataset[index]
            inputs = sample.get("multi_modal_inputs", {})
            pixels = inputs.get("pixel_values")
            if pixels is None or not torch.is_tensor(pixels):
                raise RuntimeError(f"sample {index}: pixel_values missing")
            assert_internvl_image_alignment(sample["input_ids"], pixels, processor, f"SFT sample {index}")
            mask = sample["loss_mask"].bool()
            if not mask.any() or mask.all():
                raise RuntimeError(f"sample {index}: assistant loss mask is empty or covers the prompt")
            if mask[sample["input_ids"] == processor.image_token_id].any():
                raise RuntimeError(f"sample {index}: image context is included in the loss")
            row = copy.deepcopy(dataset.dataframe.iloc[index].to_dict())
            messages = dataset._build_messages(row)
            reference = processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
            if not torch.equal(sample["input_ids"], reference["input_ids"][0]):
                raise RuntimeError(f"sample {index}: SFT token sequence differs from HF processor")
            if not torch.equal(pixels, reference["pixel_values"]):
                raise RuntimeError(f"sample {index}: SFT image tensor differs from HF processor")
            print(f"sample={index} tokens={sample['input_ids'].numel()} image_features={pixels.shape[0]} "
                  f"assistant_tokens={int(mask.sum())}", flush=True)

    print("INTERNVL_SFT_PREFLIGHT=PASS", flush=True)


if __name__ == "__main__":
    main()
