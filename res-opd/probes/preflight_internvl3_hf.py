#!/usr/bin/env python3
"""Check InternVL HF checkpoint metadata and image-processing alignment."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--teacher-ratio", type=float, default=0.75)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.teacher_ratio <= 1:
        raise SystemExit("--teacher-ratio must be between 0 and 1")
    model_path = args.model_path.expanduser().resolve()
    config_path = model_path / "config.json"
    processor_path = model_path / "processor_config.json"
    if not config_path.is_file() or not processor_path.is_file():
        raise SystemExit("Missing config.json or processor_config.json in downloaded model directory")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    processor_config = json.loads(processor_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "internvl":
        raise SystemExit(f"Expected HF InternVL model_type=internvl, got {config.get('model_type')!r}")
    if "InternVLForConditionalGeneration" not in config.get("architectures", []):
        raise SystemExit(f"Unexpected architecture: {config.get('architectures')!r}")
    if processor_config.get("processor_class") != "InternVLProcessor":
        raise SystemExit(f"Unexpected processor: {processor_config.get('processor_class')!r}")

    from PIL import Image, ImageDraw
    from transformers import AutoProcessor
    from verl.utils.internvl_hf import assert_internvl_image_alignment, internvl_vllm_prompt
    from verl.utils.tokenizer import hf_processor

    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    if type(processor).__name__ != "InternVLProcessor":
        raise SystemExit(f"Loaded unexpected processor: {type(processor).__name__}")
    adapted_processor = hf_processor(str(model_path), local_files_only=True)
    if adapted_processor is None or hasattr(adapted_processor, "get_rope_index"):
        raise SystemExit("verl did not load InternVLProcessor with standard 1D positions")

    image = Image.new("RGB", (640, 480), color=(80, 120, 160))
    draw = ImageDraw.Draw(image)
    for x in range(0, image.width, 3):
        draw.line((x, 0, x, image.height), fill=(240, 30, 90))
    for y in range(0, image.height, 3):
        draw.line((0, y, image.width, y), fill=(20, 220, 70))
    if args.teacher_ratio <= 0:
        lowres = Image.new("RGB", image.size, color=(128, 128, 128))
    elif args.teacher_ratio >= 1:
        lowres = image.copy()
    else:
        small_size = tuple(max(1, int(round(side * args.teacher_ratio))) for side in image.size)
        lowres = image.resize(small_size, Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.LANCZOS)
    image_token_id = config.get("image_token_id")
    if not isinstance(image_token_id, int):
        raise SystemExit("Missing integer image_token_id")

    import torch

    visual_batches = []
    for view, picture in (("original", image), (f"teacher_{args.teacher_ratio}", lowres)):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": picture},
            {"type": "text", "text": "Describe the image."},
        ]}]
        batch = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        image_tokens = int((input_ids == image_token_id).sum().item())
        vision_keys = [key for key in batch if key not in {"input_ids", "attention_mask"}]
        if image_tokens == 0 or not vision_keys:
            raise SystemExit(f"{view}: image tokens or visual tensors missing")
        if "pixel_values" not in batch:
            raise SystemExit(f"{view}: pixel_values missing")
        raw_prompt = adapted_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        vllm_prompt = internvl_vllm_prompt(raw_prompt, adapted_processor, 1)
        vllm_prompt_ids = adapted_processor.tokenizer(vllm_prompt, return_tensors="pt")["input_ids"]
        if vllm_prompt.count("<image>") != 1 or vllm_prompt_ids.shape[-1] >= input_ids.shape[-1]:
            raise SystemExit(f"{view}: vLLM image placeholder was not kept compact")
        rollout_batch = adapted_processor(
            text=[raw_prompt], images=[picture], return_tensors="pt"
        )
        teacher_batch = adapted_processor(
            text=[raw_prompt], images=[picture], return_tensors="pt",
            truncation=True, max_length=8192,
        )
        visual_batch = adapted_processor(
            text=[adapted_processor.image_token], images=[picture], return_tensors="pt"
        )
        if not torch.equal(input_ids, rollout_batch["input_ids"]):
            raise SystemExit(f"{view}: rollout prompt tokenization differs from HF reference")
        if not torch.equal(batch["pixel_values"], visual_batch["pixel_values"]):
            raise SystemExit(f"{view}: rollout visual preprocessing differs from HF reference")
        assert_internvl_image_alignment(input_ids, visual_batch["pixel_values"], adapted_processor, view)
        teacher_tokens = int((teacher_batch["input_ids"] == image_token_id).sum().item())
        expected_teacher_tokens = teacher_batch["pixel_values"].shape[0] * adapted_processor.image_seq_length
        if teacher_tokens != expected_teacher_tokens:
            raise SystemExit(f"{view}: teacher prompt truncation split the image context tokens")
        visual_batches.append(batch["pixel_values"])
        print(
            f"{view}: hf_prompt_tokens={input_ids.shape[-1]} "
            f"vllm_placeholder_tokens={vllm_prompt_ids.shape[-1]} "
            f"image_tokens={image_tokens} vision_keys={vision_keys}"
        )

    if visual_batches[0].shape != visual_batches[1].shape:
        raise SystemExit("Original and low-res views have different visual tensor shapes")
    if args.teacher_ratio < 1 and torch.equal(visual_batches[0], visual_batches[1]):
        raise SystemExit("Original and low-res visual tensors are identical")

    print("PASS: HF checkpoint metadata, two-view preprocessing, and prompt bridge smoke")
    print("NOTE: Live vLLM prompt expansion, teacher forward, and FSDP training still need a GPU smoke")


if __name__ == "__main__":
    main()
