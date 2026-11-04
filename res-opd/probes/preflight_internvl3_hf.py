#!/usr/bin/env python3
"""Check InternVL HF checkpoint metadata and image-processing alignment."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
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

    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    if type(processor).__name__ != "InternVLProcessor":
        raise SystemExit(f"Loaded unexpected processor: {type(processor).__name__}")

    image = Image.new("RGB", (640, 480), color=(80, 120, 160))
    draw = ImageDraw.Draw(image)
    for x in range(0, image.width, 3):
        draw.line((x, 0, x, image.height), fill=(240, 30, 90))
    for y in range(0, image.height, 3):
        draw.line((0, y, image.width, y), fill=(20, 220, 70))
    lowres = image.resize((320, 240), Image.Resampling.BICUBIC).resize(
        image.size, Image.Resampling.BICUBIC
    )
    image_token_id = config.get("image_token_id")
    if not isinstance(image_token_id, int):
        raise SystemExit("Missing integer image_token_id")

    visual_batches = []
    for view, picture in (("original", image), ("lowres_0.5", lowres)):
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
        visual_batches.append(batch["pixel_values"])
        print(f"{view}: tokens={input_ids.shape[-1]} image_tokens={image_tokens} vision_keys={vision_keys}")

    import torch

    if visual_batches[0].shape != visual_batches[1].shape:
        raise SystemExit("Original and low-res views have different visual tensor shapes")
    if torch.equal(visual_batches[0], visual_batches[1]):
        raise SystemExit("Original and low-res visual tensors are identical")

    print("PASS: HF checkpoint metadata and two-view processor smoke")
    print("NOTE: This does not validate verl rollout, teacher forward, FSDP training, or vLLM")


if __name__ == "__main__":
    main()
