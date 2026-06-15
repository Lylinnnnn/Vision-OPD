"""
Prepare COCO val2017 data for Resolution-Aware Self-Distillation training.

Creates a parquet file compatible with verl's RLHFDataset, containing:
  - images:        original images (student degradation is done ONLINE by
                   ResOPDDataset, so student_px can be changed freely)
  - hires_images:  teacher images resized to target_px (sharp, consistent size
                   to prevent OOM from variable original resolutions)
  - prompt:        caption generation prompt in chat format
  - reward_model:  ground truth captions for CHAIR evaluation
  - extra_info:    metadata (image_id, original captions, object categories)

Usage:
    python scripts/prepare_data.py --data-dir ./data
    python scripts/prepare_data.py --data-dir ./data --target-px 448 --train-count 1500
"""

import argparse
import json
import os
import sys
from pathlib import Path

import datasets
import numpy as np
from PIL import Image

COCO_IMAGE_DIR = "/home/liuyanlin.lyl/notebook/data/COCO/coco2017val/val2017"
COCO_INSTANCES_PATH = "/home/liuyanlin.lyl/notebook/data/COCO/coco2017val/annotations/instances_val2017.json"
COCO_CAPTIONS_PATH = "/home/liuyanlin.lyl/notebook/data/COCO/coco2017val/annotations/captions_val2017.json"

PROMPT_TEXT = (
    "Please describe this image in detail. Include all visible objects, "
    "their spatial relationships, colors, sizes, and any text or fine details."
)

SPLIT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare COCO data for Res-OPD training.")
    parser.add_argument("--data-dir", default="./data", help="Output directory for processed data")
    parser.add_argument("--target-px", type=int, default=448,
                        help="Teacher image resolution (images are resized to this for consistent size)")
    parser.add_argument("--train-count", type=int, default=1500, help="Number of training samples")
    parser.add_argument("--eval-count", type=int, default=30, help="Number of eval samples (reserved)")
    parser.add_argument("--test-count", type=int, default=300, help="Number of test samples (reserved)")
    return parser.parse_args()


def make_hires_image(src_path: str, target_px: int, output_path: str) -> str:
    """Resize original image to target_px for teacher input (sharp, consistent size)."""
    img = Image.open(src_path).convert("RGB")
    hires = img.resize((target_px, target_px), Image.LANCZOS)
    hires.save(output_path, "JPEG", quality=95)
    return output_path


def load_coco_metadata():
    """Load COCO annotations and return image_id → metadata mapping."""
    with open(COCO_INSTANCES_PATH) as f:
        instances = json.load(f)
    with open(COCO_CAPTIONS_PATH) as f:
        captions_data = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in instances["categories"]}

    image_objects = {}
    for ann in instances["annotations"]:
        img_id = ann["image_id"]
        cat_name = cat_id_to_name.get(ann["category_id"], "unknown")
        image_objects.setdefault(img_id, set()).add(cat_name)

    image_captions = {}
    for ann in captions_data["annotations"]:
        img_id = ann["image_id"]
        image_captions.setdefault(img_id, []).append(ann["caption"])

    image_files = {img["id"]: img["file_name"] for img in instances["images"]}

    return image_files, image_objects, image_captions


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    teacher_dir = os.path.join(data_dir, "teacher_images")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(teacher_dir, exist_ok=True)

    print("Loading COCO metadata ...")
    image_files, image_objects, image_captions = load_coco_metadata()

    all_image_ids = sorted(image_files.keys())
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(all_image_ids)

    total_needed = args.train_count + args.eval_count + args.test_count
    if total_needed > len(all_image_ids):
        print(f"Warning: requested {total_needed} samples but only {len(all_image_ids)} available.")
        total_needed = len(all_image_ids)

    train_ids = all_image_ids[:args.train_count]
    eval_ids = all_image_ids[args.train_count:args.train_count + args.eval_count]
    test_ids = all_image_ids[args.train_count + args.eval_count:total_needed]

    print(f"Split: train={len(train_ids)}, eval={len(eval_ids)}, test={len(test_ids)}")
    print(f"Teacher resolution: {args.target_px}px (pre-resized for consistent size)")

    # Save split info
    split_info = {
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "test_ids": test_ids,
        "target_px": args.target_px,
    }
    with open(os.path.join(data_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # Build training records
    # - images: original COCO image path (student degradation done online)
    # - hires_images: pre-resized to target_px (consistent teacher size, prevents OOM)
    records = []
    for idx, image_id in enumerate(train_ids):
        file_name = image_files[image_id]
        src_path = os.path.join(COCO_IMAGE_DIR, file_name)
        if not os.path.exists(src_path):
            print(f"  Warning: {src_path} not found, skipping.")
            continue

        # Pre-resize teacher image to target_px for consistent memory usage
        stem = Path(file_name).stem
        teacher_path = os.path.join(teacher_dir, f"{stem}_t{args.target_px}.jpg")
        if not os.path.exists(teacher_path):
            make_hires_image(src_path, args.target_px, teacher_path)

        captions = image_captions.get(image_id, [])
        objects = sorted(image_objects.get(image_id, set()))
        ground_truth = captions[0] if captions else ""

        record = {
            "data_source": "coco_res_opd",
            "prompt": [{"role": "user", "content": f"<image>{PROMPT_TEXT}"}],
            "images": [{"path": src_path}],
            "hires_images": [{"path": teacher_path}],
            "ability": "image_captioning",
            "reward_model": {
                "style": "none",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "image_id": image_id,
                "file_name": file_name,
                "captions": captions,
                "objects": objects,
                "question": PROMPT_TEXT,
            },
        }
        records.append(record)

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(train_ids)} images")

    print(f"Built {len(records)} training records")

    dataset = datasets.Dataset.from_list(records)
    output_path = os.path.join(data_dir, "train.parquet")
    dataset.to_parquet(output_path)
    print(f"Saved to {output_path}")

    # Build eval parquet for verl's native validation (val_files)
    # Same format as train parquet so verl can do rollout + reward scoring
    eval_records = []
    for image_id in eval_ids:
        file_name = image_files[image_id]
        src_path = os.path.join(COCO_IMAGE_DIR, file_name)
        if not os.path.exists(src_path):
            continue

        stem = Path(file_name).stem
        teacher_path = os.path.join(teacher_dir, f"{stem}_t{args.target_px}.jpg")
        if not os.path.exists(teacher_path):
            make_hires_image(src_path, args.target_px, teacher_path)

        captions = image_captions.get(image_id, [])
        objects = sorted(image_objects.get(image_id, set()))
        ground_truth = captions[0] if captions else ""

        eval_records.append({
            "data_source": "coco_res_opd",
            "prompt": [{"role": "user", "content": f"<image>{PROMPT_TEXT}"}],
            "images": [{"path": src_path}],
            "hires_images": [{"path": teacher_path}],
            "ability": "image_captioning",
            "reward_model": {
                "style": "none",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "image_id": image_id,
                "file_name": file_name,
                "captions": captions,
                "objects": objects,
                "question": PROMPT_TEXT,
            },
        })

    eval_dataset = datasets.Dataset.from_list(eval_records)
    eval_parquet_path = os.path.join(data_dir, "eval.parquet")
    eval_dataset.to_parquet(eval_parquet_path)
    print(f"Saved eval parquet ({len(eval_records)} samples) to {eval_parquet_path}")

    # Save eval/test metadata as JSON for standalone CHAIR evaluation
    for split_name, split_ids in [("eval", eval_ids), ("test", test_ids)]:
        split_records = []
        for image_id in split_ids:
            file_name = image_files[image_id]
            src_path = os.path.join(COCO_IMAGE_DIR, file_name)
            captions = image_captions.get(image_id, [])
            objects = sorted(image_objects.get(image_id, set()))
            split_records.append({
                "image_id": image_id,
                "file_name": file_name,
                "image_path": src_path,
                "captions": captions,
                "objects": objects,
            })
        split_path = os.path.join(data_dir, f"{split_name}.json")
        with open(split_path, "w") as f:
            json.dump(split_records, f, indent=2)
        print(f"Saved {split_name} metadata ({len(split_records)} samples) to {split_path}")

    print(f"\nData preparation complete.")
    print(f"  Training data:  {output_path}")
    print(f"  Eval parquet:   {eval_parquet_path} (for verl val_files)")
    print(f"  Eval metadata:  {os.path.join(data_dir, 'eval.json')}")
    print(f"  Test metadata:  {os.path.join(data_dir, 'test.json')}")


if __name__ == "__main__":
    main()
