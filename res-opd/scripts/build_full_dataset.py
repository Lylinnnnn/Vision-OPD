#!/usr/bin/env python3
"""
Build full-scale OPD dataset:
  - Train: 5k images sampled from COCO train2017 (no caption/annotation needed for OPD)
  - Val:   100 images sampled from COCO val2017 (mini-eval during training)
  - Test:  1000 images sampled from COCO val2017 (with captions & objects, same format as existing test.json)

Output:
  - data/train_5k.parquet    (training set, parquet format matching existing train.parquet)
  - data/val.parquet         (mini-eval set, 100 samples, parquet format)
  - data/test_1000.json      (test set, JSON format matching existing test.json)
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TRAIN_SAMPLE_SIZE = 5000
VAL_SAMPLE_SIZE = 100
TEST_SAMPLE_SIZE = 1000

RES_OPD_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = RES_OPD_ROOT / "data"
COCO_ROOT = Path("/home/liuyanlin.lyl/notebook/data/COCO")
TRAIN_IMG_DIR = COCO_ROOT / "train2017"
VAL_IMG_DIR = COCO_ROOT / "coco2017val" / "val2017"
VAL_ANNOTATIONS = COCO_ROOT / "coco2017val" / "annotations"

PROMPT_TEMPLATE = [
    {
        "role": "user",
        "content": "<image>Please describe this image in detail. Include all visible objects, their spatial relationships, colors, sizes, and any text or fine details.",
    }
]


def build_train_set():
    """Sample 10k images from train2017, build parquet with OPD-compatible format."""
    print(f"[Train] Scanning {TRAIN_IMG_DIR} ...")
    all_images = sorted(TRAIN_IMG_DIR.glob("*.jpg"))
    print(f"[Train] Found {len(all_images)} images")

    if len(all_images) < TRAIN_SAMPLE_SIZE:
        raise RuntimeError(
            f"Not enough images: {len(all_images)} < {TRAIN_SAMPLE_SIZE}. "
            "Is train2017 still being extracted?"
        )

    random.seed(SEED)
    sampled = random.sample(all_images, TRAIN_SAMPLE_SIZE)
    sampled.sort(key=lambda p: p.name)

    records = []
    for img_path in sampled:
        file_name = img_path.name
        image_id = int(file_name.replace(".jpg", ""))
        records.append(
            {
                "data_source": "coco_res_opd",
                "prompt": np.array(PROMPT_TEMPLATE, dtype=object),
                "images": np.array([{"path": str(img_path)}], dtype=object),
                "hires_images": np.array([{"path": str(img_path)}], dtype=object),
                "ability": "image_captioning",
                "reward_model": {"style": "none", "ground_truth": ""},
                "extra_info": {
                    "image_id": image_id,
                    "file_name": file_name,
                },
            }
        )

    dataframe = pd.DataFrame(records)
    output_path = DATA_DIR / "train_5k.parquet"
    dataframe.to_parquet(output_path, index=False)
    print(f"[Train] Saved {len(dataframe)} samples to {output_path}")
    return output_path


def build_test_set():
    """Sample 1500 images from val2017 with captions & objects, matching existing test.json format."""
    print(f"[Test] Loading val2017 annotations ...")

    with open(VAL_ANNOTATIONS / "captions_val2017.json") as caption_file:
        captions_data = json.load(caption_file)

    with open(VAL_ANNOTATIONS / "instances_val2017.json") as instances_file:
        instances_data = json.load(instances_file)

    image_id_to_captions = {}
    for caption_entry in captions_data["annotations"]:
        image_id = caption_entry["image_id"]
        image_id_to_captions.setdefault(image_id, []).append(caption_entry["caption"])

    image_id_to_objects = {}
    category_map = {cat["id"]: cat["name"] for cat in instances_data["categories"]}
    for annotation in instances_data["annotations"]:
        image_id = annotation["image_id"]
        category_name = category_map.get(annotation["category_id"], "unknown")
        image_id_to_objects.setdefault(image_id, set()).add(category_name)

    image_id_to_filename = {
        img["id"]: img["file_name"] for img in captions_data["images"]
    }

    valid_image_ids = [
        image_id
        for image_id in image_id_to_filename
        if (VAL_IMG_DIR / image_id_to_filename[image_id]).exists()
    ]
    print(f"[Test] Found {len(valid_image_ids)} valid val2017 images")

    random.seed(SEED)
    sampled_ids = random.sample(valid_image_ids, min(TEST_SAMPLE_SIZE, len(valid_image_ids)))
    sampled_ids.sort()

    records = []
    for image_id in sampled_ids:
        file_name = image_id_to_filename[image_id]
        image_path = str(VAL_IMG_DIR / file_name)
        captions = image_id_to_captions.get(image_id, [])
        objects = sorted(image_id_to_objects.get(image_id, set()))

        records.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "image_path": image_path,
                "captions": captions,
                "objects": objects,
            }
        )

    output_path = DATA_DIR / "test_1000.json"
    with open(output_path, "w") as output_file:
        json.dump(records, output_file, indent=2, ensure_ascii=False)
    print(f"[Test] Saved {len(records)} samples to {output_path}")
    return output_path


def build_val_set():
    """Sample 100 images from val2017 for mini-eval, parquet format matching existing val.parquet."""
    print(f"[Val] Loading val2017 annotations ...")

    with open(VAL_ANNOTATIONS / "captions_val2017.json") as caption_file:
        captions_data = json.load(caption_file)
    with open(VAL_ANNOTATIONS / "instances_val2017.json") as instances_file:
        instances_data = json.load(instances_file)

    image_id_to_captions = {}
    for caption_entry in captions_data["annotations"]:
        image_id = caption_entry["image_id"]
        image_id_to_captions.setdefault(image_id, []).append(caption_entry["caption"])

    image_id_to_objects = {}
    category_map = {cat["id"]: cat["name"] for cat in instances_data["categories"]}
    for annotation in instances_data["annotations"]:
        image_id = annotation["image_id"]
        category_name = category_map.get(annotation["category_id"], "unknown")
        image_id_to_objects.setdefault(image_id, set()).add(category_name)

    image_id_to_filename = {img["id"]: img["file_name"] for img in captions_data["images"]}

    valid_image_ids = [
        image_id
        for image_id in image_id_to_filename
        if (VAL_IMG_DIR / image_id_to_filename[image_id]).exists()
    ]
    print(f"[Val] Found {len(valid_image_ids)} valid val2017 images")

    random.seed(SEED + 1)  # Different seed from test set to avoid overlap
    sampled_ids = random.sample(valid_image_ids, min(VAL_SAMPLE_SIZE, len(valid_image_ids)))
    sampled_ids.sort()

    records = []
    for image_id in sampled_ids:
        file_name = image_id_to_filename[image_id]
        img_path = str(VAL_IMG_DIR / file_name)
        caps = image_id_to_captions.get(image_id, [])
        objs = sorted(image_id_to_objects.get(image_id, set()))
        gt_caption = caps[0] if caps else ""

        records.append({
            "data_source": "coco_res_opd",
            "prompt": np.array(PROMPT_TEMPLATE, dtype=object),
            "images": np.array([{"path": img_path}], dtype=object),
            "hires_images": np.array([{"path": img_path}], dtype=object),
            "ability": "image_captioning",
            "reward_model": {"style": "none", "ground_truth": gt_caption},
            "extra_info": {
                "image_id": image_id,
                "file_name": file_name,
                "captions": np.array(caps, dtype=object),
                "objects": np.array(objs, dtype=object),
                "question": "",
            },
        })

    dataframe = pd.DataFrame(records)
    output_path = DATA_DIR / "val.parquet"
    dataframe.to_parquet(output_path, index=False)
    print(f"[Val] Saved {len(dataframe)} samples to {output_path}")
    return output_path


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = build_train_set()
    val_path = build_val_set()
    test_path = build_test_set()
    print(f"\n✅ Done!")
    print(f"   Train: {train_path}")
    print(f"   Val:   {val_path}")
    print(f"   Test:  {test_path}")
