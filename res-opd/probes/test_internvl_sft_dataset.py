#!/usr/bin/env python3
"""Regression checks for InternVL SFT image loading and processor setup."""

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "res-opd" / "datasets"))

from res_opd_sft_dataset import ResOPDSFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class GotOcr2ImageProcessor:
    pass


class InternVLProcessor:
    config = type("Config", (), {"model_type": "internvl"})()
    image_processor = GotOcr2ImageProcessor()


class Qwen3VLProcessor:
    image_processor = type("ImageProcessor", (), {"patch_size": 16})()


class InternVLSFTDatasetTest(unittest.TestCase):
    def test_missing_patch_size_does_not_prevent_dataset_creation(self):
        with patch.object(MultiTurnSFTDataset, "_download"), patch.object(
            MultiTurnSFTDataset, "_read_files_and_process"
        ):
            dataset = MultiTurnSFTDataset([], object(), {}, processor=InternVLProcessor())
        self.assertEqual(dataset.image_patch_size, 14)

    def test_internvl_original_image_is_not_qwen_resized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            image = Image.new("RGB", (19, 17), color=(22, 44, 66))
            image.save(path)
            dataset = object.__new__(ResOPDSFTDataset)
            dataset.messages_key = "messages"
            dataset.image_key = "images"
            dataset.video_key = "videos"
            dataset.processor = InternVLProcessor()
            dataset.image_patch_size = 14
            example = {
                "messages": [{"role": "user", "content": "<image> Describe it."}],
                "images": [{"path": str(path)}],
            }
            messages = dataset._build_messages(example)
            loaded = messages[0]["content"][0]["image"]
            self.assertEqual(loaded.size, image.size)
            self.assertEqual(loaded.getpixel((0, 0)), image.getpixel((0, 0)))
            self.assertEqual(example["messages"][0]["content"], "<image> Describe it.")
            repeated = dataset._build_messages(example)
            self.assertEqual(repeated[0]["content"][0]["image"].size, image.size)

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            loaded_bytes = dataset._load_internvl_image({"bytes": buffer.getvalue()})
            self.assertEqual(loaded_bytes.size, image.size)

    def test_qwen_path_still_uses_patch_aware_loader(self):
        dataset = object.__new__(ResOPDSFTDataset)
        dataset.messages_key = "messages"
        dataset.image_key = "images"
        dataset.video_key = "videos"
        dataset.processor = Qwen3VLProcessor()
        dataset.image_patch_size = 16
        expected = Image.new("RGB", (16, 16))
        with patch("res_opd_sft_dataset.process_image", return_value=expected) as loader:
            messages = dataset._build_messages({
                "messages": [{"role": "user", "content": "<image> Describe it."}],
                "images": [{"path": "example.jpg"}],
            })
        loader.assert_called_once_with({"path": "example.jpg"}, image_patch_size=16)
        self.assertIs(messages[0]["content"][0]["image"], expected)


if __name__ == "__main__":
    unittest.main()
