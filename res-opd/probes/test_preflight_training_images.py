#!/usr/bin/env python3
"""Check path extraction without loading the training dataset."""

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("preflight_training_images.py")
SPEC = importlib.util.spec_from_file_location("preflight_training_images", MODULE_PATH)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class ImagePathTest(unittest.TestCase):
    def test_string_and_path_entries(self):
        self.assertEqual(list(preflight.image_paths(["/a.jpg", Path("/b.jpg")])), ["/a.jpg", "/b.jpg"])

    def test_parquet_style_dict_entries(self):
        values = [{"bytes": None, "path": "/a.jpg"}, {"image": {"path": "/b.jpg"}}]
        self.assertEqual(list(preflight.image_paths(values)), ["/a.jpg", "/b.jpg"])

    def test_embedded_bytes_do_not_require_paths(self):
        self.assertEqual(list(preflight.image_paths({"bytes": b"image", "path": "/missing.jpg"})), [])


if __name__ == "__main__":
    unittest.main()
