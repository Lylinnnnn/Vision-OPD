import importlib.util
import sys
import unittest
from pathlib import Path

from PIL import Image


try:
    import torch
except ImportError:
    torch = None


MODULE_PATH = Path(__file__).resolve().parents[1] / "baselines" / "halc_qwen3vl.py"


@unittest.skipIf(torch is None, "torch is not installed")
class HALCAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("halc_qwen3vl", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.module = module

    def test_focal_crops_are_nonempty_and_expand(self):
        image = Image.new("RGB", (100, 80))
        crops = self.module.focal_crops(image, (0.2, 0.2, 0.8, 0.8), 4, 0.6)
        self.assertEqual(len(crops), 4)
        self.assertTrue(all(crop.width > 0 and crop.height > 0 for crop in crops))
        self.assertLess(crops[0].width, crops[1].width)
        self.assertLess(crops[1].width, crops[2].width)

    def test_bidirectional_contrast_proposes_four_tokens(self):
        views = [
            torch.tensor([8.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 8.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 8.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 8.0]),
        ]
        candidates = self.module.focal_contrast_candidates(views, 4, 0.05)
        self.assertEqual(len(candidates), 4)
        self.assertTrue(all(0 <= token < 4 for token in candidates))

    def test_invalid_config_rejected(self):
        with self.assertRaisesRegex(ValueError, "even"):
            self.module.HALCConfig(candidate_count=3).validate()


class HALCStaticContractTest(unittest.TestCase):
    def test_smoke_is_separate_from_paper_results(self):
        evaluator = MODULE_PATH.with_name("eval_halc_chair.py").read_text()
        self.assertIn("halc_qwen3vl_adaptation_smoke", evaluator)
        self.assertIn("--require-trigger", evaluator)
        self.assertIn("--require-change", evaluator)
        self.assertIn("compute_per_sample", evaluator)
        self.assertIn('"Describe the image."', evaluator)


if __name__ == "__main__":
    unittest.main()
