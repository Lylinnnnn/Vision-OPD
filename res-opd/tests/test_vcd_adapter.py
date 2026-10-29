import importlib.util
import sys
import unittest
from pathlib import Path


try:
    import torch
except ImportError:
    torch = None


MODULE_PATH = Path(__file__).resolve().parents[1] / "baselines" / "vcd_qwen3vl.py"


@unittest.skipIf(torch is None, "torch is not installed")
class VCDLogitsProcessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("vcd_qwen3vl", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.processor_class = module.VCDLogitsProcessor

    def test_greedy_formula_and_plausibility_cutoff(self):
        processor = self.processor_class(
            alpha=1.0,
            beta=0.5,
            decode_mode="greedy",
        )
        scores = torch.tensor(
            [
                [4.0, 3.5, 1.0],
                [3.9, 0.0, -2.0],
            ]
        )
        output = processor(torch.zeros((2, 1), dtype=torch.long), scores)
        self.assertTrue(torch.equal(output[0], output[1]))
        self.assertEqual(int(output[0].argmax()), 1)
        self.assertTrue(torch.isneginf(output[0, 2]))

    def test_sampling_is_synchronized_and_reproducible(self):
        kwargs = {
            "alpha": 1.0,
            "beta": 0.1,
            "decode_mode": "sample",
            "temperature": 1.0,
            "seed": 123,
        }
        scores = torch.tensor([[2.0, 1.5, 1.0], [1.0, 1.0, 1.0]])
        first = self.processor_class(**kwargs)(
            torch.zeros((2, 1), dtype=torch.long), scores
        )
        second = self.processor_class(**kwargs)(
            torch.zeros((2, 1), dtype=torch.long), scores
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first[0], first[1]))
        self.assertEqual(int(torch.isfinite(first[0]).sum()), 1)

    def test_invalid_beta_is_rejected(self):
        with self.assertRaises(ValueError):
            self.processor_class(beta=0)


if __name__ == "__main__":
    unittest.main()
