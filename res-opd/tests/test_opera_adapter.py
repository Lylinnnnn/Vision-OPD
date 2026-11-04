import importlib.util
import sys
import unittest
from pathlib import Path


try:
    import torch
except ImportError:
    torch = None


MODULE_PATH = Path(__file__).resolve().parents[1] / "baselines" / "opera_qwen3vl.py"


@unittest.skipIf(torch is None, "torch is not installed")
class OperaAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("opera_qwen3vl", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.module = module

    def test_locate_image_span(self):
        span = self.module.locate_image_span(torch.tensor([[1, 9, 9, 9, 2]]), 9)
        self.assertEqual(
            span,
            {"image_start": 1, "image_end": 3, "response_start": 5},
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.module.locate_image_span(torch.tensor([[1, 9, 2, 9]]), 9)

    def test_first_ten_tokens_reward_image_attention(self):
        first = torch.tensor(
            [[[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.8, 0.1, 0.1]]]]
        )
        second = torch.tensor(
            [[[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.2, 0.4, 0.4]]]]
        )
        adjusted, locations, response_length = self.module.compute_opera_candidate_scores(
            torch.tensor([[1.0, 1.0]]),
            [first, second],
            {"image_start": 0, "image_end": 0, "response_start": 2},
        )
        self.assertEqual(response_length, 1)
        self.assertTrue(torch.equal(locations, torch.zeros_like(locations)))
        self.assertAlmostEqual(float(adjusted[0, 0]), 1.8, places=5)
        self.assertAlmostEqual(float(adjusted[0, 1]), 1.2, places=5)

    def test_late_tokens_use_response_attention_chain(self):
        sequence_length = 12

        def attention_with_chain(local_position):
            attention = torch.zeros((1, 1, sequence_length, sequence_length))
            attention[..., 0] = 1.0
            absolute_position = 1 + local_position
            for row in range(absolute_position, sequence_length):
                attention[..., row, 0] = 0.5
                attention[..., row, absolute_position] = 0.5
            return attention

        adjusted, locations, response_length = self.module.compute_opera_candidate_scores(
            torch.tensor([[1.0, 1.0]]),
            [attention_with_chain(2), attention_with_chain(4)],
            {"image_start": 0, "image_end": 0, "response_start": 1},
        )
        self.assertEqual(response_length, 11)
        self.assertEqual(locations.tolist(), [[2, 4]])
        self.assertTrue(torch.all(adjusted < 1.0))

    def test_rollback_requires_every_beam_to_exceed_threshold(self):
        history = [(index, torch.tensor([12, 12])) for index in range(3)]
        self.assertTrue(self.module.should_rollback(history, 12, threshold=2))
        history[-1] = (2, torch.tensor([12, 8]))
        self.assertFalse(self.module.should_rollback(history, 12, threshold=2))


class OperaStaticContractTest(unittest.TestCase):
    def test_runner_uses_upstream_defaults_and_isolated_output(self):
        runner = MODULE_PATH.with_name("run_opera_chair_sharded_8gpu.sh").read_text()
        self.assertIn("OPERA_NUM_BEAMS:-5", runner)
        self.assertIn("OPERA_SCALE_FACTOR:-50.0", runner)
        self.assertIn("OPERA_THRESHOLD:-15", runner)
        self.assertIn("OPERA_NUM_ATTN_CANDIDATES:-5", runner)
        self.assertIn("chair_external_opera_official", runner)

    def test_evaluator_pins_supported_transformers_and_official_prompt(self):
        evaluator = MODULE_PATH.with_name("eval_opera_chair.py").read_text()
        self.assertIn('DEFAULT_PROMPT = "Describe the image."', evaluator)
        self.assertIn('transformers.__version__ != "5.5.0"', evaluator)
        self.assertIn('attn_implementation="eager"', evaluator)
        self.assertIn('"decoding_method": "opera"', evaluator)


if __name__ == "__main__":
    unittest.main()
