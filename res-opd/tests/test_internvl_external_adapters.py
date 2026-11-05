import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


try:
    import torch
except ImportError:
    torch = None


BASELINES = Path(__file__).resolve().parents[1] / "baselines"


@unittest.skipIf(torch is None, "torch is not installed")
class InternVLInputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "internvl3_inputs_test", BASELINES / "internvl3_inputs.py"
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_single_image_alignment_and_dtype(self):
        class Processor:
            image_token_id = 9
            image_seq_length = 2

            def apply_chat_template(self, *args, **kwargs):
                return {
                    "input_ids": torch.tensor([[1, 9, 9, 9, 9, 2]]),
                    "attention_mask": torch.ones(1, 6, dtype=torch.long),
                    "pixel_values": torch.ones(2, 3, 4, 4),
                }

        result = self.module.build_inputs(Processor(), object(), "Describe the image.",
                                          torch.device("cpu"), torch.bfloat16)
        self.assertEqual(result["pixel_values"].dtype, torch.bfloat16)
        self.assertEqual(result["input_ids"].dtype, torch.long)

    def test_alignment_rejects_missing_image_tokens(self):
        processor = SimpleNamespace(image_token_id=9, image_seq_length=2)
        batch = {"input_ids": torch.tensor([[1, 9, 2]]),
                 "pixel_values": torch.ones(2, 3, 4, 4)}
        with self.assertRaisesRegex(ValueError, "alignment mismatch"):
            self.module.validate_image_alignment(batch, processor)

    def test_vcd_only_noises_second_visual_branch(self):
        class Processor:
            image_token_id = 9
            image_seq_length = 2

            def apply_chat_template(self, *args, **kwargs):
                return {
                    "input_ids": torch.tensor([[1, 9, 9, 9, 9, 2], [1, 9, 9, 9, 9, 2]]),
                    "attention_mask": torch.ones(2, 6, dtype=torch.long),
                    "pixel_values": torch.ones(4, 3, 4, 4),
                }

        noise_module = ModuleType("vcd_utils.vcd_add_noise")
        noise_module.add_diffusion_noise = lambda value, step: value + step / 1000
        with mock.patch.dict(sys.modules, {
            "vcd_utils": ModuleType("vcd_utils"),
            "vcd_utils.vcd_add_noise": noise_module,
        }):
            batch = self.module.build_vcd_inputs(
                Processor(), object(), "Describe the image.", 500, 42
            )
        self.assertTrue(torch.equal(batch["pixel_values"][:2], torch.ones(2, 3, 4, 4)))
        self.assertTrue(torch.equal(batch["pixel_values"][2:], torch.full((2, 3, 4, 4), 1.5)))

    def test_continuation_preserves_visual_inputs(self):
        class Model:
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(scores=[torch.tensor([[0.2, 0.8]])])

        model = Model()
        batch = {"input_ids": torch.tensor([[1, 9, 9, 2]]),
                 "attention_mask": torch.ones(1, 4, dtype=torch.long),
                 "pixel_values": torch.ones(1, 3, 4, 4)}
        logits = self.module.next_token_logits(model, batch, [3, 4])
        self.assertAlmostEqual(float(logits[0]), 0.2)
        self.assertAlmostEqual(float(logits[1]), 0.8)
        self.assertEqual(model.kwargs["input_ids"].tolist(), [[1, 9, 9, 2, 3, 4]])
        self.assertEqual(batch["input_ids"].shape[-1], 4)


class InternVLAdapterContractTest(unittest.TestCase):
    def test_family_routing_and_isolated_results(self):
        for method in ("vcd", "opera", "halc"):
            evaluator = (BASELINES / f"eval_{method}_chair.py").read_text()
            runner = (BASELINES / f"run_{method}_chair_sharded_8gpu.sh").read_text()
            self.assertIn('"internvl3"', evaluator)
            self.assertIn("MODEL_FAMILY", runner)
            self.assertIn("internvl3", runner)
            self.assertIn("--model-family", runner)

    def test_halc_uses_model_specific_input_and_logits(self):
        source = (BASELINES / "halc_qwen3vl.py").read_text()
        self.assertIn("self.input_builder", source)
        self.assertIn("self.logits_fn", source)
        self.assertIn("internvl_next_token_logits", (BASELINES / "eval_halc_chair.py").read_text())


if __name__ == "__main__":
    unittest.main()
