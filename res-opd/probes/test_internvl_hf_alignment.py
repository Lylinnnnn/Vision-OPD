#!/usr/bin/env python3
"""Small dependency-free checks for the InternVL rollout prompt bridge."""

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "verl/utils/internvl_hf.py"
SPEC = importlib.util.spec_from_file_location("internvl_hf_alignment", MODULE_PATH)
internvl_hf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(internvl_hf)


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Mask:
    def __init__(self, values):
        self.values = values

    def sum(self):
        return Scalar(sum(self.values))


class TokenIds:
    def __init__(self, values):
        self.values = values

    def __eq__(self, token_id):
        return Mask([value == token_id for value in self.values])


class InternVLProcessor:
    image_token = "<IMG_CONTEXT>"
    image_token_id = 151667
    image_seq_length = 4
    config = type("Config", (), {"model_type": "internvl"})()


class InternVLAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.processor = InternVLProcessor()

    def test_vllm_placeholder_preserves_text_and_image_order(self):
        prompt = "before <IMG_CONTEXT> between <IMG_CONTEXT> after"
        self.assertEqual(
            internvl_hf.internvl_vllm_prompt(prompt, self.processor, 2),
            "before <image> between <image> after",
        )

    def test_prompt_rejects_wrong_image_count(self):
        with self.assertRaisesRegex(ValueError, "image markers"):
            internvl_hf.internvl_vllm_prompt("<IMG_CONTEXT>", self.processor, 2)

    def test_prompt_rejects_existing_placeholder(self):
        with self.assertRaisesRegex(ValueError, "already contains"):
            internvl_hf.internvl_vllm_prompt("<image><IMG_CONTEXT>", self.processor, 1)

    def test_visual_token_alignment(self):
        tokens = TokenIds([151667] * 8 + [1])
        pixels = type("Pixels", (), {"shape": (2, 3, 4, 4)})()
        internvl_hf.assert_internvl_image_alignment(tokens, pixels, self.processor, "test")

    def test_visual_token_mismatch_fails(self):
        tokens = TokenIds([151667] * 7)
        pixels = type("Pixels", (), {"shape": (2, 3, 4, 4)})()
        with self.assertRaisesRegex(ValueError, "7 image context tokens for 8 visual features"):
            internvl_hf.assert_internvl_image_alignment(tokens, pixels, self.processor, "test")

    def test_rollout_accepts_only_exact_placeholder_or_expansion(self):
        compact = [1, 2, 3]
        expanded = [1, 4, 4, 4, 2, 3]
        check = internvl_hf.assert_internvl_rollout_prompt_alignment
        self.assertEqual(check(compact, compact, expanded), "placeholder")
        self.assertEqual(check(expanded, compact, expanded), "expanded")
        with self.assertRaisesRegex(ValueError, "prompt mismatch"):
            check([1, 4, 2, 3], compact, expanded)
        with self.assertRaisesRegex(ValueError, "prompt mismatch"):
            check(None, compact, expanded)

    def test_processor_is_not_mistaken_for_qwen(self):
        self.assertTrue(internvl_hf.is_internvl_hf_processor(self.processor))
        self.assertFalse(internvl_hf.is_internvl_hf_processor(None))


if __name__ == "__main__":
    unittest.main()
