import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "tmp"
    / "summarize_paper_chair_accounting.py"
)
SPEC = importlib.util.spec_from_file_location("paper_chair_accounting", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PaperChairAccountingTest(unittest.TestCase):
    def test_add_base_deltas_uses_same_scale_base(self):
        base = {
            "scale": "2b",
            "label": "2b/base/original",
            "response_tokens_mean": 100.0,
            "official_correct_mentions": 1000,
            "official_hallucinated_mentions": 100,
            "official_correct_mentions_per_caption": 1.0,
            "official_hallucinated_mentions_per_caption": 0.1,
            "CHAIRi": 0.1,
            "CHAIRs": 0.2,
            "ObjRecall": 0.7,
            "ObjF1": 0.75,
        }
        method = {
            "scale": "2b",
            "label": "2b/riskmask/tr075",
            "response_tokens_mean": 90.0,
            "official_correct_mentions": 1010,
            "official_hallucinated_mentions": 80,
            "official_correct_mentions_per_caption": 1.01,
            "official_hallucinated_mentions_per_caption": 0.08,
            "CHAIRi": 0.08,
            "CHAIRs": 0.18,
            "ObjRecall": 0.71,
            "ObjF1": 0.76,
        }
        rows = [base, method]
        MODULE.add_base_deltas(rows)
        self.assertEqual(method["base_label"], "2b/base/original")
        self.assertEqual(method["delta_official_correct_mentions"], 10)
        self.assertEqual(method["delta_official_hallucinated_mentions"], -20)
        self.assertEqual(method["delta_response_tokens_mean"], -10.0)

    def test_validate_metrics_checks_official_totals(self):
        metrics = {
            "metric_schema": MODULE.OFFICIAL_SCHEMA,
            "total_captions": 2,
            "invalid_final_answer": 0,
            "eval_prompt": MODULE.OFFICIAL_PROMPT,
            "total_object_mentions": 10,
            "total_correct_mentions": 8,
            "total_hallucinated_mentions": 2,
            "hallucinated_captions": 1,
            "CHAIRi": 0.2,
            "CHAIRs": 0.5,
        }

        class Args:
            expected_samples = 2
            expected_prompt = MODULE.OFFICIAL_PROMPT

        MODULE.validate_metrics(metrics, [{}, {}], Args(), Path("metrics.json"))
        metrics["total_hallucinated_mentions"] = 3
        with self.assertRaisesRegex(ValueError, "do not add up"):
            MODULE.validate_metrics(metrics, [{}, {}], Args(), Path("metrics.json"))

    def test_final_response_and_word_length(self):
        row = {"generated_caption": "reasoning</think> A red car is parked."}
        text = MODULE.final_response_text(row)
        length, source = MODULE.response_length(row, text, "words", None)
        self.assertEqual(text, "A red car is parked.")
        self.assertEqual(length, 5)
        self.assertEqual(source, "word_regex")


if __name__ == "__main__":
    unittest.main()
