import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tmp/summarize_three_paper_tables.py"
SPEC = importlib.util.spec_from_file_location("summarize_three_paper_tables", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ThreeTableSummaryTests(unittest.TestCase):
    def test_official_chair_length_and_mentions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "chair_official_v2"
            output.mkdir()
            metrics = {
                "metric_schema": MODULE.SCHEMA, "eval_prompt": MODULE.PROMPT,
                "total_captions": 2, "total_results": 2,
                "invalid_final_answer": 0, "total_correct_mentions": 3,
                "total_hallucinated_mentions": 1, "total_object_mentions": 4,
                "CHAIRi": 0.25, "CHAIRs": 0.5, "hallucinated_captions": 1,
                "ObjF1": 0.8, "ObjRecall": 0.75, "total_correct": 3,
            }
            (output / "chair_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            records = [
                {"image_id": 1, "generated_caption": "<think>ignore</think>Red cat.",
                 "eval_prompt": MODULE.PROMPT, "eval_student_ratio": 1.0},
                {"image_id": 2, "generated_caption": "A blue dog", "eval_prompt": MODULE.PROMPT,
                 "eval_student_ratio": 1.0},
            ]
            (output / "eval_results.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
            result = MODULE.valid_chair(output, "1.0", 2)
            self.assertEqual(result["response_words_mean"], 2.5)
            self.assertEqual(result["response_chars_mean"], (len("Red cat.") + len("A blue dog")) / 2)
            self.assertEqual(result["gt_correct_mentions"], 3)
            self.assertEqual(result["hall_mentions"], 1)
            self.assertEqual(result["unique_gt_categories"], 4)
            self.assertIsNone(MODULE.valid_chair(output, "0.5", 2))

    def test_step_discovery_and_table_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name = MODULE.experiment_name("2b", "0.5", rho=30)
            for step in (150, 50):
                (root / f"{name}_global_step_{step}" / "train5000_test1000_original_sr1p0").mkdir(parents=True)
            rows, _ = MODULE.collect(root, samples=2)
            chosen = [row for row in rows if row["table"] == "rho" and row["rho"] == 30]
            self.assertEqual([row["step"] for row in chosen], [50, 150])
            md = MODULE.markdown(rows, [])
            self.assertIn("Table 2 - Native", md)
            self.assertIn("Table 3 - Teacher", md)
            self.assertIn("Table 4 - Top-rho", md)
            self.assertIn("Words/img", md)
            self.assertIn("Response chars/img", md)
            self.assertIn("Hall mentions", md)

    def test_judge_json_array_and_recent_output_priority(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp)
            output = dataset / "three_tables_fill_official_v2"
            judge = output / "mmstar/judge/model_answer.jsonl"
            judge.parent.mkdir(parents=True)
            judge.write_text(json.dumps([{"judge": "Yes"}, {"judge": "No"}]), encoding="utf-8")
            self.assertEqual(MODULE.result_dirs(dataset)[0], output)
            self.assertEqual(MODULE.judge_accuracy(output, "mmstar")[0], 0.5)


if __name__ == "__main__":
    unittest.main()
