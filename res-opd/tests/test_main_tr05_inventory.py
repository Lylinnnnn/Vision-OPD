import runpy
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tmp/summarize_main_tr05_inventory.py"
MODULE = runpy.run_path(str(SCRIPT))


class MainTr05InventoryTest(unittest.TestCase):
    def test_vision_judge_json_array(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            judge = directory / "mmstar/judge/model_answer.jsonl"
            judge.parent.mkdir(parents=True)
            judge.write_text(json.dumps([{"judge": "Yes"}] * 1500), encoding="utf-8")
            result = MODULE["vision_result"](directory, "mmstar")
            self.assertEqual(result[0]["MMStar"], 1.0)

    def test_chair_accounting_shares_main_table(self):
        row = {
            "spec": MODULE["Spec"]("Qwen3VL", "2B", "Base", "Qwen3VL-2B-Instruct"),
            "step": None,
            "checkpoint": "Base",
            "values": {"CHAIRi": 0.1, "CHAIRs": 0.2, "ObjF1": 0.7,
                       "words_per_image": 53.5, "chars_per_image": 312.0,
                       "correct_mentions": 1234, "hall_mentions": 56},
            "sources": {},
            "status": "待补推理",
        }
        rendered = MODULE["render"]([row], [], [], False, Path("/missing"))
        self.assertNotIn("## CHAIR 回答长度与对象计数", rendered)
        lines = rendered.splitlines()
        header = next(line for line in lines if line.startswith("| ID | 家族 |"))
        data = next(line for line in lines if line.startswith("| M001 |"))
        self.assertEqual(len(header.split("|")), len(data.split("|")))
        self.assertIn("| 53.50 | 312.00 | 1234 | 56 |", data)

    def test_presentation_halc_is_not_added_to_formal_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "eval_results"
            formal = (root / "external_baselines/full/Qwen3VL-8B-Instruct" /
                      "train5000_test1000_original_sr1p0" /
                      "chair_external_halc_qwen3vl_adapted_official")
            formal.mkdir(parents=True)
            (formal / "halc_config.json").write_text(json.dumps({
                "model_path": "/models/Qwen3VL-8B-Instruct"}), encoding="utf-8")
            presentation = formal / "presentation_mixed_8192_2048"
            presentation.mkdir()
            (presentation / "halc_config.json").write_text(json.dumps({
                "model_path": "/models/Qwen3VL-8B-Instruct",
                "presentation_only": True, "not_for_paper": True}), encoding="utf-8")
            issues = []
            rows = MODULE["external_rows"](issues, root)
            formal_rows = [row for row in rows if row["spec"].size == "8B" and
                           row["spec"].method == "HALC (external)"]
            self.assertEqual(len(formal_rows), 1)
            self.assertEqual(formal_rows[0]["dataset"], str(formal))

            rows = MODULE["external_rows"](issues, root, include_presentation_halc=True)
            mixed_rows = [row for row in rows if row["spec"].size == "8B" and
                          row["spec"].method == "HALC (external)" and
                          row["dataset"] == str(presentation)]
            self.assertEqual(len(mixed_rows), 1)
            self.assertEqual(mixed_rows[0]["status"], "混合解码展示，待补推理")


if __name__ == "__main__":
    unittest.main()
