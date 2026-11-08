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


if __name__ == "__main__":
    unittest.main()
