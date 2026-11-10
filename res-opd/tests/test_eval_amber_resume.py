import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "eval/eval_amber.py"
SPEC = importlib.util.spec_from_file_location("eval_amber_resume", SCRIPT)
amber = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(amber)


class AmberResumeTest(unittest.TestCase):
    def test_only_valid_raw_answers_are_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            official = root / "amber_a_responses.json"
            raw = root / "raw_results.jsonl"
            official.write_text(json.dumps([
                {"id": 1, "response": "No"},
                {"id": 2, "response": "stale"},
            ]))
            raw.write_text("\n".join(json.dumps(row) for row in (
                {"id": 1, "model_answer": "[ERROR] HTTP 500", "official_response": "No"},
                {"id": 2, "model_answer": "a cat", "official_response": "a cat"},
                {"id": 3, "model_answer": "a dog", "official_response": "a dog"},
                {"id": 2, "model_answer": "[ERROR] retry failed", "official_response": "No"},
            )) + "\n[]\n{unfinished")
            self.assertEqual(amber.load_existing(official, raw), {
                2: {"id": 2, "response": "a cat"},
                3: {"id": 3, "response": "a dog"},
            })

    def test_successful_retry_replaces_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw_results.jsonl"
            raw.write_text(
                '{"id":1,"model_answer":"[ERROR] failure","official_response":"No"}\n'
                '{"id":1,"model_answer":"Yes","official_response":"Yes"}\n'
            )
            self.assertEqual(amber.load_existing(root / "missing.json", raw), {
                1: {"id": 1, "response": "Yes"},
            })


if __name__ == "__main__":
    unittest.main()
