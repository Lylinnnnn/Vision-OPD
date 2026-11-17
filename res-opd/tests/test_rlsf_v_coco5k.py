import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "baselines/rlsf_v"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_coco5k")
audit = load_script("audit_generation")
materialize = load_script("materialize")


class RlsfCoco5kTests(unittest.TestCase):
    def test_converter_keeps_training_prompt_and_mounted_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "COCO/train2017/000000000001.jpg"
            image.parent.mkdir(parents=True)
            image.touch()
            frame = pd.DataFrame([{
                "images": [{"path": "/stale/000000000001.jpg"}],
                "extra_info": {"image_id": 1},
                "prompt": [{"role": "user", "content": "<image>Describe this image."}],
            }])
            rows = prepare.convert_frame(frame, {2}, root)
            self.assertEqual(rows, [{
                "image_id": "1", "image_path": str(image),
                "question": "Describe this image.",
            }])
            with self.assertRaisesRegex(ValueError, "test-set"):
                prepare.convert_frame(frame, {1}, root)

    def test_generation_audit_rejects_missing_samples(self):
        questions = [{"image_id": "1"}]
        answers = [{"question_id": "1", "answer_id": "a", "text": "A dog."}]
        self.assertEqual(audit.audit(questions, answers, 1), (1, 1))
        with self.assertRaisesRegex(ValueError, "Incomplete generation"):
            audit.audit(questions, answers, 2)

    def test_preference_matches_original_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "000000000001.jpg"
            image.touch()
            questions = root / "questions.jsonl"
            questions.write_text(json.dumps({
                "image_id": "1", "image_path": str(image), "question": "Describe this image."
            }) + "\n")
            preferences = root / "dpo.json"
            preferences.write_text(json.dumps([{
                "img_path": str(image), "prompt": "Describe this image.",
                "chosen": "A dog.", "rejected": "A cat.",
            }]))
            self.assertEqual(len(materialize.load_preferences(preferences, questions)), 1)
            preferences.write_text(json.dumps([{
                "img_path": str(image), "prompt": "Different prompt.",
                "chosen": "A dog.", "rejected": "A cat.",
            }]))
            with self.assertRaisesRegex(ValueError, "not from the COCO 5k"):
                materialize.load_preferences(preferences, questions)


if __name__ == "__main__":
    unittest.main()
