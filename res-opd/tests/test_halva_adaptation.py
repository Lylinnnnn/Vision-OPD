"""CPU-only checks for the matched HALVA data and phrase boundaries."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1] / "baselines" / "halva"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare")
training = load("train")


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class HalvaAdaptationTest(unittest.TestCase):
    def test_matched_pair_uses_absent_same_group_object(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "000000000001.jpg"
            image.touch()
            questions = [{"image_id": "1", "image_path": str(image), "question": "Describe the image."}]
            captions = {"annotations": [{"image_id": 1, "caption": "A dog rests on grass."}]}
            instances = {
                "categories": [
                    {"id": 1, "name": "dog", "supercategory": "animal"},
                    {"id": 2, "name": "cat", "supercategory": "animal"},
                ],
                "annotations": [{"image_id": 1, "category_id": 1}],
            }
            pair, = prepare.build(questions, captions, instances)
            self.assertEqual(pair["chosen"], "A dog rests on grass.")
            self.assertEqual(pair["rejected"], "A cat rests on grass.")
            self.assertEqual(pair["chosen_phrase"], "dog")
            self.assertEqual(pair["rejected_phrase"], "cat")

    def test_missing_pair_fails_instead_of_silently_shortening_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "000000000001.jpg"
            image.touch()
            questions = [{"image_id": "1", "image_path": str(image), "question": "Describe the image."}]
            with self.assertRaisesRegex(ValueError, "Only 0/1"):
                prepare.build(questions, {"annotations": []}, {"categories": [], "annotations": []})

    def test_phrase_span_in_response(self):
        ids = [ord(char) for char in "prompt A dog rests."]
        self.assertEqual(training.phrase_token_positions(Tokenizer(), ids, "dog", 7), (8, 9, 10, 11))
        with self.assertRaisesRegex(ValueError, "got 0"):
            training.phrase_token_positions(Tokenizer(), ids, "cat", 7)


if __name__ == "__main__":
    unittest.main()
