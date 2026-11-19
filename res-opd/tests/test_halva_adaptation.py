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

    def test_annotated_object_clause_preserves_caption_when_name_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "000000000002.jpg"
            image.touch()
            questions = [{"image_id": "2", "image_path": str(image), "question": "Describe the image."}]
            captions = {"annotations": [{"image_id": 2, "caption": "A puppy sleeps on grass."}]}
            instances = {
                "categories": [
                    {"id": 1, "name": "dog", "supercategory": "animal"},
                    {"id": 2, "name": "cat", "supercategory": "animal"},
                ],
                "annotations": [{"image_id": 2, "category_id": 1}],
            }
            pair, = prepare.build(questions, captions, instances)
            self.assertEqual(pair["chosen"], "A puppy sleeps on grass. Visible object: dog.")
            self.assertEqual(pair["rejected"], "A puppy sleeps on grass. Visible object: cat.")
            self.assertEqual(pair["strategy"], "annotated_object_clause_added")

    def test_repeated_object_name_uses_single_phrase_clause(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "000000000003.jpg"
            image.touch()
            questions = [{"image_id": "3", "image_path": str(image), "question": "Describe the image."}]
            captions = {"annotations": [{"image_id": 3, "caption": "A dog and another dog play."}]}
            instances = {
                "categories": [
                    {"id": 1, "name": "dog", "supercategory": "animal"},
                    {"id": 2, "name": "cat", "supercategory": "animal"},
                ],
                "annotations": [{"image_id": 3, "category_id": 1}],
            }
            pair, = prepare.build(questions, captions, instances)
            self.assertEqual(pair["chosen"], "Visible object: dog.")
            self.assertEqual(pair["rejected"], "Visible object: cat.")
            self.assertEqual(pair["strategy"], "annotated_object_clause_only")

    def test_explicit_partial_count_allows_only_unannotated_images(self):
        with tempfile.TemporaryDirectory() as directory:
            images = [Path(directory) / f"{image_id:012d}.jpg" for image_id in (4, 5)]
            for image in images:
                image.touch()
            questions = [
                {"image_id": str(image_id), "image_path": str(image), "question": "Describe the image."}
                for image_id, image in zip((4, 5), images)
            ]
            captions = {"annotations": [
                {"image_id": 4, "caption": "A dog rests."},
                {"image_id": 5, "caption": "An empty scene."},
            ]}
            instances = {
                "categories": [
                    {"id": 1, "name": "dog", "supercategory": "animal"},
                    {"id": 2, "name": "cat", "supercategory": "animal"},
                ],
                "annotations": [{"image_id": 4, "category_id": 1}],
            }
            with self.assertRaisesRegex(ValueError, "Only 1/2"):
                prepare.build(questions, captions, instances)
            pair, = prepare.build(questions, captions, instances, expected_pairs=1)
            self.assertEqual(pair["image_id"], 4)
            with self.assertRaisesRegex(ValueError, "expected_pairs=2"):
                prepare.build(questions, captions, instances, expected_pairs=2)

            instances["annotations"].append({"image_id": 5, "category_id": 1})
            with self.assertRaisesRegex(ValueError, "without_instances=0"):
                prepare.build(questions, captions, instances, expected_pairs=1)

    def test_phrase_span_in_response(self):
        ids = [ord(char) for char in "prompt A dog rests."]
        self.assertEqual(training.phrase_token_positions(Tokenizer(), ids, "dog", 7), (8, 9, 10, 11))
        with self.assertRaisesRegex(ValueError, "got 0"):
            training.phrase_token_positions(Tokenizer(), ids, "cat", 7)


if __name__ == "__main__":
    unittest.main()
