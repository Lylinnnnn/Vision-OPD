import unittest

from res_opd_chair_test_support import load_chair_module


chair = load_chair_module()


class OfficialChairTest(unittest.TestCase):
    def setUp(self):
        self.objects, self.inverse = chair.parse_official_synonyms()
        self.double_words = chair.build_double_word_dict()

    def score(self, records):
        per_sample = chair.compute_per_sample(
            records,
            self.objects,
            self.inverse,
            self.double_words,
        )
        return chair.aggregate_from_arrays(per_sample, list(per_sample))

    def test_chair_i_counts_repeated_mentions(self):
        metrics = self.score(
            [
                {
                    "image_id": 1,
                    "generated_text": "A dog and a cat cat.",
                    "gt_objects": ["dog"],
                    "gt_captions": ["A dog is outside."],
                },
                {
                    "image_id": 2,
                    "generated_text": "A cat.",
                    "gt_objects": ["cat"],
                    "gt_captions": ["A cat is sitting."],
                },
            ]
        )
        self.assertEqual(metrics["total_object_mentions"], 4)
        self.assertEqual(metrics["total_hallucinated_mentions"], 2)
        self.assertAlmostEqual(metrics["CHAIRi"], 0.5)
        self.assertAlmostEqual(metrics["CHAIRs"], 0.5)
        self.assertEqual(
            metrics["ground_truth_protocol"],
            "instance_categories_union_reference_caption_objects",
        )

    def test_chair_s_counts_captions_not_clauses(self):
        metrics = self.score(
            [
                {
                    "image_id": 1,
                    "generated_text": "A dog is outside. A cat is nearby.",
                    "gt_objects": ["dog"],
                    "gt_captions": ["A dog is outside."],
                }
            ]
        )
        self.assertAlmostEqual(metrics["CHAIRs"], 1.0)
        self.assertAlmostEqual(metrics["CHAIRi"], 0.5)

    def test_reference_caption_objects_are_ground_truth(self):
        metrics = self.score(
            [
                {
                    "image_id": 1,
                    "generated_text": "A dog and a bench.",
                    "gt_objects": ["dog"],
                    "gt_captions": ["A dog is standing beside a bench."],
                }
            ]
        )
        self.assertAlmostEqual(metrics["CHAIRi"], 0.0)
        self.assertAlmostEqual(metrics["CHAIRs"], 0.0)

    def test_official_singularization_keeps_coco_objects(self):
        _, canonical = chair.caption_to_words(
            "Horses stand near vases, scissors, and baseball gloves.",
            self.objects,
            self.inverse,
            self.double_words,
        )
        self.assertEqual(canonical, ["horse", "vase", "scissors", "baseball glove"])

    def test_official_mode_rejects_missing_reference_captions(self):
        with self.assertRaisesRegex(ValueError, "reference captions"):
            self.score(
                [
                    {
                        "image_id": 1,
                        "generated_text": "A dog.",
                        "gt_objects": ["dog"],
                        "gt_captions": [],
                    }
                ]
            )

    def test_duplicate_image_id_is_rejected(self):
        record = {
            "image_id": 1,
            "generated_text": "A dog.",
            "gt_objects": ["dog"],
            "gt_captions": ["A dog."],
        }
        with self.assertRaisesRegex(ValueError, "one generated caption"):
            self.score([record, dict(record)])


if __name__ == "__main__":
    unittest.main()
