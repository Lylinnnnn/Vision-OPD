import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


def load_reward_score_module():
    module_path = Path(__file__).resolve().parents[2] / "verl" / "utils" / "reward_score" / "__init__.py"
    import_utils = types.ModuleType("verl.utils.import_utils")
    import_utils.deprecated = lambda _: lambda function: function
    fake_modules = {
        "verl": types.ModuleType("verl"),
        "verl.utils": types.ModuleType("verl.utils"),
        "verl.utils.import_utils": import_utils,
    }
    spec = importlib.util.spec_from_file_location("res_opd_reward_score_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
    return module


reward_score = load_reward_score_module()


class ChairRewardScoreTest(unittest.TestCase):
    def test_reference_captions_from_validation_metadata_are_used(self):
        result = reward_score.default_compute_score(
            data_source="coco_res_opd",
            solution_str="A dog next to a bench.",
            ground_truth="A dog next to a bench.",
            extra_info={
                "image_id": 776,
                "objects": np.array(["dog"], dtype=object),
                "captions": np.array(["A dog beside a bench."], dtype=object),
            },
        )
        self.assertEqual(result["chair_i"], 0.0)
        self.assertEqual(result["obj_f1"], 1.0)

    def test_missing_reference_captions_still_fail(self):
        with self.assertRaisesRegex(ValueError, "reference captions"):
            reward_score.default_compute_score(
                data_source="coco_res_opd",
                solution_str="A dog.",
                ground_truth="A dog.",
                extra_info={"image_id": 776, "objects": ["dog"]},
            )


if __name__ == "__main__":
    unittest.main()
