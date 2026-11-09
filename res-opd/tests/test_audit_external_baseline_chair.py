import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/tmp"))
from audit_external_baseline_chair import candidates, stage, statuses  # noqa: E402


class ExternalChairAuditTest(unittest.TestCase):
    def test_status_distinguishes_missing_partial_and_unscored(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.assertEqual(statuses(directory), ("未推理",) * 5)
            raw = directory / "eval_results.jsonl"
            raw.write_text('{"image_id": 1}\n', encoding="utf-8")
            self.assertEqual(statuses(directory)[0], "推理不完整(1/1000)")
            raw.write_text('{"image_id": 1}\n' * 1000, encoding="utf-8")
            self.assertEqual(statuses(directory)[0], "已推理，未算指标")
            (directory / "chair_metrics.json").write_text("{}", encoding="utf-8")
            self.assertEqual(statuses(directory)[0], "已推理，指标需核验/重算")

    def test_stage(self):
        self.assertEqual(stage(0, 1000, False), "未推理")
        self.assertEqual(stage(100, 1000, False), "推理不完整(100/1000)")
        self.assertEqual(stage(1000, 1000, False), "已推理，未算指标")
        self.assertEqual(stage(1000, 1000, False, True), "已推理，指标需核验/重算")
        self.assertEqual(stage(0, 1000, False, True), "指标在，原始推理缺失")
        self.assertEqual(stage(1000, 1000, True), "完成")

    def test_candidates_match_exact_model_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "external_baselines/full/Qwen3VL-4B-Instruct" / (
                "train5000_test1000_original_sr1p0/chair_external_opera_official")
            output.mkdir(parents=True)
            (output / "opera_config.json").write_text(
                json.dumps({"model_name": "Qwen3VL-4B-Instruct"}), encoding="utf-8")
            self.assertEqual(candidates(root, "Qwen3VL-4B-Instruct", "opera"), [output])
            self.assertEqual(candidates(root, "Qwen3VL-8B-Instruct", "opera"), [])


if __name__ == "__main__":
    unittest.main()
