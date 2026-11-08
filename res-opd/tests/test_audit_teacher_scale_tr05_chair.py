import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tmp/audit_teacher_scale_tr05_chair.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("audit_teacher_scale_tr05_chair", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_rows(path, *, count=1000, prompt="Describe the image.", ratio=1.0, refs=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            row = {"image_id": index, "generated_caption": "A dog.",
                   "eval_prompt": prompt, "eval_student_ratio": ratio,
                   "gt_captions": ["A dog."] if refs else []}
            handle.write(json.dumps(row) + "\n")


class TeacherScaleChairAuditTest(unittest.TestCase):
    def test_no_generation_and_legacy_scoring_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            state, values, evidence = MODULE.inspect_chair([directory], directory, [], "test")
            self.assertEqual(state, "未推理")

            write_rows(directory / "eval_results.jsonl")
            (directory / "chair_metrics.json").write_text(json.dumps({"CHAIRi": 0.2}))
            state, values, evidence = MODULE.inspect_chair([directory], directory, [], "test")
            self.assertEqual(state, "已推理，待新版计分")
            self.assertIn("eval_results.jsonl", evidence)

    def test_partial_wrong_protocol_and_missing_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "eval_results.jsonl"
            write_rows(path, count=999)
            self.assertEqual(MODULE.inspect_raw(path, path.parent / "chair_metrics.json")[0], "推理不完整")
            write_rows(path, prompt="What is in the image?")
            self.assertEqual(MODULE.inspect_raw(path, path.parent / "chair_metrics.json")[0],
                             "已推理，协议待核实")
            write_rows(path, refs=False)
            self.assertEqual(MODULE.inspect_raw(path, path.parent / "chair_metrics.json")[0],
                             "已推理，参考标注待核实")

    def test_separate_official_candidate_is_not_called_unscored(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_rows(directory / "eval_results.jsonl")
            (directory / "chair_metrics_official.json").write_text(json.dumps({"CHAIRi": 0.1}))
            state, values, evidence = MODULE.inspect_chair([directory], directory, [], "test")
            self.assertEqual(state, "已推理，新版候选指标待核实")
            self.assertIn("chair_metrics_official.json", evidence)

    def test_official_chair_takes_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_rows(directory / "eval_results.jsonl")
            metrics = {"metric_schema": MODULE.inventory.CHAIR_SCHEMA,
                       "eval_prompt": "Describe the image.",
                       "total_results": 1000, "total_captions": 1000,
                       "invalid_final_answer": 0, "total_correct_mentions": 1000,
                       "total_hallucinated_mentions": 0, "total_object_mentions": 1000,
                       "hallucinated_captions": 0, "CHAIRi": 0.0,
                       "CHAIRs": 0.0, "ObjF1": 1.0}
            (directory / "chair_metrics.json").write_text(json.dumps(metrics))
            state, values, evidence = MODULE.inspect_chair([directory], directory, [], "test")
            self.assertEqual(state, "新版指标齐全")
            self.assertEqual(values["ObjF1"], 1.0)

    def test_collect_checks_only_chair_for_actual_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oss = root / "oss"
            spec = MODULE.teacher_scale.experiment_spec("2B", "2B", "0.5")
            checkpoint = oss / spec.oss_name / "global_step_50"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"weight")
            (checkpoint / "config.json").write_text("{}")
            output = (root / "res-opd/eval_results/instruct/full" /
                      f"{spec.experiment}_global_step_50" / MODULE.inventory.DATASET / "chair_legacy")
            write_rows(output / "eval_results.jsonl")
            with patch.object(MODULE.inventory, "ROOT", root), \
                 patch.object(MODULE.teacher_scale, "table_specs", return_value=[
                     ("2B", "2B frozen", "tr=0.5", spec),
                 ]):
                rows, issues, mounted = MODULE.collect(oss)
            self.assertTrue(mounted)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ckpt_status"], "完整")
            self.assertEqual(rows[0]["chair_status"], "已推理，待新版计分")
            report = MODULE.render(rows, issues, mounted, oss)
            self.assertIn("## 已推理但无合规新版指标", report)
            self.assertNotIn("POPE F1", report)


if __name__ == "__main__":
    unittest.main()
