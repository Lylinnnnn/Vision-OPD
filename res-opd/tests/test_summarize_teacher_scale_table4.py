import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/tmp"
sys.path.insert(0, str(SCRIPT_DIR))
import summarize_teacher_scale_table4 as summary  # noqa: E402


def checkpoint(root, spec, step, complete=True):
    path = root / spec.oss_name / f"global_step_{step}"
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    if complete:
        (path / "model.safetensors").write_bytes(b"weights")
    return path


class TeacherScaleTable4Test(unittest.TestCase):
    def test_exact_seven_non_base_controls_and_final_steps(self):
        self.assertEqual(len(summary.controls()), 7)
        self.assertTrue(all(spec.method != "Base" for *_, spec in summary.controls()))
        self.assertEqual(summary.final_step("2B", "4B fixed"), 156)
        self.assertEqual(summary.final_step("4B", "8B fixed"), 312)
        self.assertEqual(summary.final_step("4B", "4B frozen"), 156)

    def test_missing_final_checkpoint_distinct_from_available_earlier_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oss = root / "oss"
            oss.mkdir()
            eval_root = root / "eval"
            spec = summary.teacher_scale.experiment_spec("2B", "2B", "0.5")
            checkpoint(oss, spec, 50)
            final_rows, step_rows, issues = summary.collect(oss, eval_root)
            self.assertEqual(len(final_rows), 7)
            self.assertFalse(issues)
            selected = next(row for row in final_rows
                            if row["student"] == "2B" and row["teacher"] == "2B frozen"
                            and row["view"] == "tr=0.5")
            self.assertEqual(selected["step"], 156)
            self.assertEqual(selected["ckpt_state"], "缺少 checkpoint")
            self.assertEqual(selected["observed_steps"], [50])
            self.assertEqual(next(row for row in step_rows
                                  if row["student"] == "2B" and row["teacher"] == "2B frozen"
                                  and row["view"] == "tr=0.5")["ckpt_state"], "完整")
            report = summary.render(final_rows, step_rows, issues, oss)
            self.assertIn("发现 steps=[50]", report)
            self.assertNotIn("| Base |", report)

    def test_complete_checkpoint_without_results_is_not_called_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oss = root / "oss"
            oss.mkdir()
            spec = summary.teacher_scale.experiment_spec("4B", "8B", "0.5")
            checkpoint(oss, spec, 312)
            row = summary.inspect_one("4B", "8B fixed", "tr=0.5", spec, 312,
                                      oss, root / "eval", [])
            self.assertEqual(row["ckpt_state"], "完整")
            self.assertEqual(row["states"]["CHAIR"], "未推理")
            self.assertEqual(row["states"]["POPE"], "未推理")
            self.assertEqual(row["states"]["AMBER"], "未推理")
            self.assertIn("CHAIR: 未推理", summary.gaps(row))
            self.assertIn("POPE: 未推理", summary.gaps(row))

    def test_partial_checkpoint_and_error_amber_are_flagged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oss = root / "oss"
            oss.mkdir()
            spec = summary.teacher_scale.experiment_spec("2B", "4B", "0.5")
            checkpoint(oss, spec, 156, complete=False)
            self.assertEqual(summary.checkpoint_state(oss, spec, 156, [])[0],
                             "checkpoint 不完整")
            output = (root / "eval/instruct/full" / f"{spec.experiment}_global_step_156"
                      / summary.inventory.DATASET / "official_v2" / "amber")
            output.mkdir(parents=True)
            rows = [{"id": 1, "model_answer": "A dog."},
                    {"id": 2, "model_answer": "[ERROR] HTTP 500"}]
            (output / "raw_results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (output / "amber_a_responses.json").write_text(json.dumps([
                {"id": 1, "response": "A dog."},
                {"id": 2, "response": "[ERROR] HTTP 500"},
            ]), encoding="utf-8")
            (output / "amber_metrics.json").write_text(json.dumps({
                "official_returncode": 0, "evaluation_type": "a",
                "metrics": {"generative_task_chair": 5.5, "generative_task_hal": 37.5},
            }), encoding="utf-8")
            old_expected = summary.aux_audit.EXPECTED_AMBER
            try:
                summary.aux_audit.EXPECTED_AMBER = 2
                row = summary.inspect_one("2B", "4B fixed", "tr=0.5", spec, 156,
                                          oss, root / "eval", [])
            finally:
                summary.aux_audit.EXPECTED_AMBER = old_expected
            self.assertEqual(row["states"]["AMBER"], "推理未完成")
            self.assertEqual(row["results"]["AMBER"].values, ("--", "--"))
            self.assertIn("raw_bad", row["results"]["AMBER"].detail)


if __name__ == "__main__":
    unittest.main()
