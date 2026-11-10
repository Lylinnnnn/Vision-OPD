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
    def test_exact_eight_non_base_controls(self):
        controls = summary.controls()
        self.assertEqual(len(controls), 8)
        self.assertTrue(all(spec.method != "Base" for *_, spec in controls))
        self.assertEqual(sum(view == "sr=1.0,tr=0.5" for _, _, view, _ in controls), 4)
        self.assertEqual(sum(view == "sr=0.5,tr=1.0" for _, _, view, _ in controls), 4)
        self.assertIn(("4B", "4B frozen", "sr=0.5,tr=1.0"),
                      [(student, teacher, view) for student, teacher, view, _ in controls])
        self.assertTrue(all("-sr0.5-tr1.0-" in spec.experiment
                            for _, _, view, spec in controls if view == "sr=0.5,tr=1.0"))
        self.assertTrue(all("_sr0.5_tr1.0_" in spec.oss_name
                            for _, _, view, spec in controls if view == "sr=0.5,tr=1.0"))

    def test_unmounted_oss_is_not_reported_as_missing_training(self):
        group = {"complete": [], "aliases": [], "steps": []}
        self.assertEqual(summary.group_status(group, Path("/path/that/is/not/mounted")),
                         "OSS 未挂载，无法核验")

    def test_report_does_not_invent_a_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oss = root / "oss"
            oss.mkdir()
            eval_root = root / "eval"
            spec = summary.teacher_scale.experiment_spec("2B", "2B", "0.5")
            checkpoint(oss, spec, 50)
            groups, step_rows, issues = summary.collect(oss, eval_root)
            self.assertEqual(len(groups), 8)
            self.assertFalse(issues)
            self.assertEqual(len(step_rows), 1)
            self.assertEqual(step_rows[0]["step"], 50)
            self.assertEqual(step_rows[0]["ckpt_state"], "完整")
            report = summary.render(groups, step_rows, issues, oss,
                                    "teacher_scale_matched_official_v2")
            self.assertIn("| 2B | 2B frozen | sr=1.0,tr=0.5 | 50 |", report)
            self.assertNotIn("| 2B | 2B frozen | sr=1.0,tr=0.5 | 156 |", report)
            self.assertNotIn("| Base |", report)

    def test_complete_checkpoint_without_results_is_not_called_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oss = root / "oss"
            oss.mkdir()
            spec = summary.teacher_scale.experiment_spec("4B", "8B", "0.5")
            checkpoint(oss, spec, 312)
            row = summary.inspect_one("4B", "8B fixed", "sr=1.0,tr=0.5", spec, 312,
                                      oss, root / "eval", [], "teacher_scale_matched_official_v2")
            self.assertEqual(row["ckpt_state"], "完整")
            self.assertEqual(row["states"]["CHAIR"], "未推理")
            self.assertEqual(row["states"]["POPE"], "未匹配推理")
            self.assertEqual(row["states"]["AMBER"], "未匹配推理")
            self.assertIn("CHAIR: 未推理", summary.gaps(row))
            self.assertIn("POPE: 未匹配推理", summary.gaps(row))

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
                      / summary.inventory.DATASET / "teacher_scale_matched_official_v2" / "amber")
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
                row = summary.inspect_one("2B", "4B fixed", "sr=1.0,tr=0.5", spec, 156,
                                          oss, root / "eval", [], "teacher_scale_matched_official_v2")
            finally:
                summary.aux_audit.EXPECTED_AMBER = old_expected
            self.assertEqual(row["states"]["AMBER"], "推理未完成")
            self.assertEqual(row["results"]["AMBER"].values, ("--", "--"))
            self.assertIn("raw_bad", row["results"]["AMBER"].detail)

    def test_matched_aux_requires_source_and_original_student_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pope = directory / "pope"
            for split in ("pope_adv", "pope_pop", "pope_random"):
                path = pope / split / "eval_results.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"student_ratio": 1.0,
                                            "degradation_mode": "original"}) + "\n")
            summary_path = pope / "pope_summary.json"
            summary_path.write_text(json.dumps({"benchmarks": {
                split: {"pope_source": "other"}
                for split in ("pope_adv", "pope_pop", "pope_random")
            }}), encoding="utf-8")
            self.assertIn("source", summary.matched_protocol_error(directory, "pope"))
            summary_path.write_text(json.dumps({"benchmarks": {
                split: {"pope_source": "res-opd-test"}
                for split in ("pope_adv", "pope_pop", "pope_random")
            }}), encoding="utf-8")
            self.assertIsNone(summary.matched_protocol_error(directory, "pope"))
            amber = directory / "amber"
            amber.mkdir()
            raw = amber / "raw_results.jsonl"
            raw.write_text(json.dumps({"student_ratio": 0.5,
                                       "degradation_mode": "original"}) + "\n")
            self.assertIn("student view", summary.matched_protocol_error(directory, "amber"))
            self.assertIsNone(summary.matched_protocol_error(directory, "amber", 0.5))

    def test_reverse_view_does_not_reuse_original_student_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oss = root / "oss"
            oss.mkdir()
            eval_root = root / "eval"
            reverse = summary.teacher_scale.experiment_spec("2B", "4B", "1.0", "0.5")
            original = summary.teacher_scale.experiment_spec("2B", "4B", "1.0")
            checkpoint(oss, reverse, 50)
            old = eval_root / "instruct/full" / f"{original.experiment}_global_step_50"
            (old / summary.inventory.DATASET).mkdir(parents=True)
            groups, rows, issues = summary.collect(oss, eval_root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["view"], "sr=0.5,tr=1.0")
            self.assertEqual(rows[0]["states"]["CHAIR"], "未推理")
            self.assertFalse(issues)

    def test_reverse_chair_raw_requires_compressed_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = directory / "eval_results.jsonl"
            metrics = directory / "chair_metrics.json"
            rows = [json.dumps({"image_id": index, "generated_caption": "A dog.",
                                "gt_captions": ["A dog."],
                                "eval_prompt": "Describe the image.",
                                "eval_student_ratio": 1.0})
                    for index in range(1000)]
            raw.write_text("\n".join(rows) + "\n", encoding="utf-8")
            state, _ = summary.chair_audit.inspect_raw(raw, metrics, 0.5)
            self.assertEqual(state, "已推理，协议待核实")
            rows = [json.dumps({"image_id": index, "generated_caption": "A dog.",
                                "gt_captions": ["A dog."],
                                "eval_prompt": "Describe the image.",
                                "eval_student_ratio": 0.5})
                    for index in range(1000)]
            raw.write_text("\n".join(rows) + "\n", encoding="utf-8")
            state, _ = summary.chair_audit.inspect_raw(raw, metrics, 0.5)
            self.assertEqual(state, "已推理，待新版计分")


if __name__ == "__main__":
    unittest.main()
