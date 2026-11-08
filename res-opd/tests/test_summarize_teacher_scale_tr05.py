import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tmp/summarize_teacher_scale_tr05.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("summarize_teacher_scale_tr05", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TeacherScaleTr05Test(unittest.TestCase):
    def test_exact_teacher_scale_names_and_table_rows(self):
        rows = MODULE.table_specs()
        self.assertEqual(len(rows), 9)
        self.assertEqual([row[:3] for row in rows if row[2] != "Base"], [
            ("2B", "2B frozen", "tr=0.5"), ("2B", "4B fixed", "tr=0.5"),
            ("2B", "2B frozen", "tr=1.0"), ("2B", "4B fixed", "tr=1.0"),
            ("4B", "4B frozen", "tr=0.5"), ("4B", "8B fixed", "tr=0.5"),
            ("4B", "8B fixed", "tr=1.0"),
        ])
        fixed = MODULE.experiment_spec("4B", "8B", "0.5")
        self.assertEqual(fixed.experiment,
                         "Res-OPD-Qwen3VL-4B-Instruct-orig-sr1.0-tr0.5-a1.0-fixed-rkl-"
                         "teacherQwen3VL-8B-Instruct-b16-rn4-full5k-e1")
        self.assertEqual(fixed.oss_name,
                         "ResOPD_Qwen3VL_4B_Instruct_orig_sr1.0_tr0.5_a1.0_fixed_rkl_"
                         "teacherQwen3VL_8B_Instruct_b16_rn4_full5k-e1")
        report = MODULE.render([], [], False, Path("/missing"))
        self.assertEqual(report.count("| 2B |"), 5)
        self.assertEqual(report.count("| 4B |"), 4)
        self.assertIn("tr=0.75 四组改为 tr=0.50", report)
        self.assertIn("TEACHER_MODE", report)
        self.assertIn("fixed-rkl-teacherQwen3VL", fixed.experiment)

    def test_untrained_and_checkpoint_not_reinferred_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oss = root / "oss"
            oss.mkdir()
            frozen = MODULE.experiment_spec("2B", "2B", "0.5")
            fixed = MODULE.experiment_spec("2B", "4B", "0.5")
            checkpoint = oss / frozen.oss_name / "global_step_50"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"weight")
            (checkpoint / "config.json").write_text("{}")
            with patch.object(MODULE.inventory, "ROOT", root), \
                 patch.object(MODULE, "table_specs", return_value=[
                     ("2B", "2B frozen", "tr=0.5", frozen),
                     ("2B", "4B fixed", "tr=0.5", fixed),
                 ]):
                rows, issues, mounted = MODULE.collect(oss)
            self.assertTrue(mounted)
            self.assertEqual([row["status"] for row in rows], [
                "有 checkpoint，待新版 CHAIR 推理", "待训练（OSS 无匹配 checkpoint）",
            ])
            self.assertIn(fixed.oss_name, rows[1]["checkpoint"])
            report = MODULE.render(rows, issues, mounted, oss)
            self.assertIn("global_step_50", report)
            self.assertIn("## 待训练", report)
            self.assertIn("## 待推理或补评测", report)

    def test_incomplete_upload_is_not_called_untrained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oss = root / "oss"
            fixed = MODULE.experiment_spec("2B", "4B", "0.5")
            checkpoint = oss / fixed.oss_name / "global_step_50"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}")
            with patch.object(MODULE.inventory, "ROOT", root), \
                 patch.object(MODULE, "table_specs", return_value=[
                     ("2B", "4B fixed", "tr=0.5", fixed),
                 ]):
                rows, issues, mounted = MODULE.collect(oss)
            self.assertEqual(rows[0]["status"], "checkpoint 上传不完整，待核实")
            self.assertIn(checkpoint, rows[0]["incomplete"])

    def test_possible_alternate_name_is_not_called_untrained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oss = root / "oss"
            fixed = MODULE.experiment_spec("4B", "8B", "0.5")
            alias = oss / (fixed.oss_name + "_run2")
            alias.mkdir(parents=True)
            with patch.object(MODULE.inventory, "ROOT", root), \
                 patch.object(MODULE, "table_specs", return_value=[
                     ("4B", "8B fixed", "tr=0.5", fixed),
                 ]):
                rows, issues, mounted = MODULE.collect(oss)
            self.assertEqual(rows[0]["status"], "疑似其他命名，待人工核对")
            self.assertEqual(rows[0]["aliases"], [alias])

    def test_existing_empty_training_directory_is_not_called_untrained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oss = root / "oss"
            fixed = MODULE.experiment_spec("2B", "4B", "0.5")
            (oss / fixed.oss_name).mkdir(parents=True)
            with patch.object(MODULE.inventory, "ROOT", root), \
                 patch.object(MODULE, "table_specs", return_value=[
                     ("2B", "4B fixed", "tr=0.5", fixed),
                 ]):
                rows, issues, mounted = MODULE.collect(oss)
            self.assertEqual(rows[0]["status"], "训练目录存在，尚无完整 checkpoint，待核实")

    def test_missing_checkpoint_is_not_marked_complete(self):
        spec = MODULE.experiment_spec("2B", "4B", "0.5")
        row = {"spec": spec, "step": 50, "status": "仅发现旧推理目录，checkpoint待核实",
               "values": {metric: 0.5 for metric in MODULE.TABLE_METRICS}}
        self.assertEqual(MODULE.status_for(row, True), "checkpoint 待核实")


if __name__ == "__main__":
    unittest.main()
