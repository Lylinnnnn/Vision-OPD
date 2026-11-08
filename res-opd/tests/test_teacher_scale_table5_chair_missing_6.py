import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/tmp"))
import run_paper_chair_official_batch as batch
import summarize_teacher_scale_tr05 as teacher_scale


class TeacherScaleMissingChairTest(unittest.TestCase):
    def test_manifest_is_exact_requested_six(self):
        manifest = ROOT / "eval/manifests/teacher_scale_table5_chair_missing_6.txt"
        entries = batch.load_manifest(manifest)
        expected = [
            ("2B", "2B", "100"), ("2B", "2B", "156"),
            ("2B", "4B", "50"), ("2B", "4B", "150"),
            ("4B", "8B", "100"), ("4B", "8B", "150"),
        ]
        self.assertEqual(len(entries), len(expected))
        for entry, (student, teacher, step) in zip(entries, expected):
            spec = teacher_scale.experiment_spec(student, teacher, "1.0")
            self.assertEqual(entry["source_type"], "oss_checkpoint")
            self.assertEqual(entry["primary"], spec.experiment)
            self.assertEqual(entry["oss_name"], spec.oss_name)
            self.assertEqual(entry["step"], f"global_step_{step}")
            self.assertEqual(entry["student_ratio"], "1.0")


if __name__ == "__main__":
    unittest.main()
