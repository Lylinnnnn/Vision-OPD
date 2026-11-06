import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tmp/summarize_internvl3_full_results.py"
SPEC = importlib.util.spec_from_file_location("internvl3_full_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class InternVL3FullSummaryTest(unittest.TestCase):
    def test_main_checkpoint_name_is_recognized(self):
        self.assertEqual(
            MODULE.method_name("Res-OPD-InternVL3-2B-hf-tr05-fixedmb1-full5k-e1"),
            "RiskMask",
        )

    def test_all_steps_and_suffixes_remain_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_result = root / "baseline" / "base_chair_original_1000_8gpu"
            write_json(base_result / "chair_metrics.json", {
                "metric_schema": MODULE.OFFICIAL_CHAIR_SCHEMA,
                "num_samples": 1000, "CHAIRi": 0.2, "CHAIRs": 0.3, "ObjF1": 0.7,
            })
            base = root / "InternVL3-riskmask_global_step_50" / "original_sr1p0" / "official_v2"
            write_json(base / "chair_metrics.json", {
                "metric_schema": MODULE.OFFICIAL_CHAIR_SCHEMA,
                "num_samples": 1000, "CHAIRi": 0.1, "CHAIRs": 0.2, "ObjF1": 0.8,
            })
            write_json(base / "pope/pope_summary.json", {
                "macro_f1": 0.91,
                "benchmarks": {split: {"f1": 0.91} for split in MODULE.POPE_SPLITS},
            })
            write_json(base / "amber/amber_metrics.json", {
                "official_returncode": 0,
                "metrics": {"generative_task_chair": 4.2, "generative_task_hal": 31.0},
            })
            judge = base / "mmstar/judge/a_answer.jsonl"
            judge.parent.mkdir(parents=True)
            judge.write_text('{"judge": "yes"}\n{"judge": "no"}\n', encoding="utf-8")
            cv_judge = base / "cv-bench/judge/a_answer.jsonl"
            cv_judge.parent.mkdir(parents=True)
            cv_judge.write_text('{"judge": "yes"}\n', encoding="utf-8")
            write_json(base / "cv-bench/cvbench_breakdown.json", {
                "accuracy": 0.7, "accuracy_2d": 0.6, "accuracy_3d": 0.8,
            })
            second = root / "InternVL3-riskmask_global_step_100" / "original_sr1p0" / "official_v2"
            write_json(second / "chair_metrics.json", {"num_samples": 5, "CHAIRi": 0.3})
            other_suffix = root / "InternVL3-riskmask_global_step_100" / "original_sr1p0" / "smoke"
            write_json(other_suffix / "chair_metrics.json", {"num_samples": 1, "CHAIRi": 0.4})
            write_json(root / "InternVL3-base" / "original_sr1p0" / "official_v2" / "chair_metrics.json", {"CHAIRi": 0.5})

            rows, issues = MODULE.collect(root, base_result)
            report = MODULE.markdown(rows, issues, root)
            self.assertEqual(len(rows), 4)
            self.assertEqual(issues, [])
            self.assertEqual(rows[0]["method"], "Base")
            self.assertEqual(rows[0]["chair"]["CHAIRi"], 0.2)
            self.assertIn("4.2000", report)
            self.assertIn("0.7000", report)
            self.assertIn("0.5000", report)
            self.assertIn("official", report)
            self.assertIn("unverified/partial", report)
            self.assertIn("| R001 | Base | -- | 0.2000 |", report)
            chair_header = next(line for line in report.splitlines() if line.startswith("| ID | Method | Step | CHAIRi"))
            chair_row = next(line for line in report.splitlines() if line.startswith("| R001 | Base | -- | 0.2000"))
            self.assertEqual(chair_header.count("|"), chair_row.count("|"))
            expected_pipes = None
            for line in report.splitlines():
                if line.startswith("| ID |"):
                    expected_pipes = line.count("|")
                elif line.startswith("| "):
                    self.assertEqual(line.count("|"), expected_pipes, line)
            self.assertIn("| `smoke` |", report)
            self.assertNotIn("InternVL3-base", report)

    def test_corrupt_metric_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            write_json(baseline / "chair_metrics.json", {"CHAIRi": 0.1})
            output = root / "InternVL3-rkl_global_step_50" / "dataset" / "official_v2"
            output.mkdir(parents=True)
            (output / "chair_metrics.json").write_text("{broken", encoding="utf-8")
            rows, issues = MODULE.collect(root, baseline)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(issues), 1)
            self.assertIn("| missing |", MODULE.markdown(rows, issues, root))

    def test_numeric_step_order_and_missing_baseline_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                MODULE.collect(root, root / "missing_base")
            baseline = root / "baseline"
            write_json(baseline / "chair_metrics.json", {"CHAIRi": 0.1})
            for step in (100, 50, 156):
                output = root / f"InternVL3-frozen-rkl_global_step_{step}" / "dataset" / "official_v2"
                write_json(output / "chair_metrics.json", {"CHAIRi": 0.1})
            rows, _ = MODULE.collect(root, baseline)
            self.assertEqual([row["step"] for row in rows], [None, 50, 100, 156])

    def test_unjudged_vision_file_does_not_report_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "judge.jsonl"
            path.write_text('{"judge": "yes"}\n{"judge": ""}\n', encoding="utf-8")
            result = MODULE.load_judge(path, [])
            self.assertEqual(result["judged"], 1)
            self.assertIsNone(result["accuracy"])


if __name__ == "__main__":
    unittest.main()
