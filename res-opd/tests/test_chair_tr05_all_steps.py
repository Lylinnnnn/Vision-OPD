import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "tmp"
    / "summarize_chair_tr05_all_steps.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("chair_tr05_all_steps", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_result(path, prompt="Describe the image."):
    path.mkdir(parents=True)
    metrics = {
        "metric_schema": "official_chair_caption_and_mention_v1",
        "eval_prompt": prompt,
        "total_captions": 2,
        "invalid_final_answer": 0,
        "total_object_mentions": 2,
        "total_correct_mentions": 1,
        "total_hallucinated_mentions": 1,
        "hallucinated_captions": 1,
        "CHAIRi": 0.5,
        "CHAIRs": 0.5,
        "total_mentioned": 2,
        "total_correct": 1,
        "total_hallucinated": 1,
        "ObjPrec": 0.5,
        "ObjRecall": 0.5,
        "ObjF1": 0.5,
    }
    (path / "chair_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (path / "eval_results.jsonl").write_text(
        '{"generated_caption": "A cat."}\n'
        '{"generated_caption": "A dog."}\n',
        encoding="utf-8",
    )


class Tr05AllStepsTest(unittest.TestCase):
    def test_scan_exact_setting_and_validate_every_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eval_root = root / "eval"
            eval_root.mkdir()
            manifest = root / "manifest.txt"
            primary = (
                "Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.5-a1.0-"
                "frozen-rkl-b32-rn4-full5k-e1"
            )
            extra = (
                "Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.5-a1.0-"
                "frozen-rkl-randommask-p30-b32-rn4-full5k-e1"
            )
            wrong_ratio = primary.replace("tr0.5", "tr0.75")
            manifest.write_text(
                "1|2b/base/original|local_base|/models/Qwen3VL-2B-Instruct|||1.0\n"
                f"2|2b/frozen_rkl/tr05/s50|oss_checkpoint|{primary}|oss-rkl|global_step_50|1.0\n",
                encoding="utf-8",
            )
            dataset = "train5000_test1000_original_sr1p0"
            suffix = "chair_official_v2"
            write_result(eval_root / "Qwen3VL-2B-Instruct" / dataset / suffix)
            for step in (50, 100):
                write_result(eval_root / f"{primary}_global_step_{step}" / dataset / suffix)
            write_result(eval_root / f"{extra}_global_step_50" / dataset / suffix)
            write_result(eval_root / f"{wrong_ratio}_global_step_50" / dataset / suffix)
            write_result(
                eval_root / f"{extra}_global_step_100" / dataset / suffix,
                prompt="Please describe this image in detail.",
            )

            entries, discovery_issues = MODULE.discover_entries(manifest, eval_root)
            self.assertEqual(discovery_issues, [])
            args = SimpleNamespace(
                output_suffix=suffix,
                fallback_output_suffix=[],
                expected_prompt="Describe the image.",
                expected_samples=2,
                length_mode="words",
                model_root=str(root / "models"),
            )
            rows, issues = MODULE.summarize_entries(entries, eval_root, args)
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(issues), 1)
            self.assertEqual([r["checkpoint_step"] for r in rows if r["method"] == "Frozen RKL"], ["50", "100"])
            self.assertEqual([r["checkpoint_step"] for r in rows if r["method"] == "RandomMask"], ["50"])
            report = MODULE.markdown_report(rows, issues, args, manifest)
            self.assertIn("Correct mentions", report)
            self.assertIn("no checkpoint is selected by score".capitalize(), report)
            self.assertIn("Please describe this image in detail.", report)
            self.assertNotIn(wrong_ratio, report)


if __name__ == "__main__":
    unittest.main()
