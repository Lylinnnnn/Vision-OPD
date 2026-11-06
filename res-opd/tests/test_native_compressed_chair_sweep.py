import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "tmp"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "native_compressed_chair_sweep", SCRIPT_DIR / "native_compressed_chair_sweep.py"
)
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)

BATCH_SPEC = importlib.util.spec_from_file_location(
    "run_paper_chair_official_batch", SCRIPT_DIR / "run_paper_chair_official_batch.py"
)
BATCH = importlib.util.module_from_spec(BATCH_SPEC)
BATCH_SPEC.loader.exec_module(BATCH)


class NativeCompressedChairSweepTest(unittest.TestCase):
    def test_paper_rows_resolve_to_expected_old_steps(self):
        source_manifest = (
            Path(__file__).resolve().parents[1]
            / "eval" / "manifests" / "paper_chair_describe45_sources.txt"
        )
        sources = SWEEP.lookup_sources(source_manifest)
        expected = {
            ("2b", "0.25"): "global_step_50",
            ("2b", "0.5"): "global_step_50",
            ("2b", "0.75"): "global_step_50",
            ("8b", "0.25"): "global_step_200",
            ("8b", "0.5"): "global_step_50",
            ("8b", "0.75"): "global_step_100",
        }
        self.assertEqual({key: source["step"] for key, source in sources.items()}, expected)

    def test_exact_checkpoint_discovery_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_root = root / "ckpt"
            model_root = root / "models"
            sources = {}
            for scale in ("2b", "8b"):
                model = model_root / f"Qwen3VL-{scale.upper()}-Instruct"
                model.mkdir(parents=True)
                (model / "config.json").write_text("{}", encoding="utf-8")
                for ratio in SWEEP.RATIOS:
                    name = f"Res-OPD-Qwen3VL-{scale.upper()}-Instruct-orig-sr1.0-tr{ratio}-a1.0-frozen-rkl-riskmask-nll-p30-b32-rn4-full5k-e1"
                    oss_name = f"oss_{scale}_{ratio}"
                    sources[(scale, ratio)] = {
                        "primary": name,
                        "oss_name": oss_name,
                        "step": "global_step_50",
                    }
                    for step in (50, 100):
                        path = checkpoint_root / oss_name / f"global_step_{step}"
                        path.mkdir(parents=True)
                        (path / "model.safetensors").write_bytes(b"model")
                        (path / "config.json").write_text("{}", encoding="utf-8")
                    if ratio == "0.25":
                        bad = checkpoint_root / oss_name / "global_step_150"
                        bad.mkdir()
                        (bad / "config.json").write_text("{}", encoding="utf-8")
            entries, inventory = SWEEP.make_entries(sources, checkpoint_root, model_root)
            self.assertEqual(len(entries), 20)
            self.assertEqual(len(inventory), 6)
            self.assertEqual(inventory[0][3], [50, 100])
            self.assertEqual(inventory[0][4], [(150, "model.safetensors")])
            self.assertEqual(entries[1][0], "2b/native_base/sr025")
            self.assertEqual(entries[2][0], "2b/riskmask/tr025/p30/s50")
            self.assertEqual(entries[2][5], "1.0")
            self.assertNotIn("global_step_150", [entry[4] for entry in entries])

    def test_skip_requires_prompt_and_official_gt_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary)
            metrics = {
                "total_results": 2,
                "metric_schema": "official_chair_caption_and_mention_v1",
                "eval_prompt": "Describe the image.",
                "ground_truth_protocol": "instance_categories_union_reference_caption_objects",
            }
            (result / "chair_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            record = {
                "eval_student_ratio": 0.25,
                "eval_degradation_mode": "original",
                "eval_prompt": "Describe the image.",
            }
            (result / "eval_results.jsonl").write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8"
            )
            self.assertTrue(BATCH.output_complete(result, 2, "Describe the image."))
            self.assertTrue(BATCH.output_complete(result, 2, "Describe the image.", "0.25"))
            self.assertFalse(BATCH.output_complete(result, 2, "Describe the image.", "0.5"))
            self.assertFalse(BATCH.output_complete(result, 2, "Other prompt"))
            metrics["ground_truth_protocol"] = "instances_only"
            (result / "chair_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            self.assertFalse(BATCH.output_complete(result, 2, "Describe the image."))


if __name__ == "__main__":
    unittest.main()
