import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tmp/eval_native_base_compressed_12.py"
SPEC = importlib.util.spec_from_file_location("eval_native_base_compressed_12", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class NativeBaseCompressed12Test(unittest.TestCase):
    def test_twelve_distinct_model_ratio_jobs(self):
        selected = MODULE.jobs("all", MODULE.RATIOS)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len(set(selected)), 12)
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            outputs = [MODULE.result_dir(repo, key, ratio, "native_base_official_v2") for key, ratio in selected]
            self.assertEqual(len(set(outputs)), 12)
            self.assertIn("internvl3/full/InternVL3-2B-hf/", str(outputs[-1]))
            self.assertIn("original_sr0p75", str(outputs[-1]))

    def test_complete_requires_matching_ratio_and_all_three_benchmarks(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_json(output / "chair_metrics.json", {
                "metric_schema": MODULE.CHAIR_SCHEMA, "eval_prompt": MODULE.CHAIR_PROMPT,
                "num_samples": 1000, "total_results": 1000,
            })
            (output / "eval_results.jsonl").write_text(json.dumps({
                "eval_student_ratio": 0.5, "eval_degradation_mode": "original",
                "eval_prompt": MODULE.CHAIR_PROMPT,
            }) + "\n", encoding="utf-8")
            write_json(output / "pope/pope_summary.json", {
                "macro_f1": 0.9,
                "benchmarks": {key: {"f1": 0.9} for key in MODULE.POPE_SPLITS},
            })
            write_json(output / "amber/amber_metrics.json", {
                "official_returncode": 0, "evaluation_type": "a",
                "metrics": {"generative_task_chair": 5.0},
            })
            self.assertTrue(MODULE.complete(output, "0.5"))
            self.assertFalse(MODULE.complete(output, "0.25"))
            (output / "amber/amber_metrics.json").unlink()
            self.assertFalse(MODULE.complete(output, "0.5"))

    def test_preflight_blocks_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            model_root = repo / "models"
            data_root = repo / "data"
            for path in (
                repo / "res-opd/data/train_5k.parquet", repo / "res-opd/data/test_1000.json",
                repo / "res-opd/scripts/eval_after_merge_sharded_8gpu.sh",
                model_root / "qwen/Qwen3VL-2B-Instruct/config.json",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")
            (data_root / "AMBER").mkdir(parents=True)
            args = SimpleNamespace(model_root=model_root, data_root=data_root,
                                   output_suffix="native_base_official_v2",
                                   gpu_list="0,1,2,3,4,5,6,7", shard_count=8)
            selected = [("qwen2b", "0.25")]
            MODULE.preflight(repo, args, selected)
            output = MODULE.result_dir(repo, "qwen2b", "0.25", args.output_suffix)
            output.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "Existing incomplete output"):
                MODULE.preflight(repo, args, selected)


if __name__ == "__main__":
    unittest.main()
