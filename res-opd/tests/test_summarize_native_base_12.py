import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/tmp"
sys.path.insert(0, str(SCRIPT_DIR))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class NativeBaseSummaryTest(unittest.TestCase):
    def test_collect_and_render_all_rows(self):
        native = load_script("eval_native_base_compressed_12")
        summary = load_script("summarize_native_base_12")
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            for model_key, ratio in native.jobs("all", native.RATIOS):
                output = native.result_dir(repo, model_key, ratio, "native_base_official_v2")
                write_json(output / "chair_metrics.json", {
                    "metric_schema": native.CHAIR_SCHEMA,
                    "eval_prompt": native.CHAIR_PROMPT,
                    "num_samples": 1000,
                    "total_results": 1000,
                    "CHAIRi": 0.1,
                    "CHAIRs": 0.2,
                    "ObjPrec": 0.8,
                    "ObjRecall": 0.7,
                    "ObjF1": 0.7467,
                    "total_correct_mentions": 900,
                    "total_hallucinated_mentions": 100,
                })
                write_json(output / "pope/pope_summary.json", {
                    "macro_f1": 0.9,
                    "benchmarks": {split: {"f1": 0.9} for split in native.POPE_SPLITS},
                })
                write_json(output / "amber/amber_metrics.json", {
                    "official_returncode": 0,
                    "evaluation_type": "a",
                    "metrics": {"generative_task_chair": 5.0, "generative_task_hal": 30.0},
                })
                write_json(output / "eval_results.jsonl", {
                    "eval_degradation_mode": "original",
                    "eval_student_ratio": float(ratio),
                    "eval_prompt": native.CHAIR_PROMPT,
                })

            rows = summary.collect(repo, "native_base_official_v2")
            self.assertEqual(len(rows), 12)
            self.assertEqual(rows[0]["model"], "Qwen3VL-2B-Instruct")
            self.assertEqual(rows[-1]["model"], "InternVL3-2B-hf")
            rendered = summary.markdown(rows, "native_base_official_v2")
            self.assertIn("| N01 | Qwen3VL-2B-Instruct | 0.25 | 10.00 | 20.00 |", rendered)
            self.assertIn("| N12 | InternVL3-2B-hf | 0.75 |", rendered)

            output = native.result_dir(repo, "internvl2b", "0.75", "native_base_official_v2")
            (output / "amber/amber_metrics.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "internvl2b sr=0.75"):
                summary.collect(repo, "native_base_official_v2")
