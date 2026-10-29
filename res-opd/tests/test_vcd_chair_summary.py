import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "res-opd" / "baselines" / "append_vcd_chair_summary.py"
REGISTRY = REPO_ROOT / "res-opd" / "baselines" / "registry.json"


class VCDChairSummaryTest(unittest.TestCase):
    def test_validates_and_upserts_markdown(self):
        source_commit = json.loads(REGISTRY.read_text(encoding="utf-8"))["baselines"]["vcd"]["commit"]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_dir = root / "result"
            result_dir.mkdir()
            model_path = root / "Qwen3VL-2B-Instruct"
            model_path.mkdir()
            target = root / "sensitivity.md"
            target.write_text("# Sensitivity\n", encoding="utf-8")
            summary = root / "vcd.json"

            config = {
                "method": "vcd",
                "source_commit": source_commit,
                "model_name": "Qwen3VL-2B-Instruct",
                "prompt": "Describe the image.",
                "alpha": 1.0,
                "beta": 0.1,
                "noise_step": 500,
                "decode_mode": "sample",
                "degradation_mode": "original",
                "student_ratio": 1.0,
            }
            metrics = {
                "metric_schema": "official_chair_caption_and_mention_v1",
                "total_results": 2,
                "num_samples": 2,
                "total_captions": 2,
                "hallucinated_captions": 1,
                "total_object_mentions": 4,
                "total_correct_mentions": 3,
                "total_hallucinated_mentions": 1,
                "CHAIRi": 0.25,
                "CHAIRs": 0.5,
                "ObjPrec": 0.75,
                "ObjRecall": 0.6,
                "ObjF1": 2 * 0.75 * 0.6 / (0.75 + 0.6),
                "invalid_final_answer": 0,
                "unclosed_thinking": 0,
                "final_answer_valid_rate": 1.0,
                "merged_from_shards": True,
                "shard_count": 2,
                "eval_prompt": "Describe the image.",
            }
            rows = []
            for image_id, token_ids in ((1, [1, 2]), (2, [3, 4, 5])):
                rows.append(
                    {
                        "image_id": image_id,
                        "generated_caption": "A short caption.",
                        "generated_token_ids": token_ids,
                        "generated_token_count": len(token_ids),
                        "eval_prompt": "Describe the image.",
                        "decoding_method": "vcd",
                        "eval_degradation_mode": "original",
                        "eval_student_ratio": 1.0,
                        "vcd_source_commit": source_commit,
                        "vcd_alpha": 1.0,
                        "vcd_beta": 0.1,
                        "vcd_noise_step": 500,
                        "vcd_decode_mode": "sample",
                    }
                )
            (result_dir / "vcd_config.json").write_text(json.dumps(config), encoding="utf-8")
            (result_dir / "chair_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (result_dir / "eval_results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            command = [
                sys.executable,
                str(SCRIPT),
                "--result-dir",
                str(result_dir),
                "--append-md",
                str(target),
                "--summary-json",
                str(summary),
                "--model-path",
                str(model_path),
                "--expected-samples",
                "2",
                "--expected-shards",
                "2",
                "--length-mode",
                "record",
            ]
            first = subprocess.run(command, capture_output=True, text=True, check=True)
            second = subprocess.run(command, capture_output=True, text=True, check=True)
            document = target.read_text(encoding="utf-8")
            payload = json.loads(summary.read_text(encoding="utf-8"))

            self.assertIn("VCD_CHAIR_VALIDATION=PASS", first.stdout)
            self.assertIn("VCD_CHAIR_VALIDATION=PASS", second.stdout)
            self.assertEqual(document.count("BEGIN external-vcd-chair-official"), 1)
            self.assertIn("| VCD |", document)
            self.assertEqual(payload["validation_status"], "PASS")
            self.assertEqual(payload["response_tokens_total"], 5)


if __name__ == "__main__":
    unittest.main()
