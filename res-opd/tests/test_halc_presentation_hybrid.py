import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT = Path(__file__).resolve().parents[1] / "baselines/run_halc_presentation_hybrid.py"
spec = importlib.util.spec_from_file_location("run_halc_presentation_hybrid", SCRIPT)
hybrid = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hybrid
spec.loader.exec_module(hybrid)


class HALCPresentationHybridTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "Qwen3VL-8B-Instruct"
        self.model.mkdir()
        self.samples = [{"image_id": image_id} for image_id in range(16)]
        self.old_root = self.root / "old"
        for index in range(8):
            rows = []
            for sample in self.samples:
                image_id = sample["image_id"]
                if image_id >= 10 or image_id % 8 != index:
                    continue
                rows.append({
                    "image_id": image_id,
                    "generated_caption": f"old caption {image_id}",
                    "halc_config": {
                        "model_path": str(self.model), "prompt": "Describe the image.",
                        "parameters": {"max_new_tokens": 8192},
                        "shard_count": 8, "shard_index": index,
                    },
                })
            hybrid.write_jsonl(
                self.old_root / f"shard_{index:02d}/out/eval_results.jsonl", rows
            )

    def test_old_rows_are_validated_and_missing_images_partitioned(self):
        old, source_files = hybrid.inspect_original(
            self.old_root, self.samples, self.model, "Describe the image."
        )
        self.assertEqual(len(old), 10)
        self.assertEqual(len(source_files), 8)
        missing, groups = hybrid.partition_missing(self.samples, old, 4)
        self.assertEqual([row["image_id"] for row in missing], list(range(10, 16)))
        self.assertEqual([len(group) for group in groups], [2, 2, 1, 1])

    def test_mixed_rows_keep_original_config_and_mark_provenance(self):
        old, _ = hybrid.inspect_original(
            self.old_root, self.samples, self.model, "Describe the image."
        )
        new = {image_id: {"image_id": image_id, "generated_caption": "new caption",
                          "halc_config": {"parameters": {"max_new_tokens": 2048}}}
               for image_id in range(10, 16)}
        rows = hybrid.build_mixed_rows(self.samples, old, new, 2048)
        self.assertEqual(len(rows), 16)
        self.assertEqual(rows[0]["presentation_origin"], "original_8192")
        self.assertEqual(rows[0]["halc_config"]["parameters"]["max_new_tokens"], 8192)
        self.assertEqual(rows[10]["presentation_origin"], "continuation_2048")
        self.assertEqual(rows[10]["halc_config"]["parameters"]["max_new_tokens"], 2048)
        self.assertTrue(all(row["presentation_only"] for row in rows))

    def test_rejects_duplicate_original_ids(self):
        path = self.old_root / "shard_00/out/eval_results.jsonl"
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(hybrid.read_jsonl(path)[0]) + "\n")
        with self.assertRaisesRegex(RuntimeError, "Duplicate original"):
            hybrid.inspect_original(
                self.old_root, self.samples, self.model, "Describe the image."
            )

    def test_rejects_incomplete_hybrid(self):
        old, _ = hybrid.inspect_original(
            self.old_root, self.samples, self.model, "Describe the image."
        )
        with self.assertRaisesRegex(RuntimeError, "overlapping or incomplete"):
            hybrid.build_mixed_rows(self.samples, old, {}, 2048)

    def test_new_results_must_keep_old_decoder_settings(self):
        output_dir = self.root / "presentation_only" / "test"
        path = output_dir / ".new_shards/worker_00/out/eval_results.jsonl"
        old_config = {
            "model_path": str(self.model), "prompt": "Describe the image.",
            "seed": 42, "source_commit": "pinned", "adapter_revision": "v2",
            "parameters": {"max_new_tokens": 8192, "beam_size": 3},
        }
        row = {"image_id": 10, "generated_caption": "new caption",
               "halc_config": {**old_config, "parameters": {"max_new_tokens": 2048,
                                                            "beam_size": 4}}}
        hybrid.write_jsonl(path, [row])
        with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
            hybrid.collect_new_rows([[{"image_id": 10}]], output_dir, self.model,
                                    "Describe the image.", 2048, old_config)


if __name__ == "__main__":
    unittest.main()
