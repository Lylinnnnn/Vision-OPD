"""CPU-only tests for HALVA shard recovery."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "baselines/halva/resume_eval.py"
spec = importlib.util.spec_from_file_location("halva_resume_eval", SCRIPT)
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


class HalvaResumeEvalTest(unittest.TestCase):
    def make_shards(self, root, count=2):
        for index in range(count):
            (root / f"shard_{index:02d}" / "out").mkdir(parents=True)

    def test_discovers_old_tmp_root_after_shared_shards_were_moved(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory) / "tmp"
            tmp.mkdir()
            source = tmp / "halva_qwen2b_resume_20261120_114707_568886"
            self.make_shards(source)
            self.assertEqual(
                resume.select_root(Path(directory) / "missing", tmp, "halva_coco5k_e1",
                                   "qwen2b", None), source.resolve()
            )

    def test_multiple_roots_require_explicit_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base / "out"
            tmp = base / "tmp"
            tmp.mkdir()
            self.make_shards(output / ".shards" / "20261120_000000_1")
            self.make_shards(tmp / "halva_qwen2b_resume_20261120_000000_1")
            with self.assertRaisesRegex(RuntimeError, "Pass --shard-root"):
                resume.select_root(output, tmp, "halva_coco5k_e1", "qwen2b", None)

    def test_recover_partial_amber_and_archive_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_shards(root)
            amber = root / "shard_00/out/amber"
            amber.mkdir()
            raw = amber / "raw_results.jsonl"
            good = {"id": 1, "model_answer": "A dog.", "official_response": "A dog."}
            failed = {"id": 2, "model_answer": "[ERROR] timeout", "official_response": "No"}
            raw.write_text(json.dumps(good) + "\n" + json.dumps(failed) + "\n{" , encoding="utf-8")
            (root / "shard_00/exit_code").write_text("1")
            (root / "shard_00/eval.log").write_text("previous log")

            resume.recover_amber(root, 2, "audit")
            resume.archive_status(root, 2, "audit")

            recovered = json.loads((amber / "amber_a_responses.json").read_text())
            self.assertEqual(recovered, [{"id": 1, "response": "A dog."}])
            self.assertEqual(len(raw.read_text().splitlines()), 2)
            self.assertTrue((amber / "raw_results.jsonl.before_resume_audit").exists())
            self.assertEqual((root / "shard_00/exit_code.before_resume_audit").read_text(), "1")
            self.assertEqual((root / "shard_00/eval.log.before_resume_audit").read_text(), "previous log")

    def test_stage_shared_root_without_touching_other_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            tmp = base / "tmp"
            tmp.mkdir()
            source = base / "shared" / "20261120_010203_4"
            self.make_shards(source)
            other = base / "shared" / "other_run"
            self.make_shards(other)
            staged = resume.stage_root(source, tmp, "halva_coco5k_e1", "qwen4b")
            self.assertTrue((staged / "shard_00/out").is_dir())
            self.assertFalse(source.exists())
            self.assertTrue(other.exists())


if __name__ == "__main__":
    unittest.main()
