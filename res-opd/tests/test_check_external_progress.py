import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "baselines/check_external_progress.py"
SPEC = importlib.util.spec_from_file_location("check_external_progress", SCRIPT)
progress = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(progress)


class ExternalProgressTest(unittest.TestCase):
    def test_counts_only_latest_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / ".shards/first/shard_00/out"
            second = root / ".shards/second/shard_00/out"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "eval_results.jsonl").write_text("a\nb\n")
            (second / "eval_results.jsonl").write_text("c\n")
            os.utime(root / ".shards/first", (1, 1))
            os.utime(root / ".shards/second", (2, 2))
            run, shards = progress.latest_shards(root, ".shards")
            self.assertEqual(run.name, "second")
            self.assertEqual(progress.counts(shards, Path("eval_results.jsonl")), [1])
            self.assertEqual(progress.count_lines(root / "missing"), 0)


if __name__ == "__main__":
    unittest.main()
