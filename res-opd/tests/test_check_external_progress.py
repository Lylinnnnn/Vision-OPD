import importlib.util
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
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

    def test_presentation_and_error_rows_are_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            presentation = root / "presentation_mixed_8192_2048"
            presentation.mkdir()
            (presentation / "presentation_manifest.json").write_text('{"old_count": 998}')
            (presentation / "eval_results.jsonl").write_text("row\n")
            worker = presentation / ".new_shards/worker_00/out"
            worker.mkdir(parents=True)
            (worker / "eval_results.jsonl").write_text("row\n")
            aux = presentation / ".aux_shards/run/shard_00/out/pope/pope_adv"
            aux.mkdir(parents=True)
            (aux / "eval_results.jsonl").write_text(
                '{"model_answer":"yes"}\n'
                '{"model_answer":"[ERROR] server failed"}\n'
            )
            with redirect_stdout(io.StringIO()) as output:
                progress.report(root, "halc", "4B")
            report = output.getvalue()
            self.assertIn("continuation=1/2 original=998/1000", report)
            self.assertIn("pope_adv merged=0/5223 merged_errors=0 latest=2/5223 latest_errors=1", report)


if __name__ == "__main__":
    unittest.main()
