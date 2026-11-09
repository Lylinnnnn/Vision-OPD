import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/tmp"
sys.path.insert(0, str(SCRIPT_DIR))
import run_main_tr05_missing_aux_4machines as batch  # noqa: E402


class FourMachineAuxTest(unittest.TestCase):
    def test_queue_covers_all_requested_once_excluding_opera(self):
        assigned = [job for queue in batch.QUEUES.values() for job in queue]
        self.assertEqual([len(queue) for queue in batch.QUEUES.values()], [4, 5, 5, 5])
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), set(batch.TARGET_BY_ID) | {batch.INTERNVL_VISION})
        self.assertNotIn("OPERA-2B", assigned)

    def test_checkpoint_and_output_are_exact_for_8b_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = batch.checkpoint_path("M064", root / "oss")
            self.assertEqual(checkpoint.name, "global_step_250")
            self.assertIn("tr0.5_a1.0_frozen_rkl_entropymask_p30", str(checkpoint))
            output = batch.output_dir("M064", root / "eval")
            self.assertIn("EntropyMask", batch.TARGET_BY_ID["M064"].method)
            self.assertEqual(output.name, "main_tr05_aux_official_v2")
            self.assertIn("global_step_250", str(output))
            self.assertEqual(batch.output_dir(batch.INTERNVL_VISION, root / "eval").name,
                             "main_tr05_base_vision_official_v2")
            self.assertEqual(batch.output_dir("M088", root / "eval").name,
                             "chair_external_halc_qwen3vl_adapted_official")

    def test_pending_checks_each_benchmark_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(batch, "inspect_all") as inspect:
                inspect.side_effect = [mock.Mock(state="已推理有结果"), mock.Mock(state="未推理")]
                self.assertEqual(batch.pending("M005", root), ("amber",))
            with mock.patch.object(batch, "inspect_vision_all") as inspect:
                inspect.side_effect = [mock.Mock(state="已推理有结果"), mock.Mock(state="未推理")]
                self.assertEqual(batch.pending(batch.INTERNVL_VISION, root), ("cv-bench",))


if __name__ == "__main__":
    unittest.main()
