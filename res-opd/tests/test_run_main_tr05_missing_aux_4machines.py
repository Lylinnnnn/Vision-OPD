import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/tmp"
sys.path.insert(0, str(SCRIPT_DIR))
import run_main_tr05_missing_aux_4machines as batch  # noqa: E402


class FourMachineAuxTest(unittest.TestCase):
    def test_queue_covers_all_requested_once_excluding_opera(self):
        assigned = [job for queue in batch.QUEUES.values() for job in queue]
        self.assertEqual([len(queue) for queue in batch.QUEUES.values()], [4, 4, 5, 5])
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), set(batch.TARGET_BY_ID) | {batch.INTERNVL_VISION})
        self.assertNotIn("OPERA-2B", assigned)
        self.assertNotIn("M088", assigned)
        self.assertEqual(len(assigned), 18)
        workload = lambda job: 2 if job == batch.INTERNVL_VISION else int(batch.TARGET_BY_ID[job].size[0])
        self.assertEqual([sum(workload(job) for job in queue) for queue in batch.QUEUES.values()],
                         [22, 22, 20, 20])

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

    def test_pending_checks_each_benchmark_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(batch, "inspect_all") as inspect:
                inspect.side_effect = [mock.Mock(state="已推理有结果"), mock.Mock(state="未推理")]
                self.assertEqual(batch.pending("M005", root), ("amber",))
            with mock.patch.object(batch, "inspect_vision_all") as inspect:
                inspect.side_effect = [mock.Mock(state="已推理有结果"), mock.Mock(state="未推理")]
                self.assertEqual(batch.pending(batch.INTERNVL_VISION, root), ("cv-bench",))

    def test_selects_busy_machine_subset_without_waiting(self):
        args = SimpleNamespace(gpu_memory_utilization=None, gpu_headroom_gib=4.0, base_port=28100)
        candidates = [str(index) for index in range(8)]
        memory = {gpu: (80000, 97871) for gpu in candidates}
        memory["7"] = (6162, 97871)
        with mock.patch.object(batch, "gpu_memory_mib", return_value=memory), \
                mock.patch.object(batch, "port_is_free", return_value=True) as ports:
            selected, utilization = batch.select_gpus(args, "Base-2B", candidates)
        self.assertEqual(selected, candidates[:7])
        self.assertEqual(utilization, 0.35)
        self.assertEqual(ports.call_count, 7)

    def test_scale_specific_memory_fraction(self):
        args = SimpleNamespace(gpu_memory_utilization=None, gpu_headroom_gib=4.0, base_port=28100)
        with mock.patch.object(batch, "gpu_memory_mib", return_value={"0": (50000, 97871)}), \
                mock.patch.object(batch, "port_is_free", return_value=True):
            self.assertEqual(batch.select_gpus(args, "Base-4B", ["0"]), (["0"], 0.45))
            with self.assertRaisesRegex(RuntimeError, "No GPU has enough free memory"):
                batch.select_gpus(args, "Base-8B", ["0"])

    def test_no_eligible_gpu_fails_immediately(self):
        args = SimpleNamespace(gpu_memory_utilization=None, gpu_headroom_gib=4.0, base_port=28100)
        with mock.patch.object(batch, "gpu_memory_mib", return_value={"7": (6162, 97871)}), \
                mock.patch.object(batch, "port_is_free") as ports:
            with self.assertRaisesRegex(RuntimeError, "no waiting"):
                batch.select_gpus(args, "Base-2B", ["7"])
        ports.assert_not_called()

    def test_busy_port_fails_without_cleanup(self):
        args = SimpleNamespace(gpu_memory_utilization=None, gpu_headroom_gib=4.0, base_port=28100)
        with mock.patch.object(batch, "gpu_memory_mib", return_value={"0": (80000, 97871)}), \
                mock.patch.object(batch, "port_is_free", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "ports busy"):
                batch.select_gpus(args, "Base-2B", ["0"])


if __name__ == "__main__":
    unittest.main()
