import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/tmp"
sys.path.insert(0, str(SCRIPT_DIR))
import run_main_tr05_missing_aux_4machines as batch  # noqa: E402


class FourMachineAuxTest(unittest.TestCase):
    def test_legacy_amber_requires_scored_complete_rows_with_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            amber = root / "amber"
            amber.mkdir()
            (amber / "raw_results.jsonl").write_text(
                json.dumps({"id": 1, "model_answer": "Yes"}) + "\n" +
                json.dumps({"id": 2, "model_answer": "[ERROR] context too long"}) + "\n")
            (amber / "amber_a_responses.json").write_text(json.dumps([
                {"id": 1, "response": "Yes"}, {"id": 2, "response": "No"}]))
            with mock.patch.object(batch, "amber_metrics", return_value=((1.0, 2.0), amber)):
                self.assertEqual(batch.legacy_amber_errors([root], expected=2), (1, root))
                with mock.patch.object(batch, "legacy_amber_errors", return_value=(1, root)), \
                        mock.patch.object(batch, "target_dirs", return_value=[root]), \
                        mock.patch.object(batch, "inspect_all", side_effect=[
                            mock.Mock(state="已推理有结果"), mock.Mock(state="推理未完成")]):
                    self.assertEqual(batch.pending("M005", root, accept_amber_error_rows=True), ())
                with mock.patch.object(batch, "target_dirs", return_value=[root]), \
                        mock.patch.object(batch, "inspect_all", side_effect=[
                            mock.Mock(state="已推理有结果"), mock.Mock(state="推理未完成")]):
                    self.assertEqual(batch.pending("M005", root), ("amber",))
                (amber / "raw_results.jsonl").write_text(
                    json.dumps({"id": 1, "model_answer": "Yes"}) + "\n" +
                    json.dumps({"id": 1, "model_answer": "[ERROR] context too long"}) + "\n")
                self.assertIsNone(batch.legacy_amber_errors([root], expected=2))

    def test_startup_memory_retry_marker_must_be_new_and_specific(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / ".shards" / "run1" / "failure_summary.txt"
            summary.parent.mkdir(parents=True)
            summary.write_text("Free memory on device cuda:0 is less than desired GPU memory utilization")
            self.assertTrue(batch.startup_memory_failure(root, summary.stat().st_mtime - 1))
            self.assertFalse(batch.startup_memory_failure(root, summary.stat().st_mtime + 1))
            summary.write_text("AMBER evaluator failed")
            self.assertFalse(batch.startup_memory_failure(root, 0))

    def test_startup_timeout_retry_marker_must_be_new_and_specific(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / ".shards" / "run1" / "failure_summary.txt"
            summary.parent.mkdir(parents=True)
            summary.write_text("vLLM server failed to start within 5400s")
            self.assertTrue(batch.startup_timeout_failure(root, summary.stat().st_mtime - 1))
            self.assertFalse(batch.startup_timeout_failure(root, summary.stat().st_mtime + 1))
            summary.write_text("AMBER evaluator failed")
            self.assertFalse(batch.startup_timeout_failure(root, 0))

    def test_queue_covers_all_requested_once_excluding_opera(self):
        assigned = [job for queue in batch.QUEUES.values() for job in queue]
        self.assertEqual([len(queue) for queue in batch.QUEUES.values()], [4, 4, 5, 5])
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), set(batch.TARGET_BY_ID) | {batch.INTERNVL_VISION})
        self.assertNotIn("OPERA-2B", assigned)
        self.assertNotIn("M088", assigned)
        self.assertEqual(len(assigned), 18)
        self.assertEqual(batch.SERIAL_JOBS, tuple(assigned))
        self.assertEqual(set(batch.JOB_MACHINE), set(assigned))
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

    def test_internvl_base_vision_uses_base_model_without_checkpoint_lookup(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(model_root=Path(temporary))
            with mock.patch.object(batch, "checkpoint_path") as checkpoint:
                with batch.staged_checkpoint(batch.INTERNVL_VISION, args) as model:
                    self.assertEqual(model, Path(temporary) / "internvl/InternVL3-2B-hf")
            checkpoint.assert_not_called()

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
        args = SimpleNamespace(gpu_memory_utilization=0.65, gpu_headroom_gib=6.0, base_port=28100)
        candidates = [str(index) for index in range(8)]
        memory = {gpu: (80000, 97871) for gpu in candidates}
        memory["7"] = (6162, 97871)
        with mock.patch.object(batch, "gpu_memory_mib", return_value=memory), \
                mock.patch.object(batch, "port_is_free", return_value=True) as ports:
            selected, utilization = batch.select_gpus(args, "Base-2B", candidates)
        self.assertEqual(selected, candidates[:7])
        self.assertEqual(utilization, 0.65)
        self.assertEqual(ports.call_count, 7)

    def test_rescans_before_each_experiment(self):
        args = SimpleNamespace(gpu_memory_utilization=0.65, gpu_headroom_gib=6.0, base_port=28100)
        snapshots = [{"0": (80000, 97871), "7": (6162, 97871)},
                     {"0": (80000, 97871), "7": (80000, 97871)}]
        with mock.patch.object(batch, "gpu_memory_mib", side_effect=snapshots) as scan, \
                mock.patch.object(batch, "port_is_free", return_value=True):
            self.assertEqual(batch.select_gpus(args, "Base-2B", ["0", "7"]), (["0"], 0.65))
            self.assertEqual(batch.select_gpus(args, "Base-8B", ["0", "7"]), (["0", "7"], 0.65))
        self.assertEqual(scan.call_count, 2)

    def test_no_eligible_gpu_fails_immediately(self):
        args = SimpleNamespace(gpu_memory_utilization=0.65, gpu_headroom_gib=6.0, base_port=28100)
        with mock.patch.object(batch, "gpu_memory_mib", return_value={"7": (6162, 97871)}), \
                mock.patch.object(batch, "port_is_free") as ports:
            with self.assertRaisesRegex(RuntimeError, "no waiting"):
                batch.select_gpus(args, "Base-2B", ["7"])
        ports.assert_not_called()

    def test_busy_port_fails_without_cleanup(self):
        args = SimpleNamespace(gpu_memory_utilization=0.65, gpu_headroom_gib=6.0, base_port=28100)
        with mock.patch.object(batch, "gpu_memory_mib", return_value={"0": (80000, 97871)}), \
                mock.patch.object(batch, "port_is_free", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "ports busy"):
                batch.select_gpus(args, "Base-2B", ["0"])

    def test_serial_eight_gpus_do_not_scan_memory(self):
        args = SimpleNamespace(base_port=32100, gpu_memory_utilization=0.65)
        gpus = [str(index) for index in range(8)]
        with mock.patch.object(batch, "gpu_memory_mib") as memory, \
                mock.patch.object(batch, "port_is_free", return_value=True) as ports:
            self.assertEqual(batch.serial_gpus(args, gpus), (gpus, 0.65))
        memory.assert_not_called()
        self.assertEqual(ports.call_count, 8)

    def test_serial_continues_after_failed_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(eval_root=Path(temporary), gpu_list="0,1,2,3,4,5,6,7",
                                   preflight_only=False)
            seen = []

            def run(_args, target_id, gpus):
                seen.append((target_id, gpus))
                if target_id == batch.SERIAL_JOBS[0]:
                    raise RuntimeError("synthetic failure")

            with mock.patch.object(batch, "preflight"), mock.patch.object(batch, "run_one", side_effect=run):
                with self.assertRaisesRegex(RuntimeError, "finished with failed jobs"):
                    batch.run_serial(args)
            self.assertEqual([target_id for target_id, _ in seen], list(batch.SERIAL_JOBS))
            self.assertTrue(all(len(gpus) == 8 for _, gpus in seen))
            args.gpu_list = "0,1,2,3"
            with self.assertRaisesRegex(ValueError, "exactly eight"):
                batch.run_serial(args)

    def test_staging_download_retries_without_using_oss_mount_for_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "model.safetensors").write_bytes(b"weight")
            (source / "config.json").write_text("{}")
            args = SimpleNamespace(oss_root=root, stage_dir=root, oss_uri="oss://bucket/ckpt",
                                   stage_retries=1, stage_timeout=100)
            calls = []

            def download(command, **kwargs):
                calls.append(command)
                if len(calls) == 1:
                    raise subprocess.TimeoutExpired(command, 100)
                Path(command[3]).write_bytes(b"weight")

            with mock.patch.object(batch, "checkpoint_path", return_value=source), \
                    mock.patch.object(batch.subprocess, "run", side_effect=download), \
                    mock.patch.object(batch.time, "sleep"):
                with batch.staged_checkpoint("M005", args) as staged:
                    self.assertEqual((staged / "model.safetensors").read_bytes(), b"weight")
            self.assertEqual(len(calls), 2)

    def test_staging_retries_same_size_bad_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "model.safetensors").write_bytes(b"weight")
            (source / "model.safetensors.sha256").write_text(
                hashlib.sha256(b"weight").hexdigest() + "  model.safetensors\n")
            args = SimpleNamespace(oss_root=root, stage_dir=root, oss_uri="oss://bucket/ckpt",
                                   stage_retries=1, stage_timeout=100)
            calls = []

            def download(command, **kwargs):
                calls.append(command)
                Path(command[3]).write_bytes(b"badbad" if len(calls) == 1 else b"weight")

            with mock.patch.object(batch, "checkpoint_path", return_value=source), \
                    mock.patch.object(batch.subprocess, "run", side_effect=download), \
                    mock.patch.object(batch.time, "sleep"):
                with batch.staged_checkpoint("M005", args) as staged:
                    self.assertEqual((staged / "model.safetensors").read_bytes(), b"weight")
            self.assertEqual(len(calls), 2)

    def test_serial_retries_only_startup_timeout(self):
        @contextmanager
        def staged(*_):
            yield Path("/tmp/staged-model")

        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(eval_root=Path(temporary), model_root=Path(temporary),
                                   data_root=Path(temporary), python=Path("/tmp/python3"),
                                   base_port=32100, startup_timeout=5400, kv_cache_gib=8,
                                   startup_retries=1, all_serial=True,
                                   accept_amber_error_rows=False)
            failed = subprocess.CalledProcessError(1, ["eval"])
            with mock.patch.object(batch, "pending", side_effect=[("pope",), ()]), \
                    mock.patch.object(batch, "staged_checkpoint", side_effect=staged), \
                    mock.patch.object(batch, "serial_gpus", return_value=([str(i) for i in range(8)], 0.65)), \
                    mock.patch.object(batch, "startup_timeout_failure", return_value=True), \
                    mock.patch.object(batch.time, "sleep"), \
                    mock.patch.object(batch.subprocess, "run", side_effect=[failed, None]) as run:
                batch.run_one(args, "Base-2B", [str(i) for i in range(8)])
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args.kwargs["env"]["EVAL_SHARD_COUNT"], "8")
            self.assertEqual(run.call_args.kwargs["env"]["VLLM_STARTUP_TIMEOUT"], "5400")
            self.assertEqual(run.call_args.kwargs["env"]["VLLM_PORT_CLEANUP"], "True")

    def test_serial_does_not_retry_non_startup_failure(self):
        @contextmanager
        def staged(*_):
            yield Path("/tmp/staged-model")

        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(eval_root=Path(temporary), model_root=Path(temporary),
                                   data_root=Path(temporary), python=Path("/tmp/python3"),
                                   base_port=32100, startup_timeout=5400, kv_cache_gib=8,
                                   startup_retries=2, all_serial=True,
                                   accept_amber_error_rows=False)
            with mock.patch.object(batch, "pending", return_value=("amber",)), \
                    mock.patch.object(batch, "staged_checkpoint", side_effect=staged), \
                    mock.patch.object(batch, "serial_gpus", return_value=([str(i) for i in range(8)], 0.65)), \
                    mock.patch.object(batch, "startup_timeout_failure", return_value=False), \
                    mock.patch.object(batch.subprocess, "run",
                                      side_effect=subprocess.CalledProcessError(1, ["eval"])) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    batch.run_one(args, "Base-2B", [str(i) for i in range(8)])
            self.assertEqual(run.call_count, 1)

    def test_scan_after_staging_and_pass_selected_shards(self):
        events = []

        @contextmanager
        def staged(*_):
            events.append("staged")
            yield Path("/tmp/staged-model")

        def select(*_):
            events.append("scanned")
            return ["0", "2"], 0.65

        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(eval_root=Path(temporary), model_root=Path(temporary),
                                   data_root=Path(temporary), python=Path("/tmp/python3"),
                                   base_port=28100, startup_timeout=2400,
                                   kv_cache_gib=6,
                                   accept_amber_error_rows=False)
            with mock.patch.object(batch, "pending", side_effect=[("pope",), ()]), \
                    mock.patch.object(batch, "staged_checkpoint", side_effect=staged), \
                    mock.patch.object(batch, "select_gpus", side_effect=select), \
                    mock.patch.object(batch.subprocess, "run") as run:
                batch.run_one(args, "Base-2B", ["0", "1", "2"])
        self.assertEqual(events, ["staged", "scanned"])
        self.assertEqual(run.call_args.kwargs["env"]["GPU_LIST"], "0,2")
        self.assertEqual(run.call_args.kwargs["env"]["EVAL_SHARD_COUNT"], "2")
        self.assertEqual(run.call_args.kwargs["env"]["VLLM_KV_CACHE_MEMORY_BYTES"],
                         str(6 * 1024**3))


if __name__ == "__main__":
    unittest.main()
