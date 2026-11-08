import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tmp/run_three_tables_fill.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("run_three_tables_fill", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ThreeTablesFillResourcesTest(unittest.TestCase):
    def test_wait_retries_transient_gpu_process(self):
        busy = {"910455": {"gpu": "2", "name": "[Not Found]"}}
        with patch.object(MODULE, "resource_state", side_effect=[({}, busy), ({}, {})]) as state:
            with patch.object(MODULE.time, "sleep") as sleep:
                MODULE.wait_for_release(["2"], [8600], 30)
        self.assertEqual(state.call_count, 2)
        sleep.assert_called_once_with(5)

    def test_wait_still_fails_when_gpu_remains_busy(self):
        busy = {"910455": {"gpu": "2", "name": "[Not Found]"}}
        with patch.object(MODULE, "resource_state", return_value=({}, busy)):
            with patch.object(MODULE.time, "monotonic", side_effect=[0, 10]):
                with self.assertRaisesRegex(RuntimeError, "remained busy for 10s"):
                    MODULE.wait_for_release(["2"], [8600], 10)

    def test_shared_mode_accepts_busy_gpu_with_enough_free_memory(self):
        with patch.object(MODULE, "listening_pids", return_value=set()), \
             patch.object(MODULE, "gpu_memory_mib", return_value={"0": (50000, 97871)}), \
             patch.object(MODULE, "gpu_processes", side_effect=AssertionError("must not check PIDs")):
            MODULE.wait_for_capacity(["0"], [25100], 10, 0.35, 6)

    def test_shared_mode_waits_for_memory_and_uses_only_selected_port(self):
        readings = [{"0": (20000, 97871)}, {"0": (50000, 97871)}]
        with patch.object(MODULE, "listening_pids", return_value=set()) as ports, \
             patch.object(MODULE, "gpu_memory_mib", side_effect=readings), \
             patch.object(MODULE.time, "sleep") as sleep:
            MODULE.wait_for_capacity(["0"], [25100], 10, 0.35, 6)
        self.assertEqual(ports.call_count, 2)
        ports.assert_any_call(25100)
        sleep.assert_called_once_with(5)

    def test_preflight_uses_transition_wait(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            for path in (root / "res-opd/data/test_1000.json",
                         root / "res-opd/data/train_5k.parquet"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            for path in (data / "AMBER", data / "COCO/coco2017val"):
                path.mkdir(parents=True)
            for path in (data / "mmstar.json", data / "cv_bench.json"):
                path.touch()
            args = SimpleNamespace(output_suffix="three_tables_fill_official_v2",
                                   gpu_list="1,3,5,7", eval_shard_count=4, data_root=data,
                                   transition_wait_seconds=300, startup_timeout=3600, base_port=8600,
                                   share_gpus=False)
            with patch.object(MODULE, "ROOT", root), patch.object(MODULE.shutil, "which", return_value="/bin/tmux"):
                with patch.object(MODULE.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
                    with patch.object(MODULE, "wait_for_release") as wait:
                        MODULE.preflight(args, [], Path(sys.executable))
            wait.assert_called_once_with(
                ["1", "3", "5", "7"],
                [8600 + index * 10 for index in range(4)], 300)

    def test_startup_only_output_is_safe_to_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "eval"
            output.mkdir()
            self.assertTrue(MODULE.retryable_startup_output(output))
            (output / ".shards").mkdir()
            self.assertTrue(MODULE.retryable_startup_output(output))
            (output / "chair_metrics.json").touch()
            self.assertFalse(MODULE.retryable_startup_output(output))

    def test_vision_completion_reads_judge_json_array(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            judge = output / "mmstar/judge/example_answer.jsonl"
            judge.parent.mkdir(parents=True)
            judge.write_text(json.dumps([{"judge": "Yes"}] * 1500))
            self.assertTrue(MODULE.complete(output, ("mmstar",)))

    def test_8b_retry_uses_eager_without_touching_other_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = MODULE.Task("T014", "8B", "0.25", 30, 50, ("chair",))
            args = SimpleNamespace(output_suffix="three_tables_fill_official_v2",
                                   oss_mount=root, data_root=root,
                                   gpu_list="1,3,5,7", eval_shard_count=4, base_port=8600,
                                   startup_timeout=3600, share_gpus=False)
            output = root / "res-opd/eval_results/instruct/full" / \
                f"{task.experiment}_global_step_50" / MODULE.DATASET / args.output_suffix
            (output / ".shards").mkdir(parents=True)
            with patch.object(MODULE, "ROOT", root), \
                 patch.object(MODULE, "complete", side_effect=[False, True]), \
                 patch.object(MODULE, "stage_checkpoint", return_value=nullcontext(root / "staged")), \
                 patch.object(MODULE.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)), \
                 patch.object(MODULE.subprocess, "run") as run:
                MODULE.run_one(args, task, Path(sys.executable))
            env = run.call_args.kwargs["env"]
            self.assertEqual(env["VLLM_ENFORCE_EAGER"], "True")
            self.assertEqual(env["VLLM_MAX_NUM_SEQS"], "8")
            self.assertEqual(env["VLLM_PORT_CLEANUP"], "False")
            self.assertEqual(env["EVAL_SHARD_COUNT"], "4")
            self.assertEqual(env["GPU_LIST"], "1,3,5,7")

    def test_shared_mode_sets_vllm_memory_budget_for_all_scales(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(output_suffix="three_tables_fill_official_v2",
                                   oss_mount=root, data_root=root, gpu_list="0,1,2,3,4,5,6,7",
                                   eval_shard_count=8, base_port=25100, startup_timeout=3600,
                                   share_gpus=True, shared_gpu_memory_utilization=0.35)
            task = MODULE.Task("T046", "2B", "0.5", 10, 50, ("chair",))
            with patch.object(MODULE, "ROOT", root), \
                 patch.object(MODULE, "complete", side_effect=[False, True]), \
                 patch.object(MODULE, "stage_checkpoint", return_value=nullcontext(root / "staged")), \
                 patch.object(MODULE.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)), \
                 patch.object(MODULE.subprocess, "run") as run:
                MODULE.run_one(args, task, Path(sys.executable))
            env = run.call_args.kwargs["env"]
            self.assertEqual(env["VLLM_GPU_MEMORY_UTILIZATION"], "0.35")
            self.assertEqual(env["VLLM_ENFORCE_EAGER"], "True")
            self.assertEqual(env["VLLM_MAX_NUM_SEQS"], "8")

    def test_8b_checkpoint_staged_once_and_cleaned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = MODULE.Task("T014", "8B", "0.25", 30, 50, ("chair",))
            source = task.checkpoint(root)
            source.mkdir(parents=True)
            payload = b"checkpoint-contents"
            (source / "model.safetensors").write_bytes(payload)
            (source / "model.safetensors.sha256").write_text(hashlib.sha256(payload).hexdigest())
            (source / "config.json").write_text("{}")
            args = SimpleNamespace(oss_mount=root, stage_dir=root)

            def download(command, **_kwargs):
                self.assertEqual(command[:2], ["ossutil", "cp"])
                Path(command[3]).write_bytes(payload)

            with patch.object(MODULE.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)), \
                 patch.object(MODULE.shutil, "which", return_value="/usr/bin/ossutil"), \
                 patch.object(MODULE.subprocess, "run", side_effect=download) as run:
                with MODULE.stage_checkpoint(task, args) as staged:
                    self.assertEqual((staged / "model.safetensors").read_bytes(), payload)
                    self.assertEqual((staged / "config.json").read_text(), "{}")
                self.assertFalse(staged.exists())
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
