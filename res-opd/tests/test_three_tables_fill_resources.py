import importlib.util
import sys
import tempfile
import unittest
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
                                   gpu_list="0,1,2,3,4,5,6,7", data_root=data,
                                   transition_wait_seconds=300, base_port=8600)
            with patch.object(MODULE, "ROOT", root), patch.object(MODULE.shutil, "which", return_value="/bin/tmux"):
                with patch.object(MODULE.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
                    with patch.object(MODULE, "wait_for_release") as wait:
                        MODULE.preflight(args, [], Path(sys.executable))
            wait.assert_called_once_with(
                [str(index) for index in range(8)],
                [8600 + index * 10 for index in range(8)], 300)


if __name__ == "__main__":
    unittest.main()
