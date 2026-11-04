import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class VllmProcessGroupTest(unittest.TestCase):
    def test_cleanup_does_not_touch_another_session(self):
        target = subprocess.Popen(
            [sys.executable, str(SCRIPTS / "launch_vllm_session.py"), "sleep", "60"]
        )
        other = subprocess.Popen(
            [sys.executable, str(SCRIPTS / "launch_vllm_session.py"), "sleep", "60"]
        )
        try:
            time.sleep(0.2)
            self.assertEqual(os.getpgid(target.pid), target.pid)
            self.assertEqual(os.getpgid(other.pid), other.pid)
            subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; res_opd_stop_vllm_group "$2" 2',
                    "bash",
                    str(SCRIPTS / "vllm_process_group.sh"),
                    str(target.pid),
                ],
                check=True,
                timeout=10,
            )
            target.wait(timeout=5)
            self.assertIsNone(other.poll())
        finally:
            for process in (target, other):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)

    def test_cleanup_reaches_child_after_server_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "child.pid"
            child_code = (
                "import signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "time.sleep(60)"
            )
            parent_code = (
                "import subprocess,sys,pathlib,time; "
                "child=subprocess.Popen([sys.executable,'-c',sys.argv[2]]); "
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
                "time.sleep(1)"
            )
            parent = subprocess.Popen(
                [
                    sys.executable,
                    str(SCRIPTS / "launch_vllm_session.py"),
                    sys.executable,
                    "-c",
                    parent_code,
                    str(marker),
                    child_code,
                ]
            )
            try:
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(marker.exists())
                child_pid = int(marker.read_text())
                self.assertEqual(os.getpgid(child_pid), parent.pid)
                parent.wait(timeout=5)

                subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$1"; res_opd_stop_vllm_group "$2" 1',
                        "bash",
                        str(SCRIPTS / "vllm_process_group.sh"),
                        str(parent.pid),
                    ],
                    check=True,
                    timeout=10,
                )
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                try:
                    os.killpg(parent.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                parent.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
