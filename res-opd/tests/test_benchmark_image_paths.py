import importlib.util
from pathlib import Path

import pytest


PATHS_PATH = Path(__file__).resolve().parents[2] / "eval" / "benchmark_paths.py"
SPEC = importlib.util.spec_from_file_location("res_opd_benchmark_paths", PATHS_PATH)
PATHS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATHS)


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    return path


def test_mmstar_stale_absolute_path_is_remapped(tmp_path):
    expected = touch(tmp_path / "MMStar_images" / "00480.jpg")
    records = [{"images": ["/old/data/MMStar_images/00480.jpg"]}]

    remapped = PATHS.resolve_benchmark_images(records, "mmstar", tmp_path / "mmstar.json")

    assert remapped == 1
    assert records[0]["images"] == [str(expected)]


def test_cvbench_preserves_image_directory_marker(tmp_path):
    expected = touch(tmp_path / "CVBench_3d_images" / "00016.jpg")
    touch(tmp_path / "CVBench_2d_images" / "00016.jpg")
    records = [{"images": ["/old/data/CVBench_3d_images/00016.jpg"]}]

    remapped = PATHS.resolve_benchmark_images(records, "cv-bench", tmp_path / "cv_bench.json")

    assert remapped == 1
    assert records[0]["images"] == [str(expected)]


def test_cvbench_ambiguous_basename_fails(tmp_path):
    touch(tmp_path / "CVBench_2d_images" / "00016.jpg")
    touch(tmp_path / "CVBench_3d_images" / "00016.jpg")
    records = [{"images": ["/old/data/00016.jpg"]}]

    with pytest.raises(RuntimeError, match="Ambiguous image path"):
        PATHS.resolve_benchmark_images(records, "cv-bench", tmp_path / "cv_bench.json")
