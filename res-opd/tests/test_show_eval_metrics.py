"""Tests for the general read-only evaluation metric viewer."""

import importlib.util
import json
from pathlib import Path
import sys


EVAL = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL))
SPEC = importlib.util.spec_from_file_location("show_eval_metrics", EVAL / "show_eval_metrics.py")
VIEWER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VIEWER)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_mixed_results_and_file_inputs(tmp_path):
    full = tmp_path / "Qwen3VL-2B-Instruct" / "train5000_test1000_original_sr1p0" / "run1"
    write_json(full / "chair_metrics.json", {
        "num_samples": 2, "metric_schema": "official_chair_caption_and_mention_v1",
        "CHAIRi": 0.2, "CHAIRs": 0.3, "ObjF1": 0.7,
        "total_correct_mentions": 11, "total_hallucinated_mentions": 2,
    })
    write_jsonl(full / "eval_results.jsonl", [
        {"image_id": 1, "generated_caption": "red car"},
        {"image_id": 2, "generated_caption": "blue bus"},
    ])
    write_json(full / "pope/pope_summary.json", {"macro_f1": 0.9})
    write_json(full / "amber/amber_metrics.json", {
        "official_returncode": 0,
        "metrics": {"generative_task_chair": 5.4, "generative_task_hal": 35.2},
    })
    write_jsonl(full / "amber/raw_results.jsonl", [
        {"id": 1, "model_answer": "[ERROR] timeout"},
    ])
    label, directory = VIEWER.parse_result(f"test={full / 'eval_results.jsonl'}")
    assert (label, directory) == ("test", full)
    assert VIEWER.parse_result(str(full))[0] == "Qwen3VL-2B-Instruct/run1"
    row, audit = VIEWER.summarize(label, directory)
    assert row == ["test", "20.00", "30.00", "70.00", "2.00", "7.50",
                   "11", "2", "90.00", "5.40", "35.20"]
    assert "AMBER rows=1 rc=0 errors=1" in audit

    partial = tmp_path / "other"
    partial.mkdir()
    row, _ = VIEWER.summarize("other", partial)
    assert row[1:] == ["--"] * 10


def test_failed_amber_is_not_shown(tmp_path):
    write_json(tmp_path / "amber/amber_metrics.json", {
        "official_returncode": 1, "metrics": {"generative_task_hal": 99.0},
    })
    row, audit = VIEWER.summarize("bad", tmp_path)
    assert row[-2:] == ["--", "--"]
    assert "rc=1" in audit
