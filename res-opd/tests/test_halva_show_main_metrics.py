"""Focused tests for the read-only HALVA main-table viewer."""

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "baselines/halva/show_main_metrics.py"
SPEC = importlib.util.spec_from_file_location("show_main_metrics", SCRIPT)
VIEWER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VIEWER)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_complete_metrics(tmp_path):
    write_json(tmp_path / "chair_metrics.json", {
        "metric_schema": "official_chair_caption_and_mention_v1", "num_samples": 2,
        "CHAIRi": 0.2, "CHAIRs": 0.3, "ObjF1": 0.7,
        "total_correct_mentions": 11, "total_hallucinated_mentions": 2,
    })
    write_jsonl(tmp_path / "eval_results.jsonl", [
        {"image_id": 1, "generated_caption": "red car"},
        {"image_id": 2, "generated_caption": "blue bus"},
    ])
    write_json(tmp_path / "pope/pope_summary.json", {
        "macro_f1": 0.9, "benchmarks": {split: {} for split in VIEWER.POPE_SPLITS},
    })
    for split in VIEWER.POPE_SPLITS:
        write_jsonl(tmp_path / "pope" / split / "eval_results.jsonl", [
            {"sample_uid": i, "model_answer": "yes"} for i in range(3)
        ])
    write_json(tmp_path / "amber/amber_metrics.json", {
        "official_returncode": 0,
        "metrics": {"generative_task_chair": 5.4, "generative_task_hal": 35.2},
    })
    write_jsonl(tmp_path / "amber/raw_results.jsonl", [
        {"id": i, "model_answer": "answer"} for i in range(4)
    ])
    write_json(tmp_path / "amber/amber_a_responses.json", [{"id": i} for i in range(4)])

    row, audit = VIEWER.summarize(tmp_path, "Qwen3VL", "2B", "halva_coco5k_e1", 2, 3, 4)
    assert row == ["Qwen3VL", "2B", "HALVA", "e1", "20.00", "30.00", "70.00",
                   "2.00", "7.50", "11", "2", "90.00", "5.40", "35.20"]
    assert "POPE_errors=0 AMBER_errors=0" in audit


def test_incomplete_metrics_are_not_reported(tmp_path):
    write_json(tmp_path / "amber/amber_metrics.json", {
        "official_returncode": 1,
        "metrics": {"generative_task_hal": 99.0},
    })
    write_jsonl(tmp_path / "amber/raw_results.jsonl", [
        {"id": 1, "model_answer": "[ERROR] unavailable"},
    ])
    row, audit = VIEWER.summarize(tmp_path, "InternVL3", "2B", "halva_coco5k_e1", 2, 3, 4)
    assert row[4:] == ["--"] * 10
    assert "AMBER_errors=1" in audit
