"""CPU-only checks for the APASI-Base COCO 5k adaptation."""

import importlib.util
import json
from pathlib import Path
import sys


APASI = Path(__file__).resolve().parents[1] / "baselines/apasi"


def module(name):
    spec = importlib.util.spec_from_file_location(name, APASI / f"{name}.py")
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)
    return imported


prepare = module("prepare")
materialize = module("materialize")
storage = module("storage")
generate = module("generate")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_plan_uses_late_sentence_and_absent_cooccurring_object():
    row = {"image_id": "12", "image_path": "/image.jpg", "prompt": "Describe it.",
           "caption": "A cat sits. It looks left. The room is bright. The floor is clear. The scene is calm."}
    parsed = {"node_objects": ["cat"]}
    graph = {"cat": {"chair": 8}}
    synonyms = {"cat": ["cat"], "chair": ["chair"]}
    plan = prepare.make_plan(row, parsed, graph, synonyms, 0.2, 42)
    assert plan is not None
    assert len(plan["guides"]) == 1
    assert "0" not in plan["guides"]
    assert next(iter(plan["guides"].values()))["object"] == "chair"
    assert prepare.make_plan({**row, "caption": "Only one sentence."}, parsed,
                             graph, synonyms, 0.2, 42) is None


def test_materialized_pairs_are_matched(tmp_path):
    image = tmp_path / "000000000012.jpg"
    image.touch()
    question = {"image_id": "12", "image_path": str(image), "question": "Describe it."}
    plan = {"image_id": "12", "image_path": str(image), "prompt": "Describe it.",
            "chosen": "A cat sits."}
    injected = {**plan, "rejected": "A cat sits. A chair appears."}
    write_jsonl(tmp_path / "questions.jsonl", [question])
    write_jsonl(tmp_path / "plans.jsonl", [plan])
    write_jsonl(tmp_path / "injected/shard_00.jsonl", [injected])
    rows = materialize.collect_pairs(tmp_path / "questions.jsonl", tmp_path / "plans.jsonl",
                                     tmp_path / "injected")
    assert rows[0]["instruction"] == "<image>Describe it."
    assert rows[0]["images"] == [str(image)]
    write_jsonl(tmp_path / "injected/shard_01.jsonl", [injected])
    try:
        materialize.collect_pairs(tmp_path / "questions.jsonl", tmp_path / "plans.jsonl",
                                  tmp_path / "injected")
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("Duplicate injected IDs were accepted")


def test_storage_cleanup_scope(tmp_path):
    root = tmp_path / "apasi_coco5k"
    target = storage.owned_target(root, "experiment", "qwen2b", "adapter")
    assert target == root / "experiment/qwen2b/adapter"
    for bad_root in (tmp_path / "rlsf_v_coco5k", tmp_path / "other"):
        try:
            storage.owned_target(bad_root, "experiment", "qwen2b", "adapter")
        except ValueError:
            pass
        else:
            raise AssertionError("Unrelated work root was accepted")
    run = root / "experiment"
    run.parent.mkdir(parents=True, exist_ok=True)
    run.symlink_to(tmp_path, target_is_directory=True)
    try:
        storage.owned_target(root, "experiment", "qwen2b", "adapter")
    except ValueError:
        pass
    else:
        raise AssertionError("Symlinked run root was accepted")


def test_resume_truncates_only_incomplete_final_row(tmp_path):
    output = tmp_path / "shard_00.jsonl"
    output.write_bytes(b'{"image_id": "1"}\n{"image_id": "2"')
    assert generate.completed_ids(output, {"1", "2"}) == {"1"}
    assert output.read_bytes() == b'{"image_id": "1"}\n'
    output.write_bytes(b'{"image_id": "1"}\nnot json\n')
    try:
        generate.completed_ids(output, {"1", "2"})
    except ValueError as error:
        assert "Corrupt JSONL" in str(error)
    else:
        raise AssertionError("Complete corrupt row was silently dropped")


def test_plan_reproducible_across_calls():
    row = {"image_id": "77", "image_path": "/image.jpg", "prompt": "Describe it.",
           "caption": "A cat sits. A person walks. The room is bright. The floor is clear. The scene is calm."}
    parsed = {"node_objects": ["person", "cat"]}
    graph = {"person": {"chair": 8, "table": 3}, "cat": {"table": 5, "chair": 2}}
    synonyms = {"person": ["person"], "cat": ["cat"], "chair": ["chair"], "table": ["table"]}
    first = prepare.make_plan(row, parsed, graph, synonyms, 0.2, 42)
    assert first == prepare.make_plan(row, parsed, graph, synonyms, 0.2, 42)
