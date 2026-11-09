import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts/tmp"
sys.path.insert(0, str(SCRIPT_DIR))
import check_main_tr05_pope_amber as audit  # noqa: E402


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PopeAmberStatusTest(unittest.TestCase):
    def test_pope_distinguishes_missing_partial_unscored_and_scored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(audit.inspect_pope(root, 2).state, "未推理")
            write_rows(root / "pope/pope_adv/eval_results.jsonl",
                       [{"sample_uid": "adv-1", "model_answer": "Yes"}])
            self.assertEqual(audit.inspect_pope(root, 2).state, "推理未完成")
            for split in audit.SPLITS:
                write_rows(root / "pope" / split / "eval_results.jsonl", [
                    {"sample_uid": f"{split}-{number}", "model_answer": "Yes"}
                    for number in range(2)
                ])
            self.assertEqual(audit.inspect_pope(root, 2).state, "已推理待计算")
            write_json(root / "pope/pope_summary.json", {
                "macro_f1": 0.5,
                "benchmarks": {split: {"f1": 0.5} for split in audit.SPLITS},
            })
            self.assertEqual(audit.inspect_pope(root, 2).state, "已推理有结果")

    def test_amber_distinguishes_merge_and_official_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(audit.inspect_amber(root, 2).state, "未推理")
            raw = root / "amber/raw_results.jsonl"
            write_rows(raw, [{"id": 1, "model_answer": "A dog."}])
            self.assertEqual(audit.inspect_amber(root, 2).state, "推理未完成")
            write_rows(raw, [{"id": 1, "model_answer": "A dog."},
                             {"id": 2, "model_answer": "A cat."}])
            self.assertEqual(audit.inspect_amber(root, 2).state, "已推理待合并计算")
            write_json(root / "amber/amber_a_responses.json", [
                {"id": 1, "response": "A dog."}, {"id": 2, "response": "A cat."}
            ])
            self.assertEqual(audit.inspect_amber(root, 2).state, "已推理待计算")
            write_json(root / "amber/amber_metrics.json", {
                "official_returncode": 0, "evaluation_type": "a",
                "metrics": {"generative_task_chair": 5.5, "generative_task_hal": 37.5},
            })
            self.assertEqual(audit.inspect_amber(root, 2).state, "已推理有结果")

    def test_shards_are_not_mistaken_for_absent_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in (0, 1):
                out = root / ".aux_shards/external_aux_pope_amber_standalone" / f"shard_{index:02d}/out"
                for split in audit.SPLITS:
                    write_rows(out / "pope" / split / "eval_results.jsonl", [
                        {"sample_uid": f"{split}-{index}", "model_answer": "Yes"}
                    ])
            result = audit.inspect_all([root], "pope", 2)
            self.assertEqual(result.state, "已推理待合并计算")

    def test_external_methods_do_not_read_each_others_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            eval_root = Path(temporary)
            dataset = eval_root / "external_baselines/full/Qwen3VL-2B-Instruct" / audit.DATASET
            (dataset / audit.EXTERNAL_SUFFIX["HALC"]).mkdir(parents=True)
            (dataset / audit.EXTERNAL_SUFFIX["OPERA"]).mkdir()
            halc = audit.Target("M088", "2B", "HALC")
            opera = audit.Target("OPERA-2B", "2B", "OPERA")
            self.assertEqual(audit.target_dirs(halc, eval_root), [dataset / audit.EXTERNAL_SUFFIX["HALC"]])
            self.assertEqual(audit.target_dirs(opera, eval_root), [dataset / audit.EXTERNAL_SUFFIX["OPERA"]])

    def test_requested_targets_are_unique_and_explicit(self):
        self.assertEqual(len(audit.TARGETS), 19)
        self.assertEqual(len({target.row_id for target in audit.TARGETS}), 19)
        self.assertIn(audit.Target("M064", "8B", "EntropyMask", 250), audit.TARGETS)

    def test_internvl_base_vision_statuses_and_legacy_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "internvl/full/InternVL3-2B-hf/base_chair_original_1000_8gpu"
            legacy.mkdir(parents=True)
            directories = audit.internvl_base_dirs(root)
            self.assertIn(legacy, directories)
            self.assertEqual(audit.inspect_vision_all(directories, "mmstar", 2).state, "未推理")
            answer = legacy / "vision_opd/model_answer/mmstar/base_seed42_answer.jsonl"
            write_rows(answer, [{"question_id": 1, "model_answer": "A"},
                                {"question_id": 2, "model_answer": "B"}])
            result = audit.inspect_vision_all(directories, "mmstar", 2)
            self.assertEqual(result.state, "已推理待计算")
            judge = legacy / "vision_opd/judge/mmstar/base_seed42_answer.jsonl"
            write_rows(judge, [{"judge": "yes"}, {"judge": "no"}])
            result = audit.inspect_vision_all(directories, "mmstar", 2)
            self.assertEqual(result.state, "已推理有结果")
            self.assertEqual(result.values, ("50.00",))
            self.assertIn("MMStar", audit.render([], [("mmstar", result)]))

    def test_internvl_base_vision_shard_answers_are_not_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "internvl3/full/InternVL3-2B-hf" / audit.DATASET / "official_v2"
            write_rows(canonical / ".shards/run/shard_00/out/cv-bench/model_answer/base_answer.jsonl",
                       [{"question_id": 1, "model_answer": "A"}])
            result = audit.inspect_vision_all(audit.internvl_base_dirs(root), "cv-bench", 2)
            self.assertEqual(result.state, "推理未完成")
            write_rows(canonical / ".shards/run/shard_01/out/cv-bench/model_answer/base_answer.jsonl",
                       [{"question_id": 2, "model_answer": "B"}])
            result = audit.inspect_vision_all(audit.internvl_base_dirs(root), "cv-bench", 2)
            self.assertEqual(result.state, "已推理待合并计算")


if __name__ == "__main__":
    unittest.main()
