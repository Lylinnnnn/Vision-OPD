import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "baselines/rlsf_v"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_coco5k")
audit = load_script("audit_generation")
materialize = load_script("materialize")
check_env = load_script("check_env")
sync_oss = load_script("sync_oss")


class RlsfCoco5kTests(unittest.TestCase):
    def test_oss_stat_content_length(self):
        self.assertEqual(sync_oss.content_length("Content-Length : 4877470336\n"), 4877470336)
        with self.assertRaises(ValueError):
            sync_oss.content_length("Last-Modified : today")

    def test_prune_keeps_final_adapter_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_config.json").write_text("{}")
            (root / "adapter_model.safetensors").write_bytes(b"weights")
            (root / "notes.txt").write_text("keep")
            checkpoint = root / "checkpoint-50"
            checkpoint.mkdir()
            (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
            sync_oss.prune_training_checkpoints(root)
            self.assertFalse(checkpoint.exists())
            self.assertTrue((root / "adapter_model.safetensors").is_file())
            self.assertTrue((root / "notes.txt").is_file())

    def test_prune_requires_final_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-50"
            checkpoint.mkdir()
            with self.assertRaisesRegex(RuntimeError, "Refusing to prune"):
                sync_oss.prune_training_checkpoints(root)
            self.assertTrue(checkpoint.is_dir())

    def test_upload_restore_and_cleanup_touch_only_selected_model(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            work = base / "rlsf_v_coco5k"
            adapter = work / "qwen2b/adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            checkpoint = adapter / "checkpoint-50"
            checkpoint.mkdir()
            (checkpoint / "optimizer.pt").write_bytes(b"old")
            other = work / "qwen4b/adapter"
            other.mkdir(parents=True)
            (other / "keep.txt").write_text("untouched")
            remote = base / "fake_oss"
            uri = "oss://test/rlsf_v_coco5k/qwen2b/adapter"

            def mapped(value):
                value = str(value)
                return remote / value.removeprefix("oss://test/") if value.startswith("oss://") else Path(value)

            def copy(source, destination):
                target = mapped(destination)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mapped(source), target)

            with patch.object(sync_oss, "require_ossutil"), \
                    patch.object(sync_oss, "oss_copy", side_effect=copy), \
                    patch.object(sync_oss, "oss_size", side_effect=lambda value: mapped(value).stat().st_size):
                sync_oss.sync_tree(adapter, uri, "adapter")
                sync_oss.remove_owned_tree(adapter, work, "qwen2b", "adapter")
                self.assertFalse(adapter.exists())
                self.assertTrue((other / "keep.txt").is_file())
                sync_oss.restore_tree(uri, adapter, "adapter")
            self.assertEqual((adapter / "adapter_model.safetensors").read_bytes(), b"weights")
            self.assertFalse((adapter / "checkpoint-50").exists())
            self.assertTrue((other / "keep.txt").is_file())
            with self.assertRaisesRegex(ValueError, "outside"):
                sync_oss.remove_owned_tree(other, work, "qwen2b", "adapter")

    def test_upload_failure_does_not_delete_local_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "rlsf_v_smoke"
            adapter = work / "qwen2b/adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            argv = ["sync_oss.py", "upload", "--source", str(adapter),
                    "--destination", "oss://test/adapter", "--work-root", str(work),
                    "--model", "qwen2b", "--artifact", "adapter", "--delete-source-after-sync"]
            with patch.object(sys, "argv", argv), patch.object(sync_oss, "require_ossutil"), \
                    patch.object(sync_oss, "oss_copy", side_effect=RuntimeError("network failed")):
                with self.assertRaisesRegex(RuntimeError, "network failed"):
                    sync_oss.main()
            self.assertTrue((adapter / "adapter_model.safetensors").is_file())

    def test_restore_rejects_manifest_traversal(self):
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            sync_oss.valid_manifest({"schema_version": 1, "files": [
                {"path": "../other/adapter_model.safetensors", "size": 1}
            ]}, "adapter")

    def test_merged_manifest_requires_model_weights(self):
        with self.assertRaisesRegex(ValueError, "missing merged weights"):
            sync_oss.valid_manifest({"schema_version": 1, "files": [
                {"path": "config.json", "size": 2},
                {"path": "training_args.bin", "size": 4},
            ]}, "merged")
        self.assertEqual(len(sync_oss.valid_manifest({"schema_version": 1, "files": [
            {"path": "config.json", "size": 2},
            {"path": "model-00001-of-00002.safetensors", "size": 4},
        ]}, "merged")), 2)

    def test_preflight_covers_generation_and_conversion_dependencies(self):
        self.assertIn("shortuuid", check_env.MODULES["pipeline"])
        self.assertIn("spacy", check_env.MODULES["pipeline"])
        self.assertIn("pyarrow", check_env.MODULES["pipeline"])

    def test_converter_keeps_training_prompt_and_mounted_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "COCO/train2017/000000000001.jpg"
            image.parent.mkdir(parents=True)
            image.touch()
            frame = pd.DataFrame([{
                "images": [{"path": "/stale/000000000001.jpg"}],
                "extra_info": {"image_id": 1},
                "prompt": [{"role": "user", "content": "<image>Describe this image."}],
            }])
            rows = prepare.convert_frame(frame, {2}, root)
            self.assertEqual(rows, [{
                "image_id": "1", "image_path": str(image),
                "question": "Describe this image.",
            }])
            with self.assertRaisesRegex(ValueError, "test-set"):
                prepare.convert_frame(frame, {1}, root)

    def test_generation_audit_rejects_missing_samples(self):
        questions = [{"image_id": "1"}]
        answers = [{"question_id": "1", "answer_id": "a", "text": "A dog."}]
        self.assertEqual(audit.audit(questions, answers, 1), (1, 1))
        with self.assertRaisesRegex(ValueError, "Incomplete generation"):
            audit.audit(questions, answers, 2)

    def test_preference_matches_original_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "000000000001.jpg"
            image.touch()
            questions = root / "questions.jsonl"
            questions.write_text(json.dumps({
                "image_id": "1", "image_path": str(image), "question": "Describe this image."
            }) + "\n")
            preferences = root / "dpo.json"
            preferences.write_text(json.dumps([{
                "img_path": str(image), "prompt": "Describe this image.",
                "chosen": "A dog.", "rejected": "A cat.",
            }]))
            self.assertEqual(len(materialize.load_preferences(preferences, questions)), 1)
            preferences.write_text(json.dumps([{
                "img_path": str(image), "prompt": "Different prompt.",
                "chosen": "A dog.", "rejected": "A cat.",
            }]))
            with self.assertRaisesRegex(ValueError, "not from the COCO 5k"):
                materialize.load_preferences(preferences, questions)

    def test_one_epoch_control_reuses_original_preferences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "rlsf_v_coco5k"
            control = root / "rlsf_v_coco5k_e1/qwen2b"
            template = root / "upstream/train/hf_series/LlamaFactory/qwen3.yaml"
            template.parent.mkdir(parents=True)
            template.write_text(json.dumps({"num_train_epochs": 2.0}), encoding="utf-8")
            image = root / "image.jpg"
            image.touch()
            questions = full / "questions.jsonl"
            questions.parent.mkdir(parents=True)
            questions.write_text(json.dumps({"image_path": str(image), "question": "Describe the image."}) + "\n")
            dpo = full / "qwen2b/generate/qwen2b_dpo_data.json"
            dpo.parent.mkdir(parents=True)
            dpo.write_text(json.dumps([{
                "img_path": str(image), "prompt": "Describe the image.",
                "chosen": "A cat.", "rejected": "A dog.",
            }]), encoding="utf-8")
            yaml_stub = SimpleNamespace(safe_load=json.loads, safe_dump=lambda data, **kwargs: json.dumps(data))
            argv = ["materialize.py", "train", "--model", "qwen2b", "--model-path", str(root),
                    "--upstream", str(root / "upstream"), "--run-dir", str(control),
                    "--questions", str(questions)]
            with patch.dict(sys.modules, {"yaml": yaml_stub}), patch.object(sys, "argv", argv), \
                    patch.dict("os.environ", {"RLSF_V_DPO_DATA": str(dpo), "RLSF_V_TRAIN_EPOCHS": "1"}):
                materialize.main()
            config = json.loads((control / "train.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["num_train_epochs"], 1.0)
            self.assertEqual(len(json.loads((control / "dataset/preferences.json").read_text())), 1)


if __name__ == "__main__":
    unittest.main()
