import base64
import importlib.util
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BASELINES = Path(__file__).resolve().parents[1] / "baselines"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


API = load_module("external_decode_api_test", BASELINES / "external_decode_api.py")
AUX = load_module("run_external_aux_benchmarks_test", BASELINES / "run_external_aux_benchmarks.py")


class ExternalAuxBenchmarksTest(unittest.TestCase):
    def test_benchmark_selection(self):
        self.assertEqual(AUX.parse_benchmarks("pope,amber"), ("pope", "amber"))
        for value in ("", "pope,pope", "chair"):
            with self.assertRaises(ValueError):
                AUX.parse_benchmarks(value)

    def test_qwen_4b_8b_preflight_for_pope_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            (data / "COCO/coco2017val").mkdir(parents=True)
            repo = root / "repo"
            (repo / "data").mkdir(parents=True)
            (repo / "data/test_1000.json").write_text("[]")
            for size in ("4B", "8B"):
                model = root / f"Qwen3VL-{size}-Instruct"
                model.mkdir()
                (model / "config.json").write_text("{}")
                (model / "model.safetensors.index.json").write_text("{}")
                args = SimpleNamespace(model_path=model, data_root=data,
                                       detector_config=root / "unused")
                with patch.object(AUX, "RES_OPD", repo), \
                     patch.object(AUX, "result_dir", return_value=root / "chair"), \
                     patch.object(AUX, "validate_chair_anchor", return_value=({}, 128)):
                    result = AUX.validate_inputs(args, ["opera"], ("pope",))
                self.assertEqual(result["opera"][2], 128)

    def test_standalone_opera_does_not_need_chair_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "Qwen3VL-4B-Instruct"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "model.safetensors.index.json").write_text("{}")
            data = root / "data"
            (data / "COCO/coco2017val").mkdir(parents=True)
            repo = root / "repo"
            (repo / "data").mkdir(parents=True)
            (repo / "data/test_1000.json").write_text("[]")
            output = root / "results"
            args = SimpleNamespace(model_path=model, data_root=data,
                                   detector_config=root / "unused", detector_weights=None,
                                   blip_model_path=None, standalone=True,
                                   preflight_only=True, max_new_tokens=8192,
                                   results_root=output, opera_dir=None)
            with patch.object(AUX, "RES_OPD", repo):
                configs = AUX.validate_inputs(args, ["opera"], ("pope",), 8)
                self.assertEqual(configs["opera"][2], 8192)
                self.assertFalse((output / AUX.METHOD_SUFFIX["opera"] / "opera_config.json").exists())
                args.preflight_only = False
                AUX.validate_inputs(args, ["opera"], ("pope",), 8)
            config = json.loads((output / AUX.METHOD_SUFFIX["opera"] / "opera_config.json").read_text())
            self.assertEqual(config["model_name"], model.name)
            self.assertEqual(config["num_beams"], 5)
            config["num_beams"] = 7
            (output / AUX.METHOD_SUFFIX["opera"] / "opera_config.json").write_text(json.dumps(config))
            with patch.object(AUX, "RES_OPD", repo), \
                 self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                AUX.validate_inputs(args, ["opera"], ("pope",), 8)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "HALC config needs torch")
    def test_standalone_halc_matches_sharded_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(model_path=root / "Qwen3VL-8B-Instruct",
                                   detector_weights=root / "groundingdino_swint_ogc.pth",
                                   blip_model_path=root / "blip-itm-base-coco",
                                   max_new_tokens=8192)
            config = AUX.standalone_config(args, "halc", None, 8)
            self.assertEqual(config["method"], "halc_qwen3vl_adaptation")
            self.assertEqual(config["parameters"]["max_new_tokens"], 8192)
            self.assertEqual(config["parameters"]["beam_size"], 3)
            self.assertEqual((config["shard_count"], config["shard_index"]), (8, 0))

    def test_parse_multimodal_request(self):
        image = b"test-image"
        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                 base64.b64encode(image).decode()}},
                {"type": "text", "text": "Is there a dog?"},
            ]}],
            "max_tokens": 16,
        }
        parsed = API.parse_request(payload)
        self.assertEqual(parsed[:3], (image, "Is there a dog?", 16))
        self.assertEqual(parsed[3], API.parse_request(payload)[3])

    def test_parse_rejects_multiple_images(self):
        uri = "data:image/jpeg;base64," + base64.b64encode(b"image").decode()
        payload = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": "Describe the image."},
        ]}]}
        with self.assertRaisesRegex(ValueError, "one base64 image"):
            API.parse_request(payload)

    def test_openai_completion_contract(self):
        class FakeDecoder:
            def generate(self, _image, _prompt, _tokens, _stable_id):
                return "A dog.", 3

        payload = {
            "model": "external-vcd-test", "max_tokens": 16,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                 base64.b64encode(b"image").decode()}},
                {"type": "text", "text": "Describe the image."},
            ]}],
        }
        body = json.dumps(payload).encode()
        handler = object.__new__(API.Handler)
        handler.path = "/v1/chat/completions"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        handler.server = SimpleNamespace(model_id=payload["model"], decoder=FakeDecoder())
        status = []
        handler.send_response = status.append
        handler.send_header = lambda _key, _value: None
        handler.end_headers = lambda: None
        handler.do_POST()
        result = json.loads(handler.wfile.getvalue())
        self.assertEqual(status, [200])
        self.assertEqual(result["choices"][0]["message"]["content"], "A dog.")
        self.assertEqual(result["usage"]["completion_tokens"], 3)

    def test_anchor_requires_complete_official_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "vcd_config.json").write_text(json.dumps({
                "model_name": "Qwen3VL-2B-Instruct", "max_new_tokens": 128,
            }))
            (directory / "chair_metrics.json").write_text(json.dumps({
                "metric_schema": "official_chair_caption_and_mention_v1",
                "total_results": 1000, "eval_prompt": "Describe the image.",
            }))
            (directory / "eval_results.jsonl").write_text(
                '{"generated_caption":"A dog."}\n' * 1000
            )
            _, tokens = AUX.validate_chair_anchor(
                directory, "vcd", Path("/model/Qwen3VL-2B-Instruct")
            )
            self.assertEqual(tokens, 128)
            (directory / "chair_metrics.json").write_text(json.dumps({
                "metric_schema": "legacy", "total_results": 1000,
                "eval_prompt": "Describe the image.",
            }))
            with self.assertRaisesRegex(RuntimeError, "official 1000-image"):
                AUX.validate_chair_anchor(directory, "vcd", Path("/model/Qwen3VL-2B-Instruct"))

    def test_anchor_discovery_refuses_ambiguity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("first", "second"):
                (root / name).mkdir()
                (root / name / "vcd_config.json").touch()
            args = SimpleNamespace(vcd_dir=None, results_root=root)
            with self.assertRaisesRegex(RuntimeError, "Cannot uniquely locate"):
                AUX.result_dir(args, "vcd")
            args.vcd_dir = root / "first"
            self.assertEqual(AUX.result_dir(args, "vcd"), args.vcd_dir)

    def test_opera_api_preserves_eager_attention_loading(self):
        source = (BASELINES / "external_decode_api.py").read_text()
        self.assertIn('"attn_implementation": "eager"', source)

    def test_merged_judge_is_json_array(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pope").mkdir()
            (root / "pope/pope_summary.json").write_text(json.dumps({
                "benchmarks": {name: {} for name in AUX.POPE_SPLITS},
            }))
            (root / "amber").mkdir()
            (root / "amber/amber_metrics.json").write_text(json.dumps({
                "official_returncode": 0, "evaluation_type": "a",
            }))
            model_id = "external-vcd-Qwen3VL-2B-Instruct"
            for bench in AUX.VISION_COUNTS:
                judge = root / bench / "judge"
                judge.mkdir(parents=True)
                (judge / f"{model_id}_seed42_answer.jsonl").write_text(
                    json.dumps([{"judge": "Yes"}])
                )
            with patch.object(AUX, "VISION_COUNTS", {"mmstar": 1, "cv-bench": 1}), \
                 patch.object(AUX, "count_jsonl", side_effect=lambda path, _field=None:
                              15220 if "amber" in str(path) else
                              5223 if "pope" in str(path) else 1):
                AUX.validate_merged(root, model_id)

    def test_selected_benchmarks_do_not_require_vision_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pope").mkdir()
            (root / "pope/pope_summary.json").write_text(json.dumps({
                "benchmarks": {name: {} for name in AUX.POPE_SPLITS},
            }))
            (root / "amber").mkdir()
            (root / "amber/amber_metrics.json").write_text(json.dumps({
                "official_returncode": 0, "evaluation_type": "a",
            }))
            with patch.object(AUX, "count_jsonl", side_effect=lambda path, _field=None:
                             15220 if "amber" in str(path) else 5223):
                AUX.validate_merged(root, "external-opera-Qwen3VL-4B-Instruct",
                                    ("pope", "amber"))

    def test_pope_amber_selection_does_not_launch_vision_eval(self):
        class FinishedProcess:
            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "shard_00").mkdir()
            args = SimpleNamespace(base_port=24100, data_root=root)
            with patch.object(AUX.subprocess, "Popen", return_value=FinishedProcess()) as popen:
                AUX.evaluate_shard(args, "opera", root, 0, 8,
                                   "external-opera-Qwen3VL-4B-Instruct", 8192,
                                   set(), threading.Lock(), ("pope", "amber"))
            commands = [call.args[0] for call in popen.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertTrue(commands[0][1].endswith("eval_pope.py"))
            self.assertTrue(commands[1][1].endswith("eval_amber.py"))

    def test_capacity_wait_does_not_kill_busy_jobs(self):
        capacity = sys.modules["wait_eval_capacity"]
        with patch.object(capacity, "free_gpu_memory", return_value={"0": 20000}), \
             patch.object(capacity, "port_is_free", return_value=False), \
             patch.object(capacity.time, "monotonic", side_effect=[0, 2]):
            with self.assertRaisesRegex(TimeoutError, "Capacity wait expired"):
                capacity.wait_for_capacity(["0"], [24100], 36000, 1)


if __name__ == "__main__":
    unittest.main()
