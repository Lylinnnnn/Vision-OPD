#!/usr/bin/env python3
"""Local OpenAI-compatible bridge to a pinned external decoding adapter."""

import argparse
import base64
import hashlib
import io
import json
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace


def parse_request(payload):
    messages = payload.get("messages") or []
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("Expected one user message")
    content = messages[0].get("content") or []
    if not isinstance(content, list):
        raise ValueError("Expected multimodal message content")
    texts = [part.get("text", "") for part in content if part.get("type") == "text"]
    images = [part.get("image_url", {}).get("url", "") for part in content
              if part.get("type") == "image_url"]
    if len(images) != 1 or not images[0].startswith("data:image/") or ";base64," not in images[0]:
        raise ValueError("Expected one base64 image data URI")
    image_bytes = base64.b64decode(images[0].split(";base64,", 1)[1], validate=True)
    if not image_bytes or len(image_bytes) > 32 * 1024 * 1024:
        raise ValueError("Image is empty or exceeds 32 MiB")
    prompt = "\n".join(text.strip() for text in texts if text.strip())
    if not prompt:
        raise ValueError("Prompt is empty")
    max_tokens = int(payload.get("max_tokens", 128))
    if not 1 <= max_tokens <= 8192:
        raise ValueError("max_tokens must be between 1 and 8192")
    stable_id = int.from_bytes(hashlib.sha256(image_bytes + prompt.encode()).digest()[:4], "big")
    return image_bytes, prompt, max_tokens, stable_id


class Decoder:
    def __init__(self, method, model_path, config_path, detector_config=None):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.method = method
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if method in ("vcd", "opera"):
            if self.config.get("model_name") != model_path.name:
                raise ValueError(f"Model identity mismatch in {config_path}")
        elif Path(self.config.get("model_path", "")).resolve() != model_path.resolve():
            raise ValueError(f"HALC model path mismatch in {config_path}")
        self.args = SimpleNamespace(**self.config)
        self.args.model_family = "qwen3vl"
        self.device = torch.device("cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for external decoding")
        dtype_name = self.config.get("torch_dtype", "bfloat16")
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        model_kwargs = {"attn_implementation": "eager"} if method == "opera" else {}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
            **model_kwargs,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.halc = None
        if method == "vcd":
            from eval_vcd_chair import generate_caption
            self.generate_fn = generate_caption
        elif method == "opera":
            from eval_opera_chair import generate_caption
            self.generate_fn = generate_caption
        else:
            if detector_config is None:
                raise ValueError("HALC requires --detector-config")
            from eval_halc_chair import preflight
            from halc_qwen3vl import HALCConfig, HALCDecoder

            preflight(SimpleNamespace(
                model_family="qwen3vl", model_path=str(model_path),
                test_json=str(Path(__file__).resolve().parents[1] / "data/test_1000.json"),
                detector_config=str(detector_config),
                detector_weights=self.config["detector_weights"],
                blip_model_path=self.config["blip_model_path"],
            ))
            halc_config = HALCConfig(**self.config["parameters"])
            self.halc = HALCDecoder(
                self.model, self.processor, str(detector_config),
                self.config["detector_weights"], self.config["blip_model_path"],
                halc_config, self.device,
            )
        self.scratch = tempfile.TemporaryDirectory(prefix=f"resopd_{method}_api_")

    def generate(self, image_bytes, prompt, max_tokens, stable_id):
        from PIL import Image

        if self.method == "halc":
            path = Path(self.scratch.name) / f"{stable_id}.jpg"
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.verify()
            path.write_bytes(image_bytes)
            try:
                self.halc.config.max_new_tokens = max_tokens
                result = self.halc.generate(
                    str(path), prompt=prompt, seed=int(self.config["seed"]) + stable_id,
                )
            finally:
                path.unlink(missing_ok=True)
        else:
            self.args.prompt = prompt
            self.args.max_new_tokens = max_tokens
            with Image.open(io.BytesIO(image_bytes)) as image:
                if self.method == "vcd":
                    result = self.generate_fn(
                        self.model, self.processor, self.device,
                        image.convert("RGB"), self.args, stable_id,
                    )
                else:
                    result = self.generate_fn(
                        self.model, self.processor, self.device,
                        image.convert("RGB"), self.args,
                    )
        text = result["generated_caption"]
        if not text:
            raise RuntimeError("Decoder produced an empty answer")
        return text, int(result.get("generated_token_count", 0))


class Handler(BaseHTTPRequestHandler):
    def write_json(self, code, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.write_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self.write_json(200, {"data": [{"id": self.server.model_id, "object": "model"}]})
        else:
            self.write_json(404, {"error": {"message": "Unknown route"}})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.write_json(404, {"error": {"message": "Unknown route"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 48 * 1024 * 1024:
                raise ValueError("Request body size invalid")
            payload = json.loads(self.rfile.read(length))
            if payload.get("model") != self.server.model_id:
                raise ValueError("Requested model ID does not match this decoder")
            image_bytes, prompt, max_tokens, stable_id = parse_request(payload)
            answer, generated_tokens = self.server.decoder.generate(
                image_bytes, prompt, max_tokens, stable_id,
            )
            self.write_json(200, {
                "id": f"chatcmpl-{stable_id}", "object": "chat.completion",
                "created": int(time.time()), "model": self.server.model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": generated_tokens,
                          "total_tokens": generated_tokens},
            })
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            traceback.print_exc()
            try:
                self.write_json(500, {"error": {"message": f"{type(error).__name__}: {error}"}})
            except (BrokenPipeError, ConnectionResetError):
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("vcd", "opera", "halc"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--detector-config", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not args.config.is_file() or not args.model_path.is_dir():
        parser.error("Model and pinned CHAIR configuration must exist")
    decoder = Decoder(args.method, args.model_path, args.config, args.detector_config)
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    server.decoder = decoder
    server.model_id = args.model_id
    print(f"EXTERNAL_DECODE_API=READY method={args.method} port={args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
