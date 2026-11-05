"""HF InternVL3 image inputs for external decoding baselines."""

import json
from pathlib import Path

import torch


def require_checkpoint(model_path):
    root = Path(model_path)
    config = json.loads((root / "config.json").read_text())
    processor = json.loads((root / "processor_config.json").read_text())
    if config.get("model_type") != "internvl":
        raise ValueError(f"Expected HF InternVL checkpoint, got {config.get('model_type')!r}")
    if "InternVLForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError("Checkpoint is not InternVLForConditionalGeneration")
    if processor.get("processor_class") != "InternVLProcessor":
        raise ValueError("Checkpoint is not paired with InternVLProcessor")


def build_inputs(processor, image, prompt, device=None, dtype=None):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    validate_image_alignment(inputs, processor)
    if device is not None:
        inputs = {key: value.to(device=device, dtype=dtype)
                  if torch.is_tensor(value) and value.is_floating_point() and dtype is not None
                  else value.to(device) if torch.is_tensor(value) else value
                  for key, value in inputs.items()}
    return inputs


def validate_image_alignment(inputs, processor):
    input_ids = inputs.get("input_ids")
    pixel_values = inputs.get("pixel_values")
    image_token_id = getattr(processor, "image_token_id", None)
    image_seq_length = getattr(processor, "image_seq_length", None)
    if input_ids is None or pixel_values is None or image_token_id is None or image_seq_length is None:
        raise ValueError("InternVL3 inputs lack image tokens, pixel values, or processor metadata")
    expected = pixel_values.shape[0] * int(image_seq_length)
    actual = int((input_ids == image_token_id).sum().item())
    if actual != expected:
        raise ValueError(f"InternVL3 image alignment mismatch: tokens={actual}, expected={expected}")


def build_vcd_inputs(processor, image, prompt, noise_step, noise_seed):
    from vcd_utils.vcd_add_noise import add_diffusion_noise

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        [messages, messages], padding=True, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    validate_image_alignment(inputs, processor)
    if inputs["input_ids"].shape[0] != 2 or not torch.equal(inputs["input_ids"][0], inputs["input_ids"][1]):
        raise RuntimeError("InternVL3 VCD text branches differ")
    pixels = inputs["pixel_values"]
    if pixels.shape[0] % 2 or pixels.shape[0] == 0:
        raise RuntimeError("InternVL3 VCD image patches cannot be split evenly")
    midpoint = pixels.shape[0] // 2
    tokens_per_branch = int((inputs["input_ids"] == processor.image_token_id).sum(dim=1)[0])
    if tokens_per_branch != midpoint * int(processor.image_seq_length):
        raise RuntimeError("InternVL3 VCD visual patches do not match each text branch")
    if not torch.equal(pixels[:midpoint], pixels[midpoint:]):
        raise RuntimeError("InternVL3 VCD image patches differ before noise injection")
    if not 0 <= noise_step <= 999:
        raise ValueError("noise_step must be in [0, 999]")
    pixels = pixels.clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(noise_seed))
        pixels[midpoint:] = add_diffusion_noise(pixels[midpoint:].clone(), int(noise_step))
    if torch.equal(pixels[:midpoint], pixels[midpoint:]):
        raise RuntimeError("InternVL3 VCD noise did not change the visual branch")
    inputs["pixel_values"] = pixels
    return inputs


def next_token_logits(model, inputs, generated):
    kwargs = dict(inputs)
    if generated:
        continuation = torch.tensor([generated], dtype=kwargs["input_ids"].dtype,
                                    device=kwargs["input_ids"].device)
        kwargs["input_ids"] = torch.cat((kwargs["input_ids"], continuation), dim=1)
        kwargs["attention_mask"] = torch.cat(
            (kwargs["attention_mask"], torch.ones_like(continuation)), dim=1
        )
    if kwargs["input_ids"].shape != kwargs["attention_mask"].shape:
        raise ValueError("InternVL3 continuation and attention mask are misaligned")
    output = model.generate(
        **kwargs, max_new_tokens=1, do_sample=False, num_beams=1,
        return_dict_in_generate=True, output_scores=True, use_cache=False,
    )
    if len(output.scores) != 1:
        raise RuntimeError("InternVL3 did not return one next-token score tensor")
    return output.scores[0][0]
