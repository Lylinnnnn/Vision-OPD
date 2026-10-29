import math

import torch
from transformers.generation.logits_process import LogitsProcessor


class VCDLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        alpha=1.0,
        beta=0.1,
        decode_mode="sample",
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        seed=42,
    ):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0 < beta <= 1:
            raise ValueError("beta must satisfy 0 < beta <= 1")
        if decode_mode not in {"greedy", "sample"}:
            raise ValueError("decode_mode must be greedy or sample")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must satisfy 0 < top_p <= 1")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.decode_mode = decode_mode
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.seed = int(seed)
        self.generator = None
        self.generator_device = None

    def _generator_for(self, device):
        device_key = str(device)
        if self.generator is None or self.generator_device != device_key:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(self.seed)
            self.generator_device = device_key
        return self.generator

    def _sample_token(self, logits):
        warped = logits / self.temperature
        if self.top_k > 0:
            k = min(self.top_k, warped.shape[-1])
            threshold = torch.topk(warped, k=k).values[..., -1, None]
            warped = warped.masked_fill(warped < threshold, -float("inf"))
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(warped, descending=True)
            cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_remove = cumulative > self.top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(sorted_remove).scatter(
                -1, sorted_indices, sorted_remove
            )
            warped = warped.masked_fill(remove, -float("inf"))
        probs = torch.softmax(warped, dim=-1)
        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            raise RuntimeError("VCD sampling produced invalid probabilities")
        return torch.multinomial(
            probs,
            num_samples=1,
            generator=self._generator_for(logits.device),
        ).item()

    def __call__(self, input_ids, scores):
        if scores.ndim != 2 or scores.shape[0] != 2:
            raise ValueError(
                "VCD Qwen3VL adapter expects exactly two rows: original and noisy"
            )
        original = scores[0].float()
        noisy = scores[1].float()
        cutoff = original.max() + math.log(self.beta)
        combined = (1.0 + self.alpha) * original - self.alpha * noisy
        combined = combined.masked_fill(original < cutoff, -float("inf"))
        combined = combined.masked_fill(~torch.isfinite(combined), -float("inf"))
        if not torch.isfinite(combined).any():
            raise RuntimeError("VCD plausibility constraint removed every token")

        if self.decode_mode == "sample":
            token_id = self._sample_token(combined)
            forced = torch.full_like(scores, -float("inf"))
            forced[:, token_id] = 0
            return forced

        combined = combined.to(dtype=scores.dtype)
        return torch.stack((combined, combined), dim=0)


def replace_second_image_with_diffusion_noise(inputs, noise_step, noise_seed):
    if not 0 <= noise_step <= 999:
        raise ValueError("noise_step must be in [0, 999]")
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    if pixel_values is None or image_grid_thw is None:
        raise ValueError("Qwen3VL processor output lacks pixel_values/image_grid_thw")
    if image_grid_thw.shape[0] != 2:
        raise ValueError("VCD adapter expects exactly two processed images")
    if not torch.equal(image_grid_thw[0], image_grid_thw[1]):
        raise ValueError("Original and noisy branches must have identical image grids")

    counts = [int(torch.prod(row).item()) for row in image_grid_thw]
    if sum(counts) != pixel_values.shape[0]:
        if pixel_values.shape[0] % 2 != 0:
            raise ValueError(
                "Cannot locate the second image inside flattened Qwen3VL pixel values"
            )
        counts = [pixel_values.shape[0] // 2, pixel_values.shape[0] // 2]
    start = counts[0]
    stop = start + counts[1]

    from vcd_utils.vcd_add_noise import add_diffusion_noise

    updated = pixel_values.clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(noise_seed))
        updated[start:stop] = add_diffusion_noise(
            pixel_values[start:stop].clone(),
            int(noise_step),
        )
    inputs["pixel_values"] = updated
    return inputs
