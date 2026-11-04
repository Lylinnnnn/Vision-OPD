"""Token and visual-input checks for HF-format InternVL rollouts."""


def is_internvl_hf_processor(processor):
    return (
        processor is not None
        and type(processor).__name__ == "InternVLProcessor"
        and getattr(getattr(processor, "config", None), "model_type", None) == "internvl"
    )


def internvl_vllm_prompt(raw_prompt, processor, image_count):
    marker = processor.image_token
    if raw_prompt.count(marker) != image_count:
        raise ValueError(
            f"InternVL prompt has {raw_prompt.count(marker)} image markers for {image_count} images"
        )
    if "<image>" in raw_prompt:
        raise ValueError("InternVL prompt already contains a non-HF image placeholder")
    return raw_prompt


def assert_internvl_image_alignment(input_ids, pixel_values, processor, context):
    image_tokens = int((input_ids == processor.image_token_id).sum().item())
    expected = pixel_values.shape[0] * processor.image_seq_length
    if image_tokens != expected:
        raise ValueError(
            f"{context}: {image_tokens} image context tokens for {expected} visual features"
        )


def assert_internvl_rollout_prompt_alignment(vllm_ids, placeholder_ids, expanded_ids):
    if vllm_ids == placeholder_ids:
        return "placeholder"
    if vllm_ids == expanded_ids:
        return "expanded"
    actual = vllm_ids or []
    mismatch = next(
        (i for i, (a, b) in enumerate(zip(actual, expanded_ids)) if a != b),
        min(len(actual), len(expanded_ids)),
    )
    raise ValueError(
        "InternVL vLLM/HF prompt mismatch: "
        f"vllm_len={len(actual)}, placeholder_len={len(placeholder_ids)}, "
        f"hf_len={len(expanded_ids)}, first_hf_mismatch={mismatch}"
    )
