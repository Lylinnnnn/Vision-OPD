#!/usr/bin/env python3
"""Phrase-alignment LoRA trainer for Qwen3VL and HF InternVL3."""

import argparse
import json
import math
import re
from pathlib import Path


LANGUAGE_LINEAR_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def phrase_token_positions(tokenizer, input_ids, phrase, response_start):
    ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    response = tokenizer.decode(ids[response_start:], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def whole_word(start, end):
        before = tokenizer.decode(ids[response_start:start], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        through = tokenizer.decode(ids[response_start:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        segment = through[len(before):]
        offset = segment.casefold().find(phrase.casefold())
        if offset < 0:
            return False
        left = len(before) + offset
        right = left + len(phrase)
        return (response[left:right].casefold() == phrase.casefold()
                and (left == 0 or not re.match(r"\w", response[left - 1]))
                and (right == len(response) or not re.match(r"\w", response[right])))

    for prefix in (" ", ""):
        candidate = tokenizer.encode(prefix + phrase, add_special_tokens=False)
        matches = [
            tuple(range(start, start + len(candidate)))
            for start in range(response_start, len(ids) - len(candidate) + 1)
            if ids[start:start + len(candidate)] == candidate
            and whole_word(start, start + len(candidate))
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous tokenized phrase occurrence for {phrase!r}")
    raise ValueError(f"Expected one tokenized phrase occurrence, got 0 for {phrase!r}")


def encode_pair(processor, image, prompt, caption, phrase, max_length):
    user = {"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt},
    ]}
    prefix = processor.apply_chat_template(
        [user], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    full = processor.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": caption}]}],
        add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt",
    )
    input_ids = full["input_ids"][0]
    prompt_ids = prefix["input_ids"][0]
    if input_ids.numel() > max_length:
        raise ValueError(f"Input length {input_ids.numel()} exceeds --max-length={max_length}")
    response_start = 0
    for left, right in zip(input_ids.tolist(), prompt_ids.tolist()):
        if left != right:
            break
        response_start += 1
    if response_start < prompt_ids.numel() - 2:
        raise ValueError("Full conversation does not preserve the image prompt prefix")
    positions = phrase_token_positions(processor.tokenizer, input_ids, phrase, response_start)
    if not any(key in full for key in ("pixel_values", "pixel_values_videos")):
        raise ValueError("Processor did not produce image tensors")
    return full, positions, response_start


def phrase_logprob(logits, input_ids, positions):
    import torch

    position_tensor = torch.tensor(positions, device=logits.device)
    selected = logits[0, position_tensor - 1].float()
    target = input_ids[0, position_tensor].unsqueeze(-1)
    return (selected.gather(-1, target).squeeze(-1) - selected.logsumexp(-1)).sum()


def reference_kl(policy_logits, reference_logits, response_start):
    import torch.nn.functional as functional

    policy = policy_logits[0, response_start - 1:-1].float()
    reference = reference_logits[0, response_start - 1:-1].float()
    reference_logp = functional.log_softmax(reference, dim=-1)
    return (reference_logp.exp() * (reference_logp - functional.log_softmax(policy, dim=-1))).sum()


def model_class(family):
    if family == "internvl2b":
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    from transformers import Qwen3VLForConditionalGeneration
    return Qwen3VLForConditionalGeneration


def load_model(path, family):
    import torch

    model = model_class(family).from_pretrained(
        str(path), torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    return model.to("cuda:0")


def lora_targets(model):
    import torch

    names = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.endswith(LANGUAGE_LINEAR_SUFFIXES)
        and ("language_model" in name or "text_model" in name)
    ]
    if not names:
        raise ValueError("No language-model linear layers found for LoRA")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen2b", "qwen4b", "qwen8b", "internvl2b"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--loss-alpha", type=float, default=0.4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-all", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line]
    if not rows or args.grad_accum < 1 or args.epochs < 1 or args.max_steps < 0:
        raise ValueError("Invalid training data or steps/epochs/accumulation")
    config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    expected_type = "internvl" if args.model == "internvl2b" else "qwen3_vl"
    if config.get("model_type") != expected_type:
        raise ValueError(f"Expected {expected_type} checkpoint, got {config.get('model_type')!r}")
    if (args.output_dir / "adapter_config.json").exists():
        raise FileExistsError(f"Adapter already exists: {args.output_dir}")
    for row in rows:
        if row["chosen"] == row["rejected"] or not all(
            row[key] for key in ("image_path", "prompt", "chosen_phrase", "rejected_phrase")
        ):
            raise ValueError(f"Invalid phrase pair for image {row.get('image_id')}")
    if args.preflight_only:
        from PIL import Image
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True, local_files_only=True)
        checked = rows if args.preflight_all else rows[:1]
        errors = []
        for index, row in enumerate(checked, 1):
            try:
                with Image.open(row["image_path"]) as picture:
                    image = picture.convert("RGB")
                for answer, phrase in (("chosen", "chosen_phrase"), ("rejected", "rejected_phrase")):
                    encode_pair(processor, image, row["prompt"], row[answer], row[phrase], args.max_length)
            except Exception as error:
                errors.append(f"image_id={row['image_id']}: {type(error).__name__}: {error}")
            if args.preflight_all and index % 100 == 0:
                print(f"HALVA_TRAIN_PREFLIGHT_PROGRESS={index}/{len(checked)} errors={len(errors)}", flush=True)
        if errors:
            raise RuntimeError(
                f"HALVA_TRAIN_PREFLIGHT=FAIL checked={len(checked)} errors={len(errors)}\n"
                + "\n".join(errors[:20])
            )
        print(f"HALVA_TRAIN_PREFLIGHT=PASS checked={len(checked)} pairs={len(rows)} model={args.model}")
        return

    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU")
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True, local_files_only=True)
    model = load_model(args.model_path, args.model)
    config = LoraConfig(r=128, lora_alpha=256, lora_dropout=0.0, target_modules=lora_targets(model))
    model = get_peft_model(model, config)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    total_steps = args.max_steps or math.ceil(len(rows) * args.epochs / args.grad_accum)
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for index, row in enumerate(rows):
            with Image.open(row["image_path"]) as picture:
                image = picture.convert("RGB")
                positive, pos_positions, response_start = encode_pair(
                    processor, image, row["prompt"], row["chosen"], row["chosen_phrase"], args.max_length
                )
                negative, neg_positions, _ = encode_pair(
                    processor, image, row["prompt"], row["rejected"], row["rejected_phrase"], args.max_length
                )
            positive = {key: value.to("cuda:0") if hasattr(value, "to") else value for key, value in positive.items()}
            negative = {key: value.to("cuda:0") if hasattr(value, "to") else value for key, value in negative.items()}
            pos_logits = model(**positive, use_cache=False).logits
            pos_logp = phrase_logprob(pos_logits, positive["input_ids"], pos_positions)
            with torch.no_grad(), model.disable_adapter():
                ref_logits = model(**positive, use_cache=False).logits
            divergence = reference_kl(pos_logits, ref_logits, response_start)
            del ref_logits, pos_logits
            neg_logits = model(**negative, use_cache=False).logits
            neg_logp = phrase_logprob(neg_logits, negative["input_ids"], neg_positions)
            del neg_logits
            alignment = functional.softplus(neg_logp - pos_logp)
            loss = alignment + args.loss_alpha * divergence
            (loss / args.grad_accum).backward()
            if (index + 1) % args.grad_accum == 0 or index + 1 == len(rows):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                print(f"step={step}/{total_steps} loss={loss.item():.5f} align={alignment.item():.5f} kl={divergence.item():.5f}", flush=True)
                if step >= total_steps:
                    break
        if step >= total_steps:
            break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    (args.output_dir / "halva_run.json").write_text(json.dumps({
        "base_model": str(args.model_path), "pairs": str(args.pairs),
        "pair_count": len(rows), "epochs": args.epochs, "steps": step,
        "loss_alpha": args.loss_alpha, "method": "HALVA phrase alignment + reference KL",
    }, indent=2), encoding="utf-8")
    print(f"HALVA_TRAIN=PASS steps={step} adapter={args.output_dir}")


if __name__ == "__main__":
    main()
