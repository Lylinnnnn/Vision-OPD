import os

import torch
import torch.distributed as dist
from peft.peft_model import PeftModelForCausalLM
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from llava.train.llava_trainer import maybe_zero_3


# Borrowed from peft.utils.get_peft_model_state_dict
def _get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def _get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def _is_rank0(args: TrainingArguments):
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return getattr(args, "local_rank", -1) in (-1, 0)


class LlavaCallback(TrainerCallback):
    "A callback that save the model at the end of training."

    def __init__(self, save_steps=50):
        self.save_steps = save_steps

    def save_model(self, model: PeftModelForCausalLM, args: TrainingArguments):
        if not _is_rank0(args):
            return

        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[LlavaCallback] Saving LoRA adapter to {args.output_dir}", flush=True)
        torch.cuda.synchronize()
        safe_serialization = os.environ.get("LLAVA_SAFE_SERIALIZATION", "1") == "1"
        state_dict = _get_peft_state_maybe_zero_3(model.named_parameters(), "none")
        non_lora_state_dict = _get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        model.config.save_pretrained(args.output_dir)
        model.save_pretrained(args.output_dir, state_dict=state_dict, safe_serialization=safe_serialization)
        if non_lora_state_dict:
            torch.save(non_lora_state_dict, os.path.join(args.output_dir, "non_lora_trainables.bin"))
        print("[LlavaCallback] Finished saving LoRA adapter", flush=True)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        # save model
        model = kwargs["model"]
        if isinstance(model, PeftModelForCausalLM):
            self.save_model(model, args)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        now_step = state.global_step
        if now_step != 0 and now_step > 50 and now_step % self.save_steps == 0 and now_step < 150:
            origin_dir: str = args.output_dir
            args.output_dir = os.path.join(args.output_dir, f"steps_{now_step}")
            self.save_model(kwargs["model"], args)
            args.output_dir = origin_dir
