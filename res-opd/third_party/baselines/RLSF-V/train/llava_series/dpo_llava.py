import os
import sys

import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import HfArgumentParser, LlamaTokenizer, PreTrainedTokenizer
from trl import DPOConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from llava import conversation as conversation_lib
from llava.model import LlavaLlamaForCausalLM, LlavaMptForCausalLM
from llava.train.train import smart_tokenizer_and_embedding_resize

from arguments import LlavaDataArguments, LlavaModelArguments, LlavaTrainingArguments
from callback import LlavaCallback
from data import DataCollatorForSupervisedDataset, LazySupervisedDataset
from llava_trainer import LlavaDPOTrainer
from utils import *
local_rank = None


def rank0_print(*args) -> None:
    if local_rank == 0:
        print(*args)


def find_all_linear_names(model: LlavaLlamaForCausalLM) -> list[str]:
    """
    Train only the linear layers in the model
    The result of this func should be:
    [
        "o_proj",
        "gate_proj",
        "down_proj",
        "v_proj",
        "q_proj",
        "up_proj",
        "k_proj",
    ]
    """
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def make_supervised_data_module(tokenizer: PreTrainedTokenizer, data_args: LlavaDataArguments) -> dict:
    train_dataset = LazySupervisedDataset(
        train_data_path=data_args.train_data_path,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def setup_llava_model(
    model_args: LlavaModelArguments,
    data_args: LlavaDataArguments,
    training_args: LlavaTrainingArguments,
) -> tuple[PeftModel, PreTrainedTokenizer]:
    # local rank and device
    if "LOCAL_RANK" not in os.environ:
        local_rank = None
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:  # low bit quantization
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": device},
                # load_in_4bit=training_args.bits == 4,
                # load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    if model_args.vision_tower is not None:
        if "mpt" in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config["attn_impl"] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                low_cpu_mem_usage=True,
                device_map=device,
                **bnb_model_from_pretrained_args,
            )
        else:  # Load Llava here
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                low_cpu_mem_usage=True,
                # device_map=device,
                **bnb_model_from_pretrained_args,
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            low_cpu_mem_usage=True,
            device_map=device,
            **bnb_model_from_pretrained_args,
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        lora_config = LoraConfig(
            r=training_args.lora_r,  # Rank of the low-rank matrices
            lora_alpha=training_args.lora_alpha,  # Scaling factor
            target_modules=find_all_linear_names(model),  # Targeted modules
            lora_dropout=training_args.lora_dropout,  # Dropout rate
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            elif training_args.fp16:
                model.to(torch.float16)

        rank0_print("Adding LoRA adapters...")
        model: PeftModel = get_peft_model(model, lora_config)

    if "mpt" in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer: LlamaTokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.conv_version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.conv_version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        # for llava 1.5
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.conv_version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.conv_version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    return model, tokenizer


def main():
    parser = HfArgumentParser((LlavaModelArguments, LlavaDataArguments, LlavaTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # check_data_exist(data_args)

    # setup llava model
    llava_policy_model, tokenizer = setup_llava_model(model_args, data_args, training_args)

    data_module: dict[str] = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # if not use gradient_checkpointing, do not set ddp_find_unused_parameters
    if not training_args.gradient_checkpointing:
        training_args.ddp_find_unused_parameters = False

    # initialize training arguments
    dpo_config = DPOConfig(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        max_steps=training_args.max_steps,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        gradient_checkpointing=training_args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if training_args.gradient_checkpointing else None,
        ddp_find_unused_parameters=training_args.ddp_find_unused_parameters,
        learning_rate=training_args.learning_rate,
        eval_strategy=training_args.eval_strategy,
        eval_steps=training_args.eval_steps,
        output_dir=training_args.output_dir,
        report_to=training_args.report_to,
        lr_scheduler_type=training_args.lr_scheduler_type,
        warmup_steps=training_args.warmup_steps,
        optim=training_args.optimizer_type,
        bf16=training_args.bf16,
        remove_unused_columns=False,
        run_name=training_args.run_name,
        max_grad_norm=training_args.max_grad_norm,
        deepspeed=training_args.deepspeed,
        num_train_epochs=training_args.num_train_epochs,
        save_strategy=training_args.save_strategy,
        save_total_limit=training_args.save_total_limit,
        warmup_ratio=training_args.warmup_ratio,
        tf32=training_args.tf32,
        dataloader_num_workers=training_args.dataloader_num_workers,
        fp16=training_args.fp16,
        seed=training_args.seed,
    )

    # initialize the DPO trainer
    dpo_trainer = LlavaDPOTrainer(
        model=llava_policy_model,
        args=dpo_config,
        dpo_beta=training_args.beta,
        processing_class=tokenizer,
        max_prompt_length=training_args.max_prompt_length,
        max_length=training_args.max_length,
        **data_module,
    )

    dpo_trainer.add_callback(LlavaCallback())

    dpo_trainer.train()
    dpo_trainer.save_state()


if __name__ == "__main__":
    main()
