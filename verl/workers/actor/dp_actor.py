# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import json
import logging
import os
import time
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_self_distillation_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.metric import AggregationType, Metric, reduce_metrics
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrustRegionTeacher(nn.Module):
    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.teacher_module: Optional[nn.Module] = None
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _update_teacher(self) -> None:
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if not self_distillation_cfg or loss_mode != "vopd":
            return
        teacher_model_source = getattr(self_distillation_cfg, "teacher_model_source", "legacy")
        if teacher_model_source != "legacy":
            return
        teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema")
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("Teacher updates require a separate teacher_module in the actor worker.")
        with torch.no_grad():
            if teacher_regularization == "ema":
                update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
                if update_rate == 0.0:
                    return
                for teacher_param, student_param in zip(
                    self.teacher_module.parameters(),
                    self.actor_module.parameters(),
                ):
                    student_data = student_param.data.to(device=teacher_param.device)
                    teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)
                return

            if teacher_regularization == "progressive":
                teacher_update_interval = getattr(self_distillation_cfg, "teacher_update_interval", None)
                if teacher_update_interval is None:
                    raise ValueError("Progressive teacher requires self_distillation.teacher_update_interval.")
                global_steps = getattr(self, "_current_global_steps", None)
                if global_steps is None or global_steps % teacher_update_interval != 0:
                    return
                for teacher_param, student_param in zip(
                    self.teacher_module.parameters(),
                    self.actor_module.parameters(),
                ):
                    teacher_param.data.copy_(student_param.data.to(device=teacher_param.device))
                for teacher_buffer, student_buffer in zip(
                    self.teacher_module.buffers(),
                    self.actor_module.buffers(),
                ):
                    teacher_buffer.data.copy_(student_buffer.data.to(device=teacher_buffer.device))
                return

            return

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    @staticmethod
    def _add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
        log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        log_s = torch.clamp(log_s, max=-1e-7)
        tail_log = torch.log(-torch.expm1(log_s))
        return torch.cat([log_probs, tail_log], dim=-1)

    @staticmethod
    def _build_response_positions(
        response_start_idx: torch.Tensor,
        response_length: int,
        seqlen: int,
    ) -> torch.Tensor:
        if response_start_idx.dim() != 1:
            raise ValueError(f"response_start_idx must be rank-1, got shape {tuple(response_start_idx.shape)}")
        if (response_start_idx < 1).any():
            raise ValueError("response_start_idx must be >= 1 so response logits have a preceding context token.")

        offsets = torch.arange(response_length, device=response_start_idx.device, dtype=response_start_idx.dtype)
        response_positions = response_start_idx.unsqueeze(1) - 1 + offsets.unsqueeze(0)
        if response_positions.numel() > 0:
            if response_positions.min().item() < 0 or response_positions.max().item() >= seqlen:
                raise ValueError(
                    f"Response positions out of bounds for seqlen={seqlen}: "
                    f"min={response_positions.min().item()}, max={response_positions.max().item()}"
                )
        return response_positions.to(dtype=torch.long)

    @staticmethod
    def _select_response_positions(
        hidden_states: torch.Tensor,
        response_length: int,
        response_start_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if response_start_idx is None:
            return hidden_states[:, -response_length - 1 : -1, ...]

        response_positions = DataParallelPPOActor._build_response_positions(
            response_start_idx=response_start_idx,
            response_length=response_length,
            seqlen=hidden_states.size(1),
        )
        if hidden_states.dim() == 2:
            return torch.gather(hidden_states, dim=1, index=response_positions)

        gather_index = response_positions.view(
            response_positions.size(0),
            response_positions.size(1),
            *([1] * (hidden_states.dim() - 2)),
        ).expand(response_positions.size(0), response_positions.size(1), *hidden_states.shape[2:])
        return torch.gather(hidden_states, dim=1, index=gather_index)

    def _dump_self_distillation_log_probs(
        self,
        *,
        meta_info: dict,
        self_distillation_cfg,
        dump_chunks: list[dict[str, torch.Tensor]],
    ) -> None:
        dump_root = self_distillation_cfg.get("log_prob_dump_dir", None)
        if not dump_root or not dump_chunks:
            return

        global_step = meta_info.get("global_steps")
        if global_step is None:
            return

        experiment_name = os.environ.get("EXPERIMENT", "unknown_experiment")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        normalized_root = os.path.normpath(dump_root)
        if os.path.basename(normalized_root) == experiment_name:
            save_dir = normalized_root
        else:
            save_dir = os.path.join(normalized_root, experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{int(global_step)}.rank{rank}.pt")

        student_log_probs = torch.cat([chunk["student_log_probs"] for chunk in dump_chunks], dim=0)
        teacher_log_probs = torch.cat([chunk["teacher_log_probs"] for chunk in dump_chunks], dim=0)

        torch.save(
            {
                "student_log_probs": student_log_probs,
                "teacher_log_probs": teacher_log_probs,
                "global_step": int(global_step),
                "rank": rank,
                "experiment_name": experiment_name,
                "distribution_size": int(student_log_probs.shape[-1]),
                "num_valid_tokens": int(student_log_probs.shape[0]),
                "distillation_topk": self_distillation_cfg.get("distillation_topk", None),
                "distillation_add_tail": bool(self_distillation_cfg.get("distillation_add_tail", False)),
            },
            save_path,
        )

    @staticmethod
    def _masked_mean_item(values: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
        valid = mask > 0
        if not valid.any():
            return None
        return values.detach()[valid].to(torch.float32).mean().item()

    @staticmethod
    def _json_safe(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                return value.item()
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): DataParallelPPOActor._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [DataParallelPPOActor._json_safe(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if hasattr(value, "tolist"):
            try:
                return DataParallelPPOActor._json_safe(value.tolist())
            except Exception:
                pass
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _sample_value(values, idx: int):
        if values is None:
            return None
        try:
            return values[idx]
        except Exception:
            return None

    @staticmethod
    def _find_rank(ids: list[int], token_id: int) -> Optional[int]:
        try:
            return ids.index(token_id) + 1
        except ValueError:
            return None

    @staticmethod
    def _mean_or_none(values: list[float]) -> Optional[float]:
        values = [value for value in values if value is not None]
        if not values:
            return None
        return float(sum(values) / len(values))

    def _decode_token_id(self, token_id: int) -> str:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return ""
        try:
            return tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            return ""

    def _decode_token_ids(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return ""
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        except Exception:
            return ""

    def _add_opd_train_metrics(
        self,
        micro_batch_metrics: dict,
        *,
        response_mask: torch.Tensor,
        self_distillation_mask: Optional[torch.Tensor],
        responses: torch.Tensor,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        student_topk_indices: Optional[torch.Tensor],
        student_topk_logps: Optional[torch.Tensor],
        teacher_logps_on_student_topk: Optional[torch.Tensor],
        student_entropy: Optional[torch.Tensor],
    ) -> None:
        loss_mask = response_mask
        if self_distillation_mask is not None:
            loss_mask = loss_mask * self_distillation_mask.unsqueeze(1).to(response_mask.dtype)
        valid = loss_mask > 0
        if not valid.any():
            return

        def add(name: str, values: torch.Tensor) -> None:
            metric = self._masked_mean_item(values, loss_mask)
            if metric is not None:
                micro_batch_metrics[name] = metric

        add("opd/train/student_selected_logprob_mean", student_log_probs)
        add("opd/train/teacher_selected_logprob_mean", teacher_log_probs)
        add("opd/train/teacher_minus_student_selected_logprob_mean", teacher_log_probs - student_log_probs)
        micro_batch_metrics["opd/train/teacher_selected_logprob_lt_student_frac"] = (
            (teacher_log_probs.detach() < student_log_probs.detach()).to(torch.float32)[valid].mean().item()
        )

        if student_entropy is not None:
            add("opd/train/student_entropy_mean", student_entropy)

        if student_topk_logps is not None:
            student_topk_logps_f = student_topk_logps.detach().to(torch.float32)
            add("opd/train/student_topk_mass_mean", torch.exp(student_topk_logps_f).sum(dim=-1))
            add("opd/train/student_top1_logprob_mean", student_topk_logps_f[..., 0])
            if student_topk_logps_f.size(-1) > 1:
                add(
                    "opd/train/student_top1_top2_margin_mean",
                    student_topk_logps_f[..., 0] - student_topk_logps_f[..., 1],
                )

        if student_topk_indices is not None:
            selected_in_student_topk = (student_topk_indices.detach() == responses.unsqueeze(-1)).any(dim=-1)
            micro_batch_metrics["opd/train/selected_token_in_student_topk_frac"] = (
                selected_in_student_topk.to(torch.float32)[valid].mean().item()
            )
            student_top1_is_selected = student_topk_indices.detach()[..., 0] == responses
            micro_batch_metrics["opd/train/selected_token_is_student_top1_frac"] = (
                student_top1_is_selected.to(torch.float32)[valid].mean().item()
            )

        if teacher_logps_on_student_topk is not None:
            teacher_logps_on_student_topk_f = teacher_logps_on_student_topk.detach().to(torch.float32)
            add("opd/train/teacher_mass_on_student_topk_mean", torch.exp(teacher_logps_on_student_topk_f).sum(dim=-1))
            add("opd/train/teacher_logprob_on_student_top1_mean", teacher_logps_on_student_topk_f[..., 0])

    def _dump_opd_token_trace(
        self,
        *,
        meta_info: dict,
        self_distillation_cfg,
        model_inputs: dict,
        response_mask: torch.Tensor,
        self_distillation_mask: Optional[torch.Tensor],
        responses: torch.Tensor,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        student_topk_indices: Optional[torch.Tensor],
        student_topk_logps: Optional[torch.Tensor],
        teacher_topk_indices: Optional[torch.Tensor],
        teacher_topk_logps: Optional[torch.Tensor],
        student_entropy: Optional[torch.Tensor],
        teacher_entropy: Optional[torch.Tensor],
        max_samples: Optional[int],
    ) -> int:
        dump_root = self_distillation_cfg.get("trace_dump_dir", None) or self_distillation_cfg.get(
            "log_prob_dump_dir", None
        )
        if not dump_root:
            raise ValueError("self_distillation.trace_enabled=True requires trace_dump_dir or log_prob_dump_dir.")

        global_step = meta_info.get("global_steps")
        if global_step is None:
            return 0
        if (
            student_topk_indices is None
            or student_topk_logps is None
            or teacher_topk_indices is None
            or teacher_topk_logps is None
        ):
            raise ValueError("OPD token trace requires both student and teacher top-k ids/logprobs.")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        experiment_name = os.environ.get("EXPERIMENT", "unknown_experiment")
        normalized_root = os.path.normpath(dump_root)
        if os.path.basename(normalized_root) == experiment_name:
            save_dir = normalized_root
        else:
            save_dir = os.path.join(normalized_root, experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{int(global_step)}.rank{rank}.jsonl")

        response_mask_cpu = response_mask.detach().cpu()
        distill_mask_cpu = self_distillation_mask.detach().cpu() if self_distillation_mask is not None else None
        responses_cpu = responses.detach().cpu()
        student_log_probs_cpu = student_log_probs.detach().cpu().to(torch.float32)
        teacher_log_probs_cpu = teacher_log_probs.detach().cpu().to(torch.float32)
        student_topk_indices_cpu = student_topk_indices.detach().cpu() if student_topk_indices is not None else None
        student_topk_logps_cpu = (
            student_topk_logps.detach().cpu().to(torch.float32) if student_topk_logps is not None else None
        )
        teacher_topk_indices_cpu = teacher_topk_indices.detach().cpu() if teacher_topk_indices is not None else None
        teacher_topk_logps_cpu = (
            teacher_topk_logps.detach().cpu().to(torch.float32) if teacher_topk_logps is not None else None
        )
        student_entropy_cpu = student_entropy.detach().cpu().to(torch.float32) if student_entropy is not None else None
        teacher_entropy_cpu = teacher_entropy.detach().cpu().to(torch.float32) if teacher_entropy is not None else None

        records = []
        batch_size = responses_cpu.size(0)
        for sample_idx in range(batch_size):
            if max_samples is not None and len(records) >= max_samples:
                break
            if distill_mask_cpu is not None and float(distill_mask_cpu[sample_idx].item()) <= 0.5:
                continue
            valid_positions = torch.nonzero(response_mask_cpu[sample_idx] > 0, as_tuple=False).flatten().tolist()
            if not valid_positions:
                continue

            token_ids = [int(responses_cpu[sample_idx, pos].item()) for pos in valid_positions]
            extra_info = self._json_safe(self._sample_value(model_inputs.get("extra_info"), sample_idx))
            metadata = {
                "uid": self._json_safe(self._sample_value(model_inputs.get("uid"), sample_idx)),
                "index": self._json_safe(self._sample_value(model_inputs.get("index"), sample_idx)),
                "data_source": self._json_safe(self._sample_value(model_inputs.get("data_source"), sample_idx)),
                "extra_info": extra_info,
            }
            if isinstance(extra_info, dict):
                metadata["image_id"] = extra_info.get("image_id")
                metadata["file_name"] = extra_info.get("file_name")

            token_records = []
            overlap_ratios = []
            topk_jaccards = []
            top1_matches = []
            student_selected_logps = []
            teacher_selected_logps = []
            student_entropies = []
            teacher_entropies = []
            student_topk_masses = []
            teacher_topk_masses = []
            student_top1_top2_margins = []
            teacher_top1_top2_margins = []

            for step, pos in enumerate(valid_positions):
                token_id = int(responses_cpu[sample_idx, pos].item())
                token_record = {
                    "step": step,
                    "response_position": int(pos),
                    "token_id": token_id,
                    "token_text": self._decode_token_id(token_id),
                    "student_selected_logprob": float(student_log_probs_cpu[sample_idx, pos].item()),
                    "teacher_selected_logprob": float(teacher_log_probs_cpu[sample_idx, pos].item()),
                }
                student_selected_logps.append(token_record["student_selected_logprob"])
                teacher_selected_logps.append(token_record["teacher_selected_logprob"])

                if student_entropy_cpu is not None:
                    token_record["student_entropy"] = float(student_entropy_cpu[sample_idx, pos].item())
                    student_entropies.append(token_record["student_entropy"])
                if teacher_entropy_cpu is not None:
                    token_record["teacher_entropy"] = float(teacher_entropy_cpu[sample_idx, pos].item())
                    teacher_entropies.append(token_record["teacher_entropy"])

                if student_topk_indices_cpu is not None and student_topk_logps_cpu is not None:
                    student_ids = [int(value) for value in student_topk_indices_cpu[sample_idx, pos].tolist()]
                    student_logps = [float(value) for value in student_topk_logps_cpu[sample_idx, pos].tolist()]
                    token_record["student_topk_token_ids"] = student_ids
                    token_record["student_topk_logprobs"] = student_logps
                    token_record["student_top1_token_id"] = student_ids[0] if student_ids else None
                    student_topk_mass = float(torch.exp(student_topk_logps_cpu[sample_idx, pos]).sum().item())
                    student_top1_top2_margin = student_logps[0] - student_logps[1] if len(student_logps) > 1 else None
                    token_record["student_topk_mass"] = student_topk_mass
                    token_record["student_top1_top2_margin"] = student_top1_top2_margin
                    student_topk_masses.append(student_topk_mass)
                    student_top1_top2_margins.append(student_top1_top2_margin)
                    token_record["selected_token_rank_in_student_topk"] = self._find_rank(student_ids, token_id)
                else:
                    student_ids = []

                if teacher_topk_indices_cpu is not None and teacher_topk_logps_cpu is not None:
                    teacher_ids = [int(value) for value in teacher_topk_indices_cpu[sample_idx, pos].tolist()]
                    teacher_logps = [float(value) for value in teacher_topk_logps_cpu[sample_idx, pos].tolist()]
                    token_record["teacher_topk_token_ids"] = teacher_ids
                    token_record["teacher_topk_logprobs"] = teacher_logps
                    token_record["teacher_top1_token_id"] = teacher_ids[0] if teacher_ids else None
                    teacher_topk_mass = float(torch.exp(teacher_topk_logps_cpu[sample_idx, pos]).sum().item())
                    teacher_top1_top2_margin = teacher_logps[0] - teacher_logps[1] if len(teacher_logps) > 1 else None
                    token_record["teacher_topk_mass"] = teacher_topk_mass
                    token_record["teacher_top1_top2_margin"] = teacher_top1_top2_margin
                    teacher_topk_masses.append(teacher_topk_mass)
                    teacher_top1_top2_margins.append(teacher_top1_top2_margin)
                    token_record["selected_token_rank_in_teacher_topk"] = self._find_rank(teacher_ids, token_id)
                else:
                    teacher_ids = []

                if student_ids and teacher_ids:
                    student_set = set(student_ids)
                    teacher_set = set(teacher_ids)
                    overlap_count = len(student_set & teacher_set)
                    union_count = len(student_set | teacher_set)
                    overlap_ratio = overlap_count / min(len(student_set), len(teacher_set))
                    top1_match = student_ids[0] == teacher_ids[0]
                    token_record["topk_overlap_count"] = overlap_count
                    token_record["topk_overlap_ratio"] = overlap_ratio
                    token_record["topk_jaccard"] = overlap_count / union_count if union_count else None
                    token_record["top1_match"] = top1_match
                    overlap_ratios.append(overlap_ratio)
                    topk_jaccards.append(token_record["topk_jaccard"])
                    top1_matches.append(1.0 if top1_match else 0.0)

                token_records.append(token_record)

            records.append(
                {
                    "global_step": int(global_step),
                    "rank": int(rank),
                    "sample_index_in_rank_batch": int(sample_idx),
                    "metadata": metadata,
                    "response_token_ids": token_ids,
                    "response_text": self._decode_token_ids(token_ids, skip_special_tokens=True),
                    "summary": {
                        "num_tokens": len(token_records),
                        "student_selected_logprob_mean": self._mean_or_none(student_selected_logps),
                        "teacher_selected_logprob_mean": self._mean_or_none(teacher_selected_logps),
                        "student_entropy_mean": self._mean_or_none(student_entropies),
                        "teacher_entropy_mean": self._mean_or_none(teacher_entropies),
                        "student_topk_mass_mean": self._mean_or_none(student_topk_masses),
                        "teacher_topk_mass_mean": self._mean_or_none(teacher_topk_masses),
                        "student_top1_top2_margin_mean": self._mean_or_none(student_top1_top2_margins),
                        "teacher_top1_top2_margin_mean": self._mean_or_none(teacher_top1_top2_margins),
                        "topk_overlap_ratio_mean": self._mean_or_none(overlap_ratios),
                        "topk_jaccard_mean": self._mean_or_none(topk_jaccards),
                        "top1_match_frac": self._mean_or_none(top1_matches),
                    },
                    "token_records": token_records,
                }
            )

        if not records:
            return 0

        with open(save_path, "a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return len(records)

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        trace_topk: Optional[int] = None,
        module: Optional[nn.Module] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
                if distill_topk or topk_indices is set:
                    topk_logps: (bs, response_len, k)
                    topk_indices: (bs, response_len, k)
                if trace_topk is set:
                    trace_topk_logps: (bs, response_len, trace_topk)
                    trace_topk_indices: (bs, response_len, trace_topk)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_topk = distill_topk is not None or topk_indices is not None
        use_trace_topk = trace_topk is not None
        compute_all_logps = return_all_logps and not use_topk
        return_topk_indices = use_topk and topk_indices is None
        if (return_all_logps or use_topk or use_trace_topk) and self.use_fused_kernels:
            raise ValueError("Logit distillation/tracing requires disabling fused kernels.")

        model = module or self.actor_module

        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
                and not return_all_logps
                and not use_topk
                and not use_trace_topk
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=model,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            response_start_idx = micro_batch.get("response_start_idx")
            if response_start_idx is not None:
                response_start_idx = response_start_idx.to(device=input_ids.device, dtype=torch.long)
                if response_start_idx.shape != (batch_size,):
                    raise ValueError(
                        f"response_start_idx shape must be ({batch_size},), got {tuple(response_start_idx.shape)}"
                    )
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(model, "module", model).config,
                        "vision_config",
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    all_logps_rmpad = torch.log_softmax(logits_rmpad, dim=-1) if compute_all_logps else None

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            if response_start_idx is None:
                                full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            else:
                                response_positions = self._build_response_positions(
                                    response_start_idx=response_start_idx.to(device=topk_indices.device),
                                    response_length=response_length,
                                    seqlen=seqlen,
                                )
                                batch_indices = torch.arange(batch_size, device=topk_indices.device).unsqueeze(1)
                                full_topk_indices[batch_indices, response_positions, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad
                    if use_trace_topk:
                        trace_k = min(trace_topk, logits_rmpad.shape[-1])
                        if use_topk and return_topk_indices and trace_k == topk_logps_rmpad.size(-1):
                            trace_topk_indices_rmpad = topk_indices_rmpad
                            trace_topk_logps_rmpad = topk_logps_rmpad
                        else:
                            trace_topk_logits_rmpad, trace_topk_indices_rmpad = torch.topk(
                                logits_rmpad, trace_k, dim=-1
                            )
                            logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                            trace_topk_logps_rmpad = trace_topk_logits_rmpad - logsumexp_rmpad

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    if use_trace_topk:
                        trace_topk_logps_rmpad = gather_outputs_and_unpad(
                            trace_topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        trace_topk_indices_rmpad = gather_outputs_and_unpad(
                            trace_topk_indices_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if compute_all_logps:
                        all_logps_rmpad = all_logps_rmpad[:0]
                    if use_topk:
                        topk_logps_rmpad = topk_logps_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]
                    if use_trace_topk:
                        trace_topk_logps_rmpad = trace_topk_logps_rmpad[:0]
                        trace_topk_indices_rmpad = trace_topk_indices_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if compute_all_logps:
                    full_all_logps = pad_input(
                        hidden_states=all_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                if use_trace_topk:
                    full_trace_topk_logps = pad_input(
                        hidden_states=trace_topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_trace_topk_indices = pad_input(
                        hidden_states=trace_topk_indices_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = self._select_response_positions(
                        full_entropy.squeeze(-1),
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = self._select_response_positions(
                        full_sum_pi_squared.squeeze(-1),
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                log_probs = self._select_response_positions(
                    full_log_probs.squeeze(-1),
                    response_length=response_length,
                    response_start_idx=response_start_idx,
                )
                if compute_all_logps:
                    all_logps = self._select_response_positions(
                        full_all_logps,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                if use_topk:
                    topk_logps = self._select_response_positions(
                        full_topk_logps,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                    if return_topk_indices:
                        topk_indices = self._select_response_positions(
                            full_topk_indices,
                            response_length=response_length,
                            response_start_idx=response_start_idx,
                        )
                if use_trace_topk:
                    trace_topk_logps = self._select_response_positions(
                        full_trace_topk_logps,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                    trace_topk_indices = self._select_response_positions(
                        full_trace_topk_indices,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = self._select_response_positions(
                        logits,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits.size(-1))
                            topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
                        else:
                            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
                        topk_logps = topk_logits - logsumexp
                    if use_trace_topk:
                        trace_k = min(trace_topk, logits.size(-1))
                        if use_topk and return_topk_indices and trace_k == topk_logps.size(-1):
                            trace_topk_indices = topk_indices
                            trace_topk_logps = topk_logps
                        else:
                            trace_topk_logits, trace_topk_indices = torch.topk(logits, trace_k, dim=-1)
                            logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
                            trace_topk_logps = trace_topk_logits - logsumexp
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
            if use_trace_topk:
                outputs["trace_topk_logps"] = trace_topk_logps
                outputs["trace_topk_indices"] = trace_topk_indices
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
            return grad_norm

        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self._current_global_steps = data.meta_info.get("global_steps")
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        self_distillation_enabled = loss_mode == "vopd"
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        trace_enabled = bool(
            self_distillation_enabled
            and self_distillation_cfg is not None
            and self_distillation_cfg.get("trace_enabled", False)
        )
        if self_distillation_enabled:
            if self_distillation_cfg is None:
                raise ValueError(f"loss_mode={loss_mode} requires actor.self_distillation config.")
            self_distillation_required_keys = {
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "teacher_response_start_idx",
                "self_distillation_mask",
            }
            assert self_distillation_required_keys.issubset(set(data.batch.keys())), f"Missing required keys: {self_distillation_required_keys - set(data.batch.keys())}"

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
        ]
        if not self_distillation_enabled or "advantages" in data.batch.keys():
            select_keys.append("advantages")
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self_distillation_enabled:
            select_keys.extend(list(self_distillation_required_keys))
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        has_teacher_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("teacher_multi_modal_inputs")
        )
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if has_teacher_multi_modal_inputs:
            non_tensor_select_keys.append("teacher_multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")
        if trace_enabled:
            for key in ("uid", "index", "data_source", "extra_info"):
                if key in data.non_tensor_batch.keys() and key not in non_tensor_select_keys:
                    non_tensor_select_keys.append(key)

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        if self_distillation_enabled:
            metrics["actor/grpo_loss"] = 0.0
            metrics["actor/vopd_loss"] = 0.0
            metrics["actor/vopd_loss_weighted"] = 0.0
        distill_dump_chunks = []
        trace_samples_written = 0
        trace_due_for_step = False
        trace_topk = None
        trace_max_samples = None
        trace_entropy = False
        if trace_enabled:
            global_step = data.meta_info.get("global_steps")
            trace_every_n_steps = int(self_distillation_cfg.get("trace_every_n_steps", 5))
            trace_due_for_step = global_step is not None and int(global_step) % trace_every_n_steps == 0
            trace_topk = int(
                self_distillation_cfg.get("trace_topk", None)
                or self_distillation_cfg.get("distillation_topk", None)
                or 100
            )
            configured_max_samples = int(self_distillation_cfg.get("trace_max_samples", 4))
            trace_max_samples = None if configured_max_samples <= 0 else configured_max_samples
            trace_entropy = bool(self_distillation_cfg.get("trace_entropy", True))
        stage_wall_time_totals = None
        if self_distillation_enabled:
            stage_wall_time_totals = {
                "timing_s/update_actor/student_forward": 0.0,
                "timing_s/update_actor/teacher_forward": 0.0,
                "timing_s/update_actor/loss_compute": 0.0,
                "timing_s/update_actor/backward": 0.0,
                "timing_s/update_actor/optimizer_step": 0.0,
                "timing_s/update_actor/teacher_ema_update": 0.0,
            }
        did_update = False
        for ppo_epoch_idx in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs.get("advantages")

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    policy_calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    self_distillation_mask = model_inputs.get("self_distillation_mask") if self_distillation_enabled else None
                    trace_this_micro_batch = (
                        trace_due_for_step
                        and (not self_distillation_cfg.get("trace_only_first_ppo_epoch", True) or ppo_epoch_idx == 0)
                        and (trace_max_samples is None or trace_samples_written < trace_max_samples)
                    )
                    calculate_entropy = policy_calculate_entropy or (trace_this_micro_batch and trace_entropy)
                    policy_fallback_mask = None
                    if self_distillation_enabled and self_distillation_mask is not None:
                        policy_fallback_mask = (self_distillation_mask <= 0.5).to(response_mask.dtype)
                        micro_batch_metrics["actor/policy_fallback_fraction"] = (
                            policy_fallback_mask.float().mean().detach().item()
                        )

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
                    teacher_model_source = self_distillation_cfg.get("teacher_model_source", "legacy")
                    use_trust_region_teacher = teacher_model_source == "legacy" and teacher_regularization == "trust-region"
                    if use_trust_region_teacher and self.use_fused_kernels:
                        raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
                    # all return: (bsz, response_length)
                    return_all_logps = self_distillation_cfg.full_logit_distillation and not self_distillation_cfg.distillation_topk
                    distill_topk = self_distillation_cfg.distillation_topk if self_distillation_cfg.full_logit_distillation else None
                    student_forward_start = time.perf_counter()
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=return_all_logps,
                        distill_topk=distill_topk,
                        trace_topk=trace_topk if trace_this_micro_batch else None,
                    )
                    if self_distillation_enabled:
                        student_forward_time = time.perf_counter() - student_forward_start
                        stage_wall_time_totals["timing_s/update_actor/student_forward"] += student_forward_time
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_logps = outputs.get("all_logps") if return_all_logps else None
                    student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                    student_topk_indices = outputs.get("topk_indices") if distill_topk else None
                    student_trace_topk_logps = outputs.get("trace_topk_logps") if trace_this_micro_batch else None
                    student_trace_topk_indices = outputs.get("trace_topk_indices") if trace_this_micro_batch else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    if self_distillation_enabled:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                            "response_start_idx": model_inputs["teacher_response_start_idx"],
                        }
                        if "teacher_multi_modal_inputs" in model_inputs:
                            teacher_inputs["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]
                        teacher_model = self.teacher_module or self.actor_module
                        if use_trust_region_teacher and (
                            self.teacher_module is None or self.teacher_module is self.actor_module
                        ):
                            raise ValueError("trust-region teacher requires a separate teacher_module in the actor worker.")
                        with torch.no_grad():
                            teacher_forward_start = time.perf_counter()
                            teacher_outputs = self._forward_micro_batch(
                                teacher_inputs,
                                temperature=temperature,
                                calculate_entropy=trace_this_micro_batch and trace_entropy,
                                return_all_logps=return_all_logps,
                                distill_topk=distill_topk,
                                topk_indices=student_topk_indices,
                                trace_topk=trace_topk if trace_this_micro_batch else None,
                                module=teacher_model,
                            )
                            teacher_forward_time = time.perf_counter() - teacher_forward_start
                        stage_wall_time_totals["timing_s/update_actor/teacher_forward"] += teacher_forward_time
                        teacher_log_prob = teacher_outputs["log_probs"]
                        teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                        teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None
                        teacher_entropy = teacher_outputs.get("entropys") if trace_this_micro_batch and trace_entropy else None
                        teacher_trace_topk_logps = (
                            teacher_outputs.get("trace_topk_logps") if trace_this_micro_batch else None
                        )
                        teacher_trace_topk_indices = (
                            teacher_outputs.get("trace_topk_indices") if trace_this_micro_batch else None
                        )
                        if self_distillation_cfg.get("train_metrics_enabled", True):
                            self._add_opd_train_metrics(
                                micro_batch_metrics,
                                response_mask=response_mask,
                                self_distillation_mask=self_distillation_mask,
                                responses=model_inputs["responses"],
                                student_log_probs=log_prob,
                                teacher_log_probs=teacher_log_prob,
                                student_topk_indices=student_topk_indices,
                                student_topk_logps=student_topk_logps,
                                teacher_logps_on_student_topk=teacher_topk_logps,
                                student_entropy=entropy if policy_calculate_entropy else None,
                            )
                        if trace_this_micro_batch:
                            remaining_samples = (
                                None if trace_max_samples is None else trace_max_samples - trace_samples_written
                            )
                            trace_student_topk_indices = (
                                student_trace_topk_indices
                                if student_trace_topk_indices is not None
                                else student_topk_indices
                            )
                            trace_student_topk_logps = (
                                student_trace_topk_logps if student_trace_topk_logps is not None else student_topk_logps
                            )
                            written = self._dump_opd_token_trace(
                                meta_info=data.meta_info,
                                self_distillation_cfg=self_distillation_cfg,
                                model_inputs=model_inputs,
                                response_mask=response_mask,
                                self_distillation_mask=self_distillation_mask,
                                responses=model_inputs["responses"],
                                student_log_probs=log_prob,
                                teacher_log_probs=teacher_log_prob,
                                student_topk_indices=trace_student_topk_indices,
                                student_topk_logps=trace_student_topk_logps,
                                teacher_topk_indices=teacher_trace_topk_indices,
                                teacher_topk_logps=teacher_trace_topk_logps,
                                student_entropy=entropy if trace_entropy else None,
                                teacher_entropy=teacher_entropy,
                                max_samples=remaining_samples,
                            )
                            trace_samples_written += written
                        if self_distillation_cfg.get("log_prob_dump_dir", None):
                            if distill_topk:
                                student_distill_log_probs = student_topk_logps
                                teacher_distill_log_probs = teacher_topk_logps
                                if self_distillation_cfg.distillation_add_tail:
                                    student_distill_log_probs = self._add_tail_bucket(student_distill_log_probs)
                                    teacher_distill_log_probs = self._add_tail_bucket(teacher_distill_log_probs)
                            else:
                                student_distill_log_probs = student_all_logps
                                teacher_distill_log_probs = teacher_all_logps

                            if student_distill_log_probs is None or teacher_distill_log_probs is None:
                                raise ValueError("Missing distillation log_probs for dump.")

                            loss_mask = response_mask
                            if self_distillation_mask is not None:
                                loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)
                            valid_rows = loss_mask > 0
                            if valid_rows.any():
                                distill_dump_chunks.append(
                                    {
                                        "student_log_probs": student_distill_log_probs[valid_rows].detach().cpu().to(torch.float32),
                                        "teacher_log_probs": teacher_distill_log_probs[valid_rows].detach().cpu().to(torch.float32),
                                    }
                                )
                        loss_compute_start = time.perf_counter()
                        vopd_loss, vopd_metrics = compute_self_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_prob,
                            response_mask=response_mask,
                            self_distillation_config=self_distillation_cfg,
                            old_log_probs=old_log_prob,
                            student_all_log_probs=student_all_logps,
                            teacher_all_log_probs=teacher_all_logps,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_topk_logps,
                            self_distillation_mask=self_distillation_mask,
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                            batch_num_tokens=self.config.global_batch_info.get("batch_num_tokens"),
                            global_batch_size=self.config.global_batch_info.get("global_batch_size"),
                            loss_scale_factor=self.config.global_batch_info.get("loss_scale_factor"),
                        )
                        loss_compute_time = time.perf_counter() - loss_compute_start
                        stage_wall_time_totals["timing_s/update_actor/loss_compute"] += loss_compute_time

                        vopd_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
                        micro_batch_metrics.update(vopd_metrics)

                        if policy_fallback_mask is not None and policy_fallback_mask.any().item():
                            if advantages is None:
                                raise ValueError(
                                    "Mixed SDPO/GRPO fallback requires advantages for samples without teacher images."
                                )
                            policy_loss_fn = get_policy_loss_fn("vanilla")
                            grpo_loss, grpo_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask * policy_fallback_mask.unsqueeze(1),
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )
                            pg_loss = vopd_loss + grpo_loss
                            micro_batch_metrics.update(
                                {f"actor/policy_fallback/{key.split('/', 1)[1]}": value for key, value in grpo_metrics.items()}
                            )
                        else:
                            grpo_loss = None
                            pg_loss = vopd_loss
                    else:
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # Compute policy loss (any function is expected to return 2 values)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if policy_calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    backward_start = time.perf_counter()
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if self_distillation_enabled:
                        backward_time = time.perf_counter() - backward_start
                        stage_wall_time_totals["timing_s/update_actor/backward"] += backward_time

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    if self_distillation_enabled:
                        metrics["actor/vopd_loss"] += vopd_loss.detach().item() * loss_scale_factor
                        metrics["actor/vopd_loss_weighted"] += vopd_loss.detach().item() * loss_scale_factor
                        if grpo_loss is not None:
                            metrics["actor/grpo_loss"] += grpo_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                optimizer_step_start = time.perf_counter()
                grad_norm = self._optimizer_step()
                if self_distillation_enabled:
                    optimizer_step_time = time.perf_counter() - optimizer_step_start
                if torch.isfinite(grad_norm).item():
                    did_update = True
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                if self_distillation_enabled:
                    stage_wall_time_totals["timing_s/update_actor/optimizer_step"] += optimizer_step_time
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if self_distillation_enabled and distill_dump_chunks:
            self._dump_self_distillation_log_probs(
                meta_info=data.meta_info,
                self_distillation_cfg=self_distillation_cfg,
                dump_chunks=distill_dump_chunks,
            )
        if did_update:
            teacher_update_start = time.perf_counter()
            self._update_teacher()
            if self_distillation_enabled:
                stage_wall_time_totals["timing_s/update_actor/teacher_ema_update"] += (
                    time.perf_counter() - teacher_update_start
                )
        if self_distillation_enabled:
            for key, total_time in stage_wall_time_totals.items():
                metrics[key] = Metric(aggregation=AggregationType.MAX, value=total_time)
        metric_keys_to_keep_unreduced = set(stage_wall_time_totals.keys()) if stage_wall_time_totals is not None else set()
        local_metrics_to_reduce = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, list) or (isinstance(value, Metric) and key not in metric_keys_to_keep_unreduced)
        }
        if local_metrics_to_reduce:
            reduced_local_metrics = reduce_metrics(local_metrics_to_reduce)
            metrics.update(reduced_local_metrics)
        return metrics
