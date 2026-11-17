# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, Literal, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.constants import IGNORE_INDEX
from base_trainer import BaseDPOTrainer
import torch.nn.functional as F

class LlavaDPOTrainer(BaseDPOTrainer):
    """The DPO Trainer for Llava model"""

    def dpo_concatenated_forward(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        images: torch.Tensor = inputs["images"]  # Shape: (batch_size, 3, image_size, image_size)
        chosen_input_ids: torch.Tensor = inputs["chosen_input_ids"]  # Shape: (batch_size, length)
        chosen_labels: torch.Tensor = inputs["chosen_labels"]
        chosen_attention_mask: torch.Tensor = inputs["chosen_attention_mask"]
        reject_input_ids: torch.Tensor = inputs["reject_input_ids"]
        reject_labels: torch.Tensor = inputs["reject_labels"]
        reject_attention_mask: torch.Tensor = inputs["reject_attention_mask"]

        batch_size: int = chosen_input_ids.shape[0]
        dtype, device = chosen_input_ids.dtype, chosen_input_ids.device
        max_dim = max(chosen_input_ids.shape[1], reject_input_ids.shape[1])  # max length of chosen and rejected

        # Shape: (batch_size * 2, max_length)
        batch_input_ids = torch.zeros((batch_size * 2, max_dim), dtype=dtype, device=device)
        batch_labels = torch.ones((batch_size * 2, max_dim), dtype=dtype, device=device) * IGNORE_INDEX
        batch_attention_mask = torch.zeros((batch_size * 2, max_dim), device=device).to(torch.bool)

        # Concatenate chosen and rejected inputs
        batch_input_ids[:batch_size, : chosen_input_ids.shape[1]] = chosen_input_ids
        batch_input_ids[batch_size:, : reject_input_ids.shape[1]] = reject_input_ids
        batch_labels[:batch_size, : chosen_labels.shape[1]] = chosen_labels
        batch_labels[batch_size:, : reject_labels.shape[1]] = reject_labels
        batch_attention_mask[:batch_size, : chosen_attention_mask.shape[1]] = chosen_attention_mask
        batch_attention_mask[batch_size:, : reject_attention_mask.shape[1]] = reject_attention_mask

        # prepare inputs
        (
            batch_input_ids,  # None
            batch_position_ids,
            batch_attention_mask,
            batch_past_key_values,
            batch_inputs_embeds,  # Shape: (batch_size * 2, length after adding 576 image token, hidden_size)
            batch_labels,
        ) = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=batch_input_ids,
            position_ids=None,
            attention_mask=batch_attention_mask,
            past_key_values=None,
            labels=batch_labels,
            images=torch.cat([images, images], dim=0),
        )

        # calculate logits using forward pass
        all_logits = model.forward(
            inputs_embeds=batch_inputs_embeds,
            labels=None,
            attention_mask=batch_attention_mask,
        ).logits.to(torch.float32)  # Shape: (batch_size * 2, length after adding 576 image token, vocab_size)

        all_logps = self._get_batch_logps(
            all_logits,
            batch_labels,
        )  # Log probabilities of the labels under the given logits. Shape: (batch_size * 2,)

        chosen_logps = all_logps[:batch_size]  # Shape: (batch_size,)
        rejected_logps = all_logps[batch_size:]  # Shape: (batch_size,)

        # don't count image embeds logits
        loss_mask = batch_labels != IGNORE_INDEX  # Shape: (batch_size * 2, length after adding 576 image token)
        logits = [all_logits[i][loss_mask[i]] for i in range(loss_mask.shape[0])]
        chosen_logits = logits[:batch_size]  # Len: batch_size, Shape: (Real length of chosen, vocab_size)
        rejected_logits = logits[batch_size:]
        chosen_logits = [logit.detach().cpu().mean() for logit in chosen_logits]
        rejected_logits = [logit.detach().cpu().mean() for logit in rejected_logits]
        chosen_logits = sum(chosen_logits) / batch_size
        rejected_logits = sum(rejected_logits) / batch_size

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def sft_forward(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        images: torch.Tensor = inputs["sft_images"]  # Shape: (batch_size, 3, image_size, image_size)
        input_ids: torch.Tensor = inputs["sft_input_ids"]  # Shape: (batch_size, length)
        labels: torch.Tensor = inputs["sft_labels"]
        attention_mask: torch.Tensor = inputs["sft_attention_mask"]

        # prepare inputs
        (
            input_ids,  # None
            position_ids,  # None
            attention_mask,
            past_key_values,  # None
            inputs_embeds,  # Shape: (batch_size, length after adding 576 image token, hidden_size)
            labels,
        ) = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=labels,
            images=images,
        )

        # calculate logits using forward pass, Shape: (batch_size, length after adding 576 image token, vocab_size)

        out: CausalLMOutputWithPast = model.forward(
            inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask
        )
        sft_loss = out.loss  # Shape: ()

        # # Log probabilities of the labels under the given logits. Shape: (batch_size,)
        # batch_size: int = input_ids.shape[0]
        # sft_logits = out.logits.to(torch.float32).detach()
        # sft_logps = self._get_batch_logps(sft_logits, labels)

        # # don't count image embeds logits
        # loss_mask = labels != IGNORE_INDEX  # Shape: (batch_size, length after adding 576 image token)
        # logits = [sft_logits[i][loss_mask[i]] for i in range(loss_mask.shape[0])]
        # logits = [logit.detach().cpu().mean() for logit in logits]
        # logits = sum(logits) / batch_size

        return sft_loss  # , sft_logits, sft_logps

    def compute_ref_log_probs(self, inputs: dict[str, torch.LongTensor]):
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    chosen_logps, rejected_logps, _, _ = self.dpo_concatenated_forward(self.model, inputs)
            else:
                chosen_logps, rejected_logps, _, _ = self.dpo_concatenated_forward(self.ref_model, inputs)
        return chosen_logps, rejected_logps

    def get_batch_loss_metrics(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """
        Calculate the loss and metrics for a batch of inputs.
        Args:
            inputs: The batch of inputs.
        """
        (
            policy_chosen_logps,  # Shape: (batch_size,)
            policy_rejected_logps,  # Shape: (batch_size,)
            policy_chosen_logits,  # Shape: ()
            policy_rejected_logits,  # Shape: ()
        ) = self.dpo_concatenated_forward(model, inputs)
        reference_chosen_logps, reference_rejected_logps = self.compute_ref_log_probs(inputs)

        (
            dpo_losses,  # Shape: (batch_size,)
            chosen_rewards,  # Shape: (batch_size,)
            rejected_rewards,  # Shape: (batch_size,)
        ) = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        loss = dpo_losses.mean()

        # Log metrics
        metrics = {}
        metrics["rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics["rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics["rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().cpu().mean()
        metrics["rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics["policy_logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics["policy_logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics["referece_logps/chosen"] = reference_chosen_logps.detach().cpu().mean()
        metrics["referece_logps/rejected"] = reference_rejected_logps.detach().cpu().mean()
        metrics["logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
        metrics["logits/rejected"] = policy_rejected_logits.detach().cpu().mean()

        return loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: int = None,  # Do not remove
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        loss, metrics = self.get_batch_loss_metrics(model, inputs, "train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
