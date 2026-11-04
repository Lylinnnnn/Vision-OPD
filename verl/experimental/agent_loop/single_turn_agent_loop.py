# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.internvl_hf import (
    assert_internvl_image_alignment,
    assert_internvl_rollout_prompt_alignment,
    is_internvl_hf_processor,
)
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length

        tool_config_path = self.config.data.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if is_internvl_hf_processor(self.processor):
            model_inputs = self.processor(
                text=[self._internvl_raw_prompt], images=images, return_tensors="pt"
            )
            model_prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
            assert_internvl_image_alignment(
                model_inputs["input_ids"], model_inputs["pixel_values"], self.processor, "HF forward"
            )
            if len(model_prompt_ids) > self.prompt_length:
                raise ValueError(
                    f"InternVL expanded prompt length {len(model_prompt_ids)} exceeds {self.prompt_length}"
                )
            prompt_mode = assert_internvl_rollout_prompt_alignment(
                output.prompt_token_ids, prompt_ids, model_prompt_ids
            )
            if prompt_mode == "placeholder":
                logger.info("InternVL vLLM returned image placeholders; using HF-expanded prompt for forward")
            prompt_ids = model_prompt_ids
        response_mask = [1] * len(output.token_ids)
        response_length = self._get_response_length()

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[:response_length],
            response_mask=response_mask[:response_length],
            response_logprobs=output.log_probs[:response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )
        return output
