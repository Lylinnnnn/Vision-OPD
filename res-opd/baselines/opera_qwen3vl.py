import copy


def locate_image_span(input_ids, image_token_id):
    positions = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("Qwen3VL prompt does not contain image tokens")
    if positions[-1] - positions[0] + 1 != positions.numel():
        raise ValueError("OPERA currently requires one contiguous image-token span")
    return {
        "image_start": int(positions[0]),
        "image_end": int(positions[-1]),
        "response_start": int(input_ids.shape[-1]),
    }


def _last_layer_attention(outputs):
    attentions = getattr(outputs, "attentions", None)
    if not attentions or attentions[-1] is None:
        raise RuntimeError(
            "OPERA requires language-model attentions; load Qwen3VL with "
            "attn_implementation='eager' and output_attentions=True"
        )
    return attentions[-1]


def extend_attention_history(attention_history, current_attention, beam_indices=None):
    current_attention = current_attention.max(1, keepdim=True).values.detach()
    if attention_history is None:
        return current_attention
    if current_attention.shape[-2] != 1:
        raise RuntimeError(
            "Cached OPERA decoding expected one query token, got "
            f"shape={tuple(current_attention.shape)}"
        )
    padded = __import__("torch").cat(
        [attention_history, attention_history.new_zeros((*attention_history.shape[:-1], 1))],
        dim=-1,
    )
    if beam_indices is not None:
        padded = padded[beam_indices]
    return __import__("torch").cat([padded, current_attention], dim=-2)


def compute_opera_candidate_scores(
    candidate_logits,
    candidate_attentions,
    key_position,
    scale_factor=50.0,
    penalty_weights=1.0,
    window_size=512,
):
    import torch

    attention = torch.cat(candidate_attentions, dim=1)
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    response_start = key_position["response_start"]
    local = attention[:, :, response_start:, response_start:]
    response_length = local.shape[-1]
    local = local * scale_factor
    local_scores = local.new_zeros((*local.shape[:2], local.shape[-1]), dtype=torch.float32)
    for position in range(local.shape[-1]):
        local_scores[..., position] = (
            1e-7 * local[..., position:, position].prod(dim=-1).float()
        )
    image_attention = attention[
        :,
        :,
        -1,
        key_position["image_start"] : key_position["image_end"] + 1,
    ].sum(dim=-1)
    rollback_scores, rollback_locations = local_scores.max(dim=-1)
    penalty = -image_attention if response_length <= 10 else rollback_scores
    adjusted = candidate_logits - penalty_weights * penalty
    return adjusted, rollback_locations, response_length


def should_rollback(rollback_history, rollback_location, threshold):
    import torch

    if not rollback_history:
        return False
    history = torch.stack([locations for _, locations in rollback_history], dim=-1)
    counts = (history == rollback_location).long().sum(dim=-1)
    return bool(torch.all(counts > int(threshold)).item())


def _clone_initial_kwargs(model_kwargs):
    cloned = {}
    for key, value in model_kwargs.items():
        if key == "past_key_values":
            cloned[key] = copy.deepcopy(value)
        elif hasattr(value, "clone") and key in {
            "attention_mask",
            "position_ids",
            "token_type_ids",
            "mm_token_type_ids",
        }:
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _clone_candidate_kwargs(model_kwargs):
    cloned = dict(model_kwargs)
    if model_kwargs.get("past_key_values") is not None:
        cloned["past_key_values"] = copy.deepcopy(model_kwargs["past_key_values"])
    return cloned


def _extend_replay_inputs(model_kwargs, sequence_length, prompt_length):
    import torch

    extra = sequence_length - prompt_length
    if extra <= 0:
        return model_kwargs
    if (attention_mask := model_kwargs.get("attention_mask")) is not None:
        model_kwargs["attention_mask"] = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], extra))],
            dim=-1,
        )
    if (token_types := model_kwargs.get("token_type_ids")) is not None:
        model_kwargs["token_type_ids"] = torch.cat(
            [token_types, token_types[..., -1:].expand(*token_types.shape[:-1], extra)],
            dim=-1,
        )
    if (mm_types := model_kwargs.get("mm_token_type_ids")) is not None:
        model_kwargs["mm_token_type_ids"] = torch.cat(
            [mm_types, mm_types.new_zeros((*mm_types.shape[:-1], extra))],
            dim=-1,
        )
    if (position_ids := model_kwargs.get("position_ids")) is not None:
        shape = [1] * (position_ids.dim() - 1) + [extra]
        increments = torch.arange(
            1,
            extra + 1,
            dtype=position_ids.dtype,
            device=position_ids.device,
        ).view(*shape)
        model_kwargs["position_ids"] = torch.cat(
            [position_ids, position_ids[..., -1:] + increments],
            dim=-1,
        )
    if "cache_position" in model_kwargs:
        model_kwargs["cache_position"] = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=model_kwargs["cache_position"].device,
        )
    return model_kwargs


def _cpu_clone(value):
    return value.detach().cpu().clone()


def _snapshot_state(
    response_length,
    cur_len,
    running_sequences,
    running_beam_scores,
    running_beam_indices,
    sequences,
    beam_scores,
    beam_indices,
    is_sent_finished,
    early_stop_state,
    stopping_state,
    incoming_beam_indices,
    incoming_tokens,
):
    return {
        "response_length": response_length,
        "cur_len": cur_len,
        "running_sequences": _cpu_clone(running_sequences),
        "running_beam_scores": _cpu_clone(running_beam_scores),
        "running_beam_indices": _cpu_clone(running_beam_indices),
        "sequences": _cpu_clone(sequences),
        "beam_scores": _cpu_clone(beam_scores),
        "beam_indices": _cpu_clone(beam_indices),
        "is_sent_finished": _cpu_clone(is_sent_finished),
        "early_stop_state": _cpu_clone(early_stop_state),
        "stopping_state": _cpu_clone(stopping_state),
        "incoming_beam_indices": (
            _cpu_clone(incoming_beam_indices) if incoming_beam_indices is not None else None
        ),
        "incoming_tokens": _cpu_clone(incoming_tokens) if incoming_tokens is not None else None,
    }


def _restore_state(snapshot, device):
    tensor_keys = (
        "running_sequences",
        "running_beam_scores",
        "running_beam_indices",
        "sequences",
        "beam_scores",
        "beam_indices",
        "is_sent_finished",
        "early_stop_state",
        "stopping_state",
    )
    restored = {key: snapshot[key].to(device) for key in tensor_keys}
    restored["cur_len"] = snapshot["cur_len"]
    return restored


def opera_generate(
    model,
    input_ids,
    logits_processor,
    stopping_criteria,
    generation_config,
    synced_gpus=False,
    opera_key_position=None,
    opera_scale_factor=50.0,
    opera_threshold=15,
    opera_num_attn_candidates=5,
    opera_penalty_weights=1.0,
    opera_window_size=512,
    opera_runtime=None,
    **model_kwargs,
):
    import torch
    from torch import nn

    if generation_config.do_sample:
        raise ValueError("Official OPERA decoding requires deterministic beam search")
    if generation_config.num_beams <= 1:
        raise ValueError("Official OPERA decoding requires num_beams > 1")
    if generation_config.num_return_sequences != 1:
        raise ValueError("OPERA adapter supports one returned caption per image")
    if generation_config.return_dict_in_generate:
        raise ValueError("OPERA adapter currently returns token sequences only")
    if model.config.is_encoder_decoder:
        raise ValueError("OPERA Qwen3VL adapter only supports decoder-only VLMs")
    if opera_key_position is None:
        raise ValueError("opera_key_position is required")
    if not generation_config.output_attentions:
        raise ValueError("OPERA requires output_attentions=True")
    if input_ids.shape[0] != generation_config.num_beams:
        raise ValueError("OPERA evaluator requires batch_size=1")

    pad_token_id = generation_config._pad_token_tensor
    eos_token_id = generation_config._eos_token_tensor
    max_length = generation_config.max_length
    num_beams = generation_config.num_beams
    batch_size = 1
    cur_len = input_ids.shape[1]
    prompt_length = cur_len
    vocab_size = model.config.get_text_config().vocab_size
    n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
    beams_to_keep = max(2, 1 + n_eos_tokens) * num_beams
    top_num_beam_mask = torch.cat(
        [
            torch.ones(num_beams, dtype=torch.bool),
            torch.zeros(beams_to_keep - num_beams, dtype=torch.bool),
        ]
    ).to(input_ids.device)

    fill_value = int(pad_token_id.item()) if pad_token_id is not None else -1
    running_sequences = torch.full(
        (batch_size, num_beams, max_length),
        fill_value=fill_value,
        dtype=torch.long,
        device=input_ids.device,
    )
    running_sequences[:, :, :cur_len] = model._unflatten_beam_dim(
        input_ids, batch_size, num_beams
    )
    sequences = running_sequences.detach().clone()
    running_beam_scores = torch.zeros((batch_size, num_beams), device=input_ids.device)
    running_beam_scores[:, 1:] = -1e9
    beam_scores = torch.full_like(running_beam_scores, -1e9)
    is_sent_finished = torch.zeros_like(running_beam_scores, dtype=torch.bool)
    early_stop_state = torch.ones((batch_size, 1), dtype=torch.bool, device=input_ids.device)
    stopping_state = torch.zeros_like(running_beam_scores, dtype=torch.bool)
    running_beam_indices = torch.full(
        (batch_size, num_beams, max_length - cur_len),
        -1,
        dtype=torch.int32,
        device=input_ids.device,
    )
    beam_indices = running_beam_indices.detach().clone()

    initial_model_kwargs = _clone_initial_kwargs(model_kwargs)
    flat_running_sequences = input_ids
    model_outputs = model._prefill(
        input_ids,
        generation_config,
        model_kwargs,
        is_first_iteration=not generation_config.is_assistant,
    )
    prefill_consumed = False
    attention_history = None
    selected_beam_indices = None
    incoming_tokens = None
    history_states = []
    rollback_history = []
    rejected = {}
    rollback_counts = {}
    rollback_events = []
    this_peer_finished = False

    while model._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
        if prefill_consumed:
            flat_running_sequences = model._flatten_beam_dim(
                running_sequences[:, :, :cur_len]
            )
            model_inputs = model.prepare_inputs_for_generation(
                flat_running_sequences,
                next_sequence_length=1 if model_kwargs["use_cache"] else None,
                **model_kwargs,
            )
            model_outputs = model(**model_inputs, return_dict=True)
        prefill_consumed = True
        attention_history = extend_attention_history(
            attention_history,
            _last_layer_attention(model_outputs),
            selected_beam_indices,
        )
        model_kwargs = model._update_model_kwargs_for_generation(
            model_outputs,
            model_kwargs,
            is_encoder_decoder=False,
        )
        if synced_gpus and this_peer_finished:
            continue

        response_length = cur_len - prompt_length
        snapshot = _snapshot_state(
            response_length,
            cur_len,
            running_sequences,
            running_beam_scores,
            running_beam_indices,
            sequences,
            beam_scores,
            beam_indices,
            is_sent_finished,
            early_stop_state,
            stopping_state,
            selected_beam_indices,
            incoming_tokens,
        )
        if history_states and history_states[-1]["response_length"] == response_length:
            history_states[-1] = snapshot
        else:
            history_states.append(snapshot)
        if len(history_states) > opera_window_size + 2:
            history_states.pop(0)

        logits = model_outputs.logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=input_ids.device
        )
        if response_length in rejected and rejected[response_length]:
            flat_logits = logits.view(batch_size, num_beams * vocab_size)
            banned = torch.unique(torch.cat(rejected[response_length])).to(flat_logits.device)
            flat_logits[:, banned] = torch.finfo(flat_logits.dtype).min
            logits = flat_logits.view(batch_size * num_beams, vocab_size)

        candidate_logits, candidate_tokens = torch.topk(
            logits,
            k=opera_num_attn_candidates,
            dim=-1,
            largest=True,
            sorted=True,
        )
        candidate_attentions = []
        for candidate_index in range(opera_num_attn_candidates):
            candidate_ids = torch.cat(
                [flat_running_sequences, candidate_tokens[:, candidate_index, None]],
                dim=-1,
            )
            candidate_kwargs = _clone_candidate_kwargs(model_kwargs)
            candidate_inputs = model.prepare_inputs_for_generation(
                candidate_ids,
                next_sequence_length=1 if candidate_kwargs["use_cache"] else None,
                **candidate_kwargs,
            )
            candidate_outputs = model(**candidate_inputs, return_dict=True)
            candidate_row = _last_layer_attention(candidate_outputs).max(
                1, keepdim=True
            ).values.detach()
            padded = torch.cat(
                [
                    attention_history,
                    attention_history.new_zeros((*attention_history.shape[:-1], 1)),
                ],
                dim=-1,
            )
            candidate_attentions.append(torch.cat([padded, candidate_row], dim=-2))
            del candidate_outputs, candidate_inputs, candidate_kwargs

        adjusted_candidates, rollback_locations, candidate_response_length = (
            compute_opera_candidate_scores(
                candidate_logits,
                candidate_attentions,
                opera_key_position,
                scale_factor=opera_scale_factor,
                penalty_weights=opera_penalty_weights,
                window_size=opera_window_size,
            )
        )
        per_beam_locations = rollback_locations.mode(dim=-1).values.detach()
        rollback_location = int(per_beam_locations.mode().values.item())
        rollback_history.append((response_length, per_beam_locations.cpu()))
        if len(rollback_history) > opera_window_size:
            rollback_history.pop(0)

        rollback_position = rollback_location + 1
        if rollback_counts.get(rollback_position, 0) >= opera_num_attn_candidates:
            rollback_position -= 1
        can_rollback = (
            candidate_response_length > 10
            and rollback_location >= 10
            and should_rollback(rollback_history, rollback_location, opera_threshold)
            and rollback_position >= 1
            and rollback_counts.get(rollback_position, 0) < opera_num_attn_candidates
        )
        if can_rollback:
            if history_states and rollback_position < history_states[0]["response_length"]:
                rollback_position = history_states[0]["response_length"]
            target = next(
                (
                    state
                    for state in reversed(history_states)
                    if state["response_length"] == rollback_position
                ),
                None,
            )
            rejected_state = next(
                (
                    state
                    for state in history_states
                    if state["response_length"] == rollback_position + 1
                ),
                None,
            )
            if target is not None and rejected_state is not None:
                parent = rejected_state["incoming_beam_indices"]
                tokens = rejected_state["incoming_tokens"]
                if parent is not None and tokens is not None:
                    banned = parent.long() * vocab_size + tokens.long()
                    rejected.setdefault(rollback_position, []).append(banned.flatten())
                    rollback_counts[rollback_position] = rollback_counts.get(rollback_position, 0) + 1
                    rollback_events.append(
                        {
                            "from": response_length,
                            "to": rollback_position,
                            "attention_location": rollback_location,
                        }
                    )
                    restored = _restore_state(target, input_ids.device)
                    running_sequences = restored["running_sequences"]
                    running_beam_scores = restored["running_beam_scores"]
                    running_beam_indices = restored["running_beam_indices"]
                    sequences = restored["sequences"]
                    beam_scores = restored["beam_scores"]
                    beam_indices = restored["beam_indices"]
                    is_sent_finished = restored["is_sent_finished"]
                    early_stop_state = restored["early_stop_state"]
                    stopping_state = restored["stopping_state"]
                    cur_len = restored["cur_len"]
                    flat_running_sequences = model._flatten_beam_dim(
                        running_sequences[:, :, :cur_len]
                    )
                    history_states = [
                        state
                        for state in history_states
                        if state["response_length"] <= rollback_position
                    ]
                    rollback_history = [
                        item for item in rollback_history if item[0] <= rollback_position
                    ]
                    model_kwargs = _clone_initial_kwargs(initial_model_kwargs)
                    model_kwargs = _extend_replay_inputs(
                        model_kwargs, cur_len, prompt_length
                    )
                    model_outputs = model._prefill(
                        flat_running_sequences,
                        generation_config,
                        model_kwargs,
                        is_first_iteration=True,
                    )
                    attention_history = None
                    selected_beam_indices = None
                    incoming_tokens = None
                    prefill_consumed = False
                    del candidate_attentions, candidate_logits, adjusted_candidates
                    continue

        sparse_logits = torch.full_like(logits, -999.0)
        sparse_logits.scatter_(-1, candidate_tokens, adjusted_candidates)
        log_probs = nn.functional.log_softmax(sparse_logits, dim=-1)
        log_probs = logits_processor(flat_running_sequences, log_probs)
        log_probs = model._unflatten_beam_dim(log_probs, batch_size, num_beams)
        log_probs = log_probs + running_beam_scores[:, :, None]
        log_probs = log_probs.reshape(batch_size, num_beams * vocab_size)
        topk_log_probs, topk_sequences, topk_beam_indices = model._get_top_k_continuations(
            accumulated_log_probs=log_probs,
            running_sequences=running_sequences,
            running_beam_indices=running_beam_indices,
            cur_len=cur_len,
            decoder_prompt_len=prompt_length,
            do_sample=False,
            beams_to_keep=beams_to_keep,
            num_beams=num_beams,
            vocab_size=vocab_size,
            batch_size=batch_size,
        )
        stopping_state = stopping_criteria(
            model._flatten_beam_dim(topk_sequences[:, :, : cur_len + 1]),
            None,
        )
        stopping_state = model._unflatten_beam_dim(
            stopping_state, batch_size, beams_to_keep
        )
        running_sequences, running_beam_scores, running_beam_indices = (
            model._get_running_beams_for_next_iteration(
                topk_log_probs=topk_log_probs,
                topk_running_sequences=topk_sequences,
                topk_running_beam_indices=topk_beam_indices,
                next_token_hits_stopping_criteria=stopping_state,
                num_beams=num_beams,
            )
        )
        sequences, beam_scores, beam_indices, is_sent_finished = model._update_finished_beams(
            sequences=sequences,
            topk_running_sequences=topk_sequences,
            beam_scores=beam_scores,
            topk_log_probs=topk_log_probs,
            beam_indices=beam_indices,
            topk_running_beam_indices=topk_beam_indices,
            is_early_stop_heuristic_unsatisfied=early_stop_state,
            is_sent_finished=is_sent_finished,
            next_token_hits_stopping_criteria=stopping_state,
            top_num_beam_mask=top_num_beam_mask,
            num_beams=num_beams,
            cur_len=cur_len,
            decoder_prompt_len=prompt_length,
            length_penalty=generation_config.length_penalty,
            early_stopping=generation_config.early_stopping,
        )
        selected_beam_indices = model._flatten_beam_dim(
            running_beam_indices[..., cur_len - prompt_length]
        )
        incoming_tokens = model._flatten_beam_dim(
            running_sequences[..., cur_len]
        )
        if model_kwargs.get("past_key_values") is not None:
            if hasattr(model, "_reorder_cache"):
                model_kwargs["past_key_values"] = model._reorder_cache(
                    model_kwargs["past_key_values"], selected_beam_indices
                )
            else:
                model_kwargs["past_key_values"].reorder_cache(selected_beam_indices)
        cur_len += 1
        early_stop_state = model._check_early_stop_heuristic(
            is_early_stop_heuristic_unsatisfied=early_stop_state,
            running_beam_scores=running_beam_scores,
            beam_scores=beam_scores,
            is_sent_finished=is_sent_finished,
            cur_len=cur_len,
            max_length=max_length,
            decoder_prompt_len=prompt_length,
            early_stopping=generation_config.early_stopping,
            length_penalty=generation_config.length_penalty,
        )
        this_peer_finished = not model._beam_search_has_unfinished_sequences(
            early_stop_state,
            is_sent_finished,
            stopping_state,
            generation_config.early_stopping,
        )
        del model_outputs, candidate_attentions

    sequences = model._flatten_beam_dim(sequences[:, :1, :])
    beam_indices = model._flatten_beam_dim(beam_indices[:, :1, :])
    generated_length = ((beam_indices + 1).bool()).sum(dim=1).max()
    sequences = sequences[:, : prompt_length + generated_length]
    if opera_runtime is not None:
        opera_runtime.update(
            {
                "rollback_count": len(rollback_events),
                "rollback_events": rollback_events,
                "rejected_continuations": int(sum(len(items) for items in rejected.values())),
            }
        )
    return sequences
