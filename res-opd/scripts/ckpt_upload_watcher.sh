#!/bin/bash
# =============================================================================
# Checkpoint Upload Watcher
# Monitors checkpoint directories for new global_step_* folders,
# merges them, uploads to OSS, and deletes the merged safetensors to free disk.
#
# Usage:
#   # Monitor ALL experiments (legacy mode)
#   bash res-opd/scripts/ckpt_upload_watcher.sh
#
#   # Monitor a SINGLE experiment directory (recommended)
#   bash res-opd/scripts/ckpt_upload_watcher.sh --watch-dir /path/to/checkpoints/Res-OPD-xxx
#
#   # Scan once and exit (used by run_res_opd.sh after training)
#   bash res-opd/scripts/ckpt_upload_watcher.sh --watch-dir /path/to/checkpoints/Res-OPD-xxx --once
#
# When --watch-dir is specified, only that single experiment directory is
# monitored. This avoids cross-machine conflicts on shared filesystems.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
cd "$VISION_OPD_ROOT"

# Parse arguments
WATCH_DIR=""
ONCE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch-dir)
            WATCH_DIR="$2"
            shift 2
            ;;
        --once)
            ONCE=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

OSS_BASE="${OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4}"
SCAN_INTERVAL="${SCAN_INTERVAL:-30}"  # seconds between scans
CKPT_WATCHER_PRUNE_OPTIMIZER_BEFORE_MERGE="${CKPT_WATCHER_PRUNE_OPTIMIZER_BEFORE_MERGE:-True}"
CKPT_WATCHER_MIN_FREE_MODEL_PCT="${CKPT_WATCHER_MIN_FREE_MODEL_PCT:-60}"
CKPT_WATCHER_MIN_FREE_FIXED_BYTES="${CKPT_WATCHER_MIN_FREE_FIXED_BYTES:-1073741824}"  # 1 GiB
CKPT_WATCHER_KEEP_LOCAL_FSDP_AFTER_UPLOAD="${CKPT_WATCHER_KEEP_LOCAL_FSDP_AFTER_UPLOAD:-False}"
CKPT_WATCHER_DELETE_MERGED_ON_UPLOAD_FAILURE="${CKPT_WATCHER_DELETE_MERGED_ON_UPLOAD_FAILURE:-True}"
CKPT_WATCHER_DELETE_FSDP_ON_MERGE_FAILURE="${CKPT_WATCHER_DELETE_FSDP_ON_MERGE_FAILURE:-False}"

if [[ -n "$WATCH_DIR" ]]; then
    # Single-experiment mode: monitor only the specified directory
    CKPT_BASE="$(dirname "$WATCH_DIR")"
    EXPERIMENT_NAME="$(basename "$WATCH_DIR")"
    LOG_FILE="${RES_OPD_ROOT}/logs/ckpt_watcher_${EXPERIMENT_NAME}.log"
else
    # Legacy mode: monitor all experiments
    CKPT_BASE="${RES_OPD_ROOT}/checkpoints"
    EXPERIMENT_NAME=""
    LOG_FILE="${RES_OPD_ROOT}/logs/ckpt_upload_watcher.log"
fi

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

is_truthy() {
    case "${1:-}" in
        True|true|TRUE|1|yes|YES|y|Y)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Map checkpoint dir name pattern to OSS name
get_oss_name() {
    local ckpt_dir_name="$1"
    # Generic mapping: strip the Res-OPD prefix, then replace all remaining
    # '-' with '_' and prepend 'ResOPD_'.
    #
    # Backward-compatible mappings preserve existing OSS paths:
    #   s448-t224-a0.5-ema-e2  → ResOPD_s448_t224_a0.5_ema-e2  (keep last '-')
    #   orig-sr0.75-tr1.0-a0.5-ema-e1 → ResOPD_orig_sr0.75_tr1.0_a0.5_ema-e1
    #
    # Rule: the last '-e{N}' epoch tag keeps its dash for backward compat.
    local suffix
    if [[ "$ckpt_dir_name" == Res-OPD-Qwen3VL-2B-Instruct-* ]]; then
        # Backward-compatible path for existing Instruct checkpoints.
        suffix="${ckpt_dir_name#Res-OPD-Qwen3VL-2B-Instruct-}"
    elif [[ "$ckpt_dir_name" == Res-OPD-* ]]; then
        suffix="${ckpt_dir_name#Res-OPD-}"
    else
        suffix="$ckpt_dir_name"
    fi

    # Split off the trailing epoch tag (e.g. "-e2", "-e1") to preserve it
    local epoch_tag=""
    if [[ "$suffix" =~ ^(.+)-(e[0-9]+)$ ]]; then
        suffix="${BASH_REMATCH[1]}"
        epoch_tag="-${BASH_REMATCH[2]}"
    fi

    local oss_name="ResOPD_${suffix//-/_}${epoch_tag}"
    echo "$oss_name"
}

file_size_bytes() {
    local path="$1"
    stat -c%s "$path" 2>/dev/null || stat -f%z "$path" 2>/dev/null || echo 0
}

available_bytes() {
    local path="$1"
    df -Pk "$path" | awk 'NR==2 {print $4 * 1024}'
}

sum_actor_model_shard_bytes() {
    local step_dir="$1"
    local actor_dir="${step_dir}/actor"
    local total=0
    local f size
    [[ -d "$actor_dir" ]] || {
        echo 0
        return 0
    }
    while IFS= read -r -d '' f; do
        size="$(file_size_bytes "$f")"
        total=$((total + size))
    done < <(find "$actor_dir" -maxdepth 1 -type f -name 'model_world_size_*_rank_*.pt' -print0)
    echo "$total"
}

count_actor_model_shards() {
    local step_dir="$1"
    local actor_dir="${step_dir}/actor"
    if [[ ! -d "$actor_dir" ]]; then
        echo 0
        return 0
    fi
    find "$actor_dir" -maxdepth 1 -type f -name 'model_world_size_*_rank_*.pt' | wc -l | tr -d '[:space:]'
}

expected_actor_world_size() {
    local step_dir="$1"
    local fsdp_config="${step_dir}/actor/fsdp_config.json"
    if [[ ! -f "$fsdp_config" ]]; then
        echo ""
        return 0
    fi
    grep -o '"world_size"[[:space:]]*:[[:space:]]*[0-9]\+' "$fsdp_config" 2>/dev/null \
        | grep -o '[0-9]\+' \
        | head -n 1 || true
}

fsdp_checkpoint_missing_reason() {
    local step_dir="$1"
    local actor_dir="${step_dir}/actor"
    local hf_config="${actor_dir}/huggingface/config.json"
    local fsdp_config="${actor_dir}/fsdp_config.json"
    local shard_count expected_world_size

    if [[ ! -f "${step_dir}/data.pt" ]]; then
        echo "missing data.pt"
        return 0
    fi
    if [[ ! -d "$actor_dir" ]]; then
        echo "missing actor/"
        return 0
    fi
    if [[ ! -f "$fsdp_config" ]]; then
        echo "missing actor/fsdp_config.json"
        return 0
    fi
    if [[ ! -f "$hf_config" ]]; then
        echo "missing actor/huggingface/config.json"
        return 0
    fi

    shard_count="$(count_actor_model_shards "$step_dir")"
    if [[ "$shard_count" == "0" ]]; then
        echo "missing actor model shards"
        return 0
    fi

    expected_world_size="$(expected_actor_world_size "$step_dir")"
    if [[ -n "$expected_world_size" && "$shard_count" -lt "$expected_world_size" ]]; then
        echo "only ${shard_count}/${expected_world_size} actor model shards present"
        return 0
    fi

    return 1
}

cleanup_partial_merged_files() {
    local step_dir="$1"
    find "$step_dir" -mindepth 1 -maxdepth 1 -type f \( \
        -name 'model.safetensors' -o \
        -name 'model.safetensors.sha256' -o \
        -name '*.safetensors' -o \
        -name '*.safetensors.index.json' \
    \) -print -delete >> "$LOG_FILE" 2>&1 || true
}

cleanup_uploaded_fsdp_shards() {
    local step_dir="$1"
    local reason="${2:-uploaded}"
    if [[ ! -d "$step_dir" ]]; then
        return 0
    fi
    if [[ -d "${step_dir}/actor" || -d "${step_dir}/critic" || -d "${step_dir}/ref" ]]; then
        log "Cleaning FSDP shards: $(basename "$step_dir") (${reason})"
        rm -rf "${step_dir}/actor" "${step_dir}/critic" "${step_dir}/ref"
        log "  Freed FSDP shard dirs from $(basename "$step_dir")"
    fi
}

cleanup_old_uploaded_shards() {
    local step_dir="$1"
    local ckpt_parent
    ckpt_parent="$(dirname "$step_dir")"
    local current_step_num
    current_step_num="$(basename "$step_dir" | sed 's/global_step_//')"
    local old_step_dir old_step_num
    for old_step_dir in "${ckpt_parent}"/global_step_*; do
        [[ -d "$old_step_dir" ]] || continue
        [[ "$old_step_dir" == "$step_dir" ]] && continue
        old_step_num="$(basename "$old_step_dir" | sed 's/global_step_//')"
        if [[ "$old_step_num" -lt "$current_step_num" && -f "${old_step_dir}/.oss_uploaded" ]]; then
            cleanup_uploaded_fsdp_shards "$old_step_dir" "older uploaded step"
        fi
    done
}

all_steps_uploaded() {
    local ckpt_parent="$1"
    local found=false
    local step_dir
    for step_dir in "${ckpt_parent}"/global_step_*; do
        [[ -d "$step_dir" ]] || continue
        found=true
        if [[ ! -f "${step_dir}/.oss_uploaded" ]]; then
            return 1
        fi
    done
    $found
}

prune_optimizer_state_for_merge() {
    local step_dir="$1"
    local total=0
    local f size
    if ! is_truthy "$CKPT_WATCHER_PRUNE_OPTIMIZER_BEFORE_MERGE"; then
        return 0
    fi
    while IFS= read -r -d '' f; do
        size="$(file_size_bytes "$f")"
        total=$((total + size))
        rm -f "$f"
    done < <(find "$step_dir" -type f \( \
        -name 'optim_world_size_*_rank_*.pt' -o \
        -name 'optimizer_world_size_*_rank_*.pt' -o \
        -name 'optim*.pt' -o \
        -name '*optimizer*.pt' \
    \) -print0)
    if (( total > 0 )); then
        log "Pruned optimizer shards before merge: $(basename "$step_dir") freed ${total} bytes"
        touch "${step_dir}/.optimizer_pruned_for_merge"
    fi
}

ensure_merge_space() {
    local step_dir="$1"
    local model_bytes available required
    model_bytes="$(sum_actor_model_shard_bytes "$step_dir")"
    available="$(available_bytes "$step_dir")"
    required=$((model_bytes * CKPT_WATCHER_MIN_FREE_MODEL_PCT / 100 + CKPT_WATCHER_MIN_FREE_FIXED_BYTES))

    log "Disk preflight for $(basename "$step_dir"): actor_model_shards=${model_bytes} available=${available} required=${required}"
    if (( model_bytes <= 0 )); then
        log "ERROR: no actor model shards found under ${step_dir}/actor; cannot merge"
        return 1
    fi
    if (( available < required )); then
        log "ERROR: insufficient free disk for merge: available=${available}, required=${required}"
        return 1
    fi
    return 0
}

merge_and_upload() {
    local step_dir="$1"
    local oss_name="$2"
    local step_name=$(basename "$step_dir")
    local oss_step_path="${OSS_BASE}/${oss_name}/${step_name}"
    local missing_reason

    if missing_reason="$(fsdp_checkpoint_missing_reason "$step_dir")"; then
        log "Checkpoint not ready, skip merge: ${step_name} (${missing_reason})"
        return 1
    fi

    cleanup_partial_merged_files "$step_dir"
    cleanup_old_uploaded_shards "$step_dir"
    prune_optimizer_state_for_merge "$step_dir"
    if ! ensure_merge_space "$step_dir"; then
        log "ERROR: preflight failed for ${step_dir}, skipping merge this scan"
        return 1
    fi

    log "Merging ${oss_name}/${step_name} ..."
    local merge_status=0
    bash res-opd/scripts/merge_checkpoint.sh "$step_dir" >> "$LOG_FILE" 2>&1 || merge_status=$?

    if [[ "$merge_status" -ne 0 || ! -s "${step_dir}/model.safetensors" ]]; then
        log "ERROR: merge failed for ${step_dir} (exit=${merge_status}), skipping upload"
        cleanup_partial_merged_files "$step_dir"
        if is_truthy "$CKPT_WATCHER_DELETE_FSDP_ON_MERGE_FAILURE"; then
            log "WARNING: CKPT_WATCHER_DELETE_FSDP_ON_MERGE_FAILURE=True; deleting local FSDP shards after merge failure."
            cleanup_uploaded_fsdp_shards "$step_dir" "merge failed and deletion explicitly enabled"
        fi
        return 1
    fi

    # Compute sha256 checksum before upload
    local ckpt_file="${step_dir}/model.safetensors"
    local sha256_file="${step_dir}/model.safetensors.sha256"
    log "Computing sha256 for ${step_name} ..."
    sha256sum "$ckpt_file" > "$sha256_file"
    local local_sha256=$(awk '{print $1}' "$sha256_file")
    local local_size=$(stat -c%s "$ckpt_file")
    log "  sha256=${local_sha256}  size=${local_size}"

    log "Uploading to ${oss_step_path}/ ..."
    local upload_status=0
    ossutil cp "$ckpt_file" "${oss_step_path}/model.safetensors" -f >> "$LOG_FILE" 2>&1 || upload_status=$?
    # Upload sha256 file alongside
    ossutil cp "$sha256_file" "${oss_step_path}/model.safetensors.sha256" -f >> "$LOG_FILE" 2>&1 || upload_status=$?
    for f in \
        config.json \
        tokenizer_config.json \
        tokenizer.json \
        chat_template.jinja \
        generation_config.json \
        processor_config.json \
        preprocessor_config.json \
        image_processor_config.json \
        video_processor_config.json \
        special_tokens_map.json \
        tokenizer.model \
        merges.txt \
        vocab.json; do
        if [[ -f "${step_dir}/actor/huggingface/$f" ]]; then
            ossutil cp "${step_dir}/actor/huggingface/$f" "${oss_step_path}/$f" -f >> "$LOG_FILE" 2>&1 || upload_status=$?
        fi
    done

    # Upload eval results if present
    if [[ -d "${step_dir}/eval_results" ]]; then
        log "Uploading eval results for ${step_name} ..."
        ossutil cp -r "${step_dir}/eval_results/" "${oss_step_path}/eval_results/" -f >> "$LOG_FILE" 2>&1 || upload_status=$?
    fi

    if [[ "$upload_status" -ne 0 ]]; then
        log "ERROR: upload failed for ${step_dir} (exit=${upload_status}); keeping model shards for retry"
        if is_truthy "$CKPT_WATCHER_DELETE_MERGED_ON_UPLOAD_FAILURE"; then
            cleanup_partial_merged_files "$step_dir"
        fi
        return 1
    fi

    # Verify upload by checking OSS file size matches local
    local oss_size=$(ossutil stat "${oss_step_path}/model.safetensors" 2>/dev/null | grep "Content-Length" | awk -F: '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
    if [[ "$oss_size" == "$local_size" ]]; then
        log "  ✅ Upload verified: OSS size=${oss_size} matches local size=${local_size}"
    else
        log "  ⚠️  Size mismatch! OSS=${oss_size} vs local=${local_size} — re-uploading ..."
        ossutil cp "$ckpt_file" "${oss_step_path}/model.safetensors" -f >> "$LOG_FILE" 2>&1 || {
            log "ERROR: re-upload failed for ${step_dir}; keeping model shards for retry"
            if is_truthy "$CKPT_WATCHER_DELETE_MERGED_ON_UPLOAD_FAILURE"; then
                cleanup_partial_merged_files "$step_dir"
            fi
            return 1
        }
        oss_size=$(ossutil stat "${oss_step_path}/model.safetensors" 2>/dev/null | grep "Content-Length" | awk -F: '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
        if [[ "$oss_size" != "$local_size" ]]; then
            log "ERROR: OSS size still mismatched after re-upload: OSS=${oss_size}, local=${local_size}"
            if is_truthy "$CKPT_WATCHER_DELETE_MERGED_ON_UPLOAD_FAILURE"; then
                cleanup_partial_merged_files "$step_dir"
            fi
            return 1
        fi
    fi

    # Upload traces if present (once per experiment, on first step upload)
    local trace_dir="${RES_OPD_ROOT}/traces/${EXPERIMENT_NAME}"
    local oss_trace_path="${OSS_BASE}/${oss_name}/training_artifacts/traces"
    if [[ -d "$trace_dir" ]] && [[ ! -f "${WATCH_DIR:-${CKPT_BASE}/${EXPERIMENT_NAME}}/.traces_uploaded" ]]; then
        local trace_marker="${WATCH_DIR:-${CKPT_BASE}/${EXPERIMENT_NAME}}/.traces_uploaded"
        log "Uploading traces: ${trace_dir} -> ${oss_trace_path}/ ..."
        ossutil cp -r "${trace_dir%/}/" "${oss_trace_path%/}/" -f >> "$LOG_FILE" 2>&1
        touch "$trace_marker"
        log "  ✅ Traces uploaded and marked."
    fi

    # Delete merged safetensors to free disk space
    rm -f "$ckpt_file" "$sha256_file"
    touch "${step_dir}/.oss_uploaded"
    log "DONE: ${oss_step_path}/ (sha256=${local_sha256})"

    # Clean up old uploaded FSDP shards and, by default, the current uploaded
    # shard dirs too. The merged HF checkpoint is now verified in OSS; keeping
    # local FSDP shards is only useful for local resume and costs tens of GB.
    local ckpt_parent=$(dirname "$step_dir")
    cleanup_old_uploaded_shards "$step_dir"
    if ! is_truthy "$CKPT_WATCHER_KEEP_LOCAL_FSDP_AFTER_UPLOAD"; then
        cleanup_uploaded_fsdp_shards "$step_dir" "current uploaded step"
    fi

    if all_steps_uploaded "$ckpt_parent"; then
        if [[ -d "$trace_dir" ]]; then
            log "All steps uploaded, cleaning local trace dir: ${trace_dir}"
            rm -rf "$trace_dir"
            log "  ✅ Trace dir cleaned."
        fi
    fi
}

# =============================================================================
# Main loop
# =============================================================================
if [[ -n "$WATCH_DIR" ]]; then
    log "=========================================="
    log "Checkpoint Upload Watcher started (single-experiment mode)"
    log "Monitoring: ${WATCH_DIR}"
    log "OSS target: ${OSS_BASE}"
    log "Scan interval: ${SCAN_INTERVAL}s"
    log "Once mode: ${ONCE}"
    log "=========================================="

    oss_name=$(get_oss_name "$EXPERIMENT_NAME")
    if [[ -z "$oss_name" ]]; then
        log "ERROR: Cannot parse experiment name '${EXPERIMENT_NAME}'"
        exit 1
    fi

    while true; do
        for step_dir in "${WATCH_DIR}"/global_step_*; do
            [[ -d "$step_dir" ]] || continue

            # Skip already uploaded
            if [[ -f "${step_dir}/.oss_uploaded" ]]; then
                continue
            fi

            missing_reason="$(fsdp_checkpoint_missing_reason "$step_dir")"
            if [[ "$?" -eq 0 ]]; then
                if [[ -d "${step_dir}/actor" || -f "${step_dir}/data.pt" ]]; then
                    log "Checkpoint not ready: ${EXPERIMENT_NAME}/$(basename "$step_dir") (${missing_reason})"
                fi
                continue
            fi

            log "Found FSDP checkpoint: ${EXPERIMENT_NAME}/$(basename "$step_dir")"
            merge_and_upload "$step_dir" "$oss_name"
        done

        if $ONCE; then
            log "One-shot scan completed."
            break
        fi
        sleep "$SCAN_INTERVAL"
    done
else
    log "=========================================="
    log "Checkpoint Upload Watcher started (all-experiments mode)"
    log "Monitoring: ${CKPT_BASE}"
    log "OSS target: ${OSS_BASE}"
    log "Scan interval: ${SCAN_INTERVAL}s"
    log "Once mode: ${ONCE}"
    log "=========================================="

    # NOTE: all-experiments mode only scans directories matching Res-OPD-*.
    # If future experiments use a different prefix, use --watch-dir instead.
    while true; do
        for ckpt_dir in "${CKPT_BASE}"/Res-OPD-*; do
            [[ -d "$ckpt_dir" ]] || continue

            oss_name=$(get_oss_name "$(basename "$ckpt_dir")")
            if [[ -z "$oss_name" ]]; then
                continue
            fi

            for step_dir in "${ckpt_dir}"/global_step_*; do
                [[ -d "$step_dir" ]] || continue

                # Skip already uploaded
                if [[ -f "${step_dir}/.oss_uploaded" ]]; then
                    continue
                fi

                missing_reason="$(fsdp_checkpoint_missing_reason "$step_dir")"
                if [[ "$?" -eq 0 ]]; then
                    if [[ -d "${step_dir}/actor" || -f "${step_dir}/data.pt" ]]; then
                        log "Checkpoint not ready: $(basename "$ckpt_dir")/$(basename "$step_dir") (${missing_reason})"
                    fi
                    continue
                fi

                log "Found FSDP checkpoint: $(basename "$ckpt_dir")/$(basename "$step_dir")"
                merge_and_upload "$step_dir" "$oss_name"
            done
        done

        if $ONCE; then
            log "One-shot scan completed."
            break
        fi
        sleep "$SCAN_INTERVAL"
    done
fi
