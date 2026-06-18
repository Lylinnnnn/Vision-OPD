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

# Map checkpoint dir name pattern to OSS name
get_oss_name() {
    local ckpt_dir_name="$1"
    # Generic mapping: strip the model-name prefix (Res-OPD-Qwen3VL-2B-Instruct-)
    # then replace all remaining '-' with '_' and prepend 'ResOPD_'.
    # This works for any naming convention (square, original, future modes)
    # without hard-coded regex patterns.
    #
    # Backward-compatible mappings preserve existing OSS paths:
    #   s448-t224-a0.5-ema-e2  → ResOPD_s448_t224_a0.5_ema-e2  (keep last '-')
    #   orig-sr0.75-tr1.0-a0.5-ema-e1 → ResOPD_orig_sr0.75_tr1.0_a0.5_ema-e1
    #
    # Rule: the last '-e{N}' epoch tag keeps its dash for backward compat.
    local suffix
    suffix="${ckpt_dir_name#Res-OPD-Qwen3VL-2B-Instruct-}"
    if [[ "$suffix" == "$ckpt_dir_name" ]]; then
        # Prefix not found; fall back to full replacement
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

merge_and_upload() {
    local step_dir="$1"
    local oss_name="$2"
    local step_name=$(basename "$step_dir")
    local oss_step_path="${OSS_BASE}/${oss_name}/${step_name}"

    log "Merging ${oss_name}/${step_name} ..."
    bash res-opd/scripts/merge_checkpoint.sh "$step_dir" >> "$LOG_FILE" 2>&1

    if [[ ! -f "${step_dir}/model.safetensors" ]]; then
        log "ERROR: merge failed for ${step_dir}, skipping upload"
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
    ossutil cp "$ckpt_file" "${oss_step_path}/model.safetensors" -f >> "$LOG_FILE" 2>&1
    # Upload sha256 file alongside
    ossutil cp "$sha256_file" "${oss_step_path}/model.safetensors.sha256" -f >> "$LOG_FILE" 2>&1
    for f in config.json tokenizer_config.json tokenizer.json chat_template.jinja generation_config.json processor_config.json; do
        if [[ -f "${step_dir}/actor/huggingface/$f" ]]; then
            ossutil cp "${step_dir}/actor/huggingface/$f" "${oss_step_path}/$f" -f >> "$LOG_FILE" 2>&1
        fi
    done

    # Upload eval results if present
    if [[ -d "${step_dir}/eval_results" ]]; then
        log "Uploading eval results for ${step_name} ..."
        ossutil cp -r "${step_dir}/eval_results/" "${oss_step_path}/eval_results/" -f >> "$LOG_FILE" 2>&1
    fi

    # Verify upload by checking OSS file size matches local
    local oss_size=$(ossutil stat "${oss_step_path}/model.safetensors" 2>/dev/null | grep "Content-Length" | awk -F: '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
    if [[ "$oss_size" == "$local_size" ]]; then
        log "  ✅ Upload verified: OSS size=${oss_size} matches local size=${local_size}"
    else
        log "  ⚠️  Size mismatch! OSS=${oss_size} vs local=${local_size} — re-uploading ..."
        ossutil cp "$ckpt_file" "${oss_step_path}/model.safetensors" -f >> "$LOG_FILE" 2>&1
    fi

    # Delete merged safetensors to free disk space
    rm -f "$ckpt_file" "$sha256_file"
    touch "${step_dir}/.oss_uploaded"
    log "DONE: ${oss_step_path}/ (sha256=${local_sha256})"

    # Clean up old FSDP shards: keep only the latest uploaded step per experiment
    local ckpt_parent=$(dirname "$step_dir")
    local current_step_num=$(basename "$step_dir" | sed 's/global_step_//')
    local all_uploaded=true
    for old_step_dir in "${ckpt_parent}"/global_step_*; do
        [[ -d "$old_step_dir" ]] || continue
        [[ "$old_step_dir" == "$step_dir" ]] && continue
        local old_step_num=$(basename "$old_step_dir" | sed 's/global_step_//')
        # Only clean steps older than current AND already uploaded
        if [[ "$old_step_num" -lt "$current_step_num" && -f "${old_step_dir}/.oss_uploaded" ]]; then
            log "Cleaning old FSDP shards: $(basename "$old_step_dir") (uploaded, keeping only latest)"
            rm -rf "${old_step_dir}/actor" "${old_step_dir}/critic" "${old_step_dir}/ref"
            # Keep data.pt and .oss_uploaded marker for reference
            log "  Freed space from $(basename "$old_step_dir")"
        fi
        # Track whether any step is NOT yet uploaded
        if [[ ! -f "${old_step_dir}/.oss_uploaded" ]]; then
            all_uploaded=false
        fi
    done

    # If ALL steps in this experiment are uploaded, also clean current step's FSDP shards
    if $all_uploaded; then
        log "All steps uploaded for $(basename "$ckpt_parent"), cleaning current FSDP shards: $(basename "$step_dir")"
        rm -rf "${step_dir}/actor" "${step_dir}/critic" "${step_dir}/ref"
        log "  Freed space from $(basename "$step_dir")"
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

            # Check if FSDP checkpoint is complete (actor dir + data.pt exist)
            if [[ -d "${step_dir}/actor" && -f "${step_dir}/data.pt" ]]; then
                log "Found FSDP checkpoint: ${EXPERIMENT_NAME}/$(basename "$step_dir")"
                merge_and_upload "$step_dir" "$oss_name"
            fi
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

                # Check if FSDP checkpoint is complete (actor dir + data.pt exist)
                if [[ -d "${step_dir}/actor" && -f "${step_dir}/data.pt" ]]; then
                    log "Found FSDP checkpoint: $(basename "$ckpt_dir")/$(basename "$step_dir")"
                    merge_and_upload "$step_dir" "$oss_name"
                fi
            done
        done

        if $ONCE; then
            log "One-shot scan completed."
            break
        fi
        sleep "$SCAN_INTERVAL"
    done
fi
