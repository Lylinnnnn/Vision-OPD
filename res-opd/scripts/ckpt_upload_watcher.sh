#!/bin/bash
# =============================================================================
# Checkpoint Upload Watcher
# Monitors all Res-OPD checkpoint directories for new global_step_* folders,
# merges them, uploads to OSS, and deletes the merged safetensors to free disk.
#
# Usage: bash res-opd/scripts/ckpt_upload_watcher.sh
# Run in a separate tmux session alongside training.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
cd "$VISION_OPD_ROOT"

CKPT_BASE="${RES_OPD_ROOT}/checkpoints"
OSS_BASE="oss://industry-algo/yanlin/ckpt/OPD/v4"
LOG_FILE="${RES_OPD_ROOT}/logs/ckpt_upload_watcher.log"
SCAN_INTERVAL=30  # seconds between scans

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Map checkpoint dir name pattern to OSS name
get_oss_name() {
    local ckpt_dir_name="$1"
    # Extract experiment config from dir name
    # Pattern: Res-OPD-Qwen3VL-2B-Instruct-s{STUDENT}-t{TEACHER}-a{ALPHA}-{MODE}
    # or:      Res-OPD-Qwen3VL-2B-Instruct-s{STUDENT}-a{ALPHA}-{MODE}
    local student_px teacher_px alpha mode
    if [[ "$ckpt_dir_name" =~ s([0-9]+)-t([0-9]+)-a([0-9.]+)-(.+) ]]; then
        student_px="${BASH_REMATCH[1]}"
        teacher_px="${BASH_REMATCH[2]}"
        alpha="${BASH_REMATCH[3]}"
        mode="${BASH_REMATCH[4]}"
        echo "ResOPD_s${student_px}_t${teacher_px}_a${alpha}_${mode}"
    elif [[ "$ckpt_dir_name" =~ s([0-9]+)-a([0-9.]+)-(.+) ]]; then
        student_px="${BASH_REMATCH[1]}"
        alpha="${BASH_REMATCH[2]}"
        mode="${BASH_REMATCH[3]}"
        echo "ResOPD_s${student_px}_a${alpha}_${mode}"
    else
        echo ""
    fi
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
    done
}

# =============================================================================
# Main loop
# =============================================================================
log "=========================================="
log "Checkpoint Upload Watcher started"
log "Monitoring: ${CKPT_BASE}"
log "OSS target: ${OSS_BASE}"
log "Scan interval: ${SCAN_INTERVAL}s"
log "=========================================="

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

    sleep "$SCAN_INTERVAL"
done
