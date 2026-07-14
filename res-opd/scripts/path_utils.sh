#!/usr/bin/env bash

# Shared path discovery for machines that keep the repo/model/data roots in
# different user directories. Explicit environment variables still win.

res_opd_pick_existing_path() {
    local fallback="$1"
    shift || true
    local path
    for path in "$@"; do
        if [[ -n "${path:-}" && -e "$path" ]]; then
            printf '%s\n' "$path"
            return 0
        fi
    done
    printf '%s\n' "$fallback"
}

res_opd_pick_executable() {
    local fallback="$1"
    shift || true
    local path
    for path in "$@"; do
        if [[ -n "${path:-}" && -x "$path" ]]; then
            printf '%s\n' "$path"
            return 0
        fi
    done
    printf '%s\n' "$fallback"
}

res_opd_default_python_bin() {
    local home="${HOME:-}"
    local path_python
    path_python="$(command -v python3 2>/dev/null || true)"
    local fallback="${path_python:-${home}/.conda/envs/vision-opd/bin/python3}"
    res_opd_pick_executable \
        "$fallback" \
        "${home}/.conda/envs/vision-opd/bin/python3" \
        "/home/zhengyanzhao.zyz/.conda/envs/vision-opd/bin/python3" \
        "/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3" \
        "$path_python"
}

res_opd_default_model_root() {
    local home="${HOME:-}"
    local fallback="${home}/notebook/model/qwen"
    res_opd_pick_existing_path \
        "$fallback" \
        "${RES_OPD_MODEL_ROOT:-}" \
        "${home}/notebook/model/qwen" \
        "${home}/notebook/yanlin/model/qwen" \
        "/home/zhengyanzhao.zyz/notebook/yanlin/model/qwen" \
        "/home/liuyanlin.lyl/notebook/model/qwen"
}

res_opd_default_data_root() {
    local home="${HOME:-}"
    local fallback="${home}/notebook/data"
    res_opd_pick_existing_path \
        "$fallback" \
        "${BENCHMARK_DATA_DIR:-}" \
        "${VISION_BENCHMARK_DATA_DIR:-}" \
        "${home}/notebook/data" \
        "${home}/notebook/yanlin/data" \
        "/home/zhengyanzhao.zyz/notebook/yanlin/data" \
        "/home/liuyanlin.lyl/notebook/data"
}
