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
    local fallback="${home}/.conda/envs/vision-opd/bin/python3"
    res_opd_pick_executable \
        "$fallback" \
        "${home}/.conda/envs/vision-opd/bin/python3" \
        "/home/zhengyanzhao.zyz/.conda/envs/vision-opd/bin/python3" \
        "/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3" \
        "$path_python"
}

res_opd_is_truthy() {
    [[ "${1:-}" == "True" || "${1:-}" == "true" || "${1:-}" == "1" || "${1:-}" == "yes" || "${1:-}" == "YES" || "${1:-}" == "y" || "${1:-}" == "Y" ]]
}

res_opd_validate_python_bin() {
    local python_bin="${1:-}"
    if [[ -z "$python_bin" ]]; then
        echo "ERROR: PYTHON_BIN is empty." >&2
        return 1
    fi
    if [[ ! -x "$python_bin" ]]; then
        echo "ERROR: PYTHON_BIN is not executable: $python_bin" >&2
        return 1
    fi
    case "$python_bin" in
        */envs/vision-opd/bin/python*|*/vision-opd/bin/python*)
            return 0
            ;;
    esac
    if res_opd_is_truthy "${RES_OPD_ALLOW_NON_VISION_OPD_PYTHON:-False}"; then
        echo "WARNING: PYTHON_BIN does not look like the vision-opd conda env: $python_bin" >&2
        return 0
    fi
    echo "ERROR: PYTHON_BIN does not look like the vision-opd conda env: $python_bin" >&2
    echo "Set PYTHON_BIN=/path/to/.conda/envs/vision-opd/bin/python3, or set RES_OPD_ALLOW_NON_VISION_OPD_PYTHON=True to override." >&2
    return 1
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
