#!/bin/bash

# Shared Qwen answer stage for normalized HieraMamba temporal proposals.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER_KIT="${ROOT}/data/wearable-ai/starter_kit"
RUN_DATE="$(date +%F)"
RUN_START="$(date +%s)"

: "${HIERA_STRATEGY:?HIERA_STRATEGY is required}"
: "${RUN_STEM:?RUN_STEM is required}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX is required}"

PROPOSALS="${HIERA_PROPOSALS:-/scratch/inf0/user/agaur/wai-26/data/wearable-ai/hieramamba/dev20/proposals.jsonl}"
SUBSET_FILE="${SUBSET_FILE:-${ROOT}/configs/egolongqa_dev20_seed20260709.json}"
PROOFPACK="${PROOFPACK:-${ROOT}/runs/egolongqa/qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_dev_2026-07-11/proofpack.jsonl}"
SELECTION_CACHE_DIR="${SELECTION_CACHE_DIR:-/scratch/inf0/user/agaur/wai-26/data/wearable-ai/hieramamba/tcot_cache}"
RUN_NAME="${RUN_NAME:-${RUN_STEM}_${RUN_DATE}}"
OUTPUT_DIR="output/egolongqa/${RUN_NAME}"
ARCHIVE_DIR="${ROOT}/runs/egolongqa/${RUN_NAME}"

for required in "${PROPOSALS}" "${SUBSET_FILE}"; do
    if [[ ! -s "${required}" ]]; then
        echo "ERROR: required HieraMamba input is missing: ${required}" >&2
        exit 2
    fi
done
if [[ "${HIERA_STRATEGY}" == "interval_pivot" && ! -s "${PROOFPACK}" ]]; then
    echo "ERROR: temporal-pivot proof pack is missing: ${PROOFPACK}" >&2
    exit 2
fi

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai

SCRATCH_CACHE_ROOT="${SCRATCH_CACHE_ROOT:-/scratch/inf0/user/agaur/wai-26/cache}"
export HF_HOME="${HF_HOME:-${SCRATCH_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_CACHE_ROOT}/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_CACHE_ROOT}/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${SCRATCH_CACHE_ROOT}/cuda}"
export QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-784}"
export QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-451584}"
export VLLM_QWEN_MAX_MODEL_LEN="${VLLM_QWEN_MAX_MODEL_LEN:-49152}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
export VLLM_LOG_DIR="${STARTER_KIT}/${OUTPUT_DIR}"

mkdir -p "${ROOT}/slurm_logs" "${STARTER_KIT}/${OUTPUT_DIR}" "${ARCHIVE_DIR}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "${SELECTION_CACHE_DIR}" "${HUGGINGFACE_HUB_CACHE}" "${TRITON_CACHE_DIR}"
fi
cd "${STARTER_KIT}"
USER_SITE="$(python -c 'import site; print(site.USER_SITE)')"
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi
python -c 'import psutil, vllm' || {
    echo "ERROR: Qwen/vLLM dependencies are unavailable in the answer environment" >&2
    exit 2
}

cmd=(
    python run_generate_longqa_hieramamba.py
    --video-folder ../egolongqa/val
    --subset-file "${SUBSET_FILE}"
    --proposals "${PROPOSALS}"
    --strategy "${HIERA_STRATEGY}"
    --top-k "${HIERA_TOP_K:-1}"
    --padding-seconds "${HIERA_PADDING_SECONDS:-2}"
    --candidate-frames "${CANDIDATE_FRAMES:-128}"
    --local-frames "${LOCAL_FRAMES:-48}"
    --global-frames "${GLOBAL_FRAMES:-16}"
    --max-frames "${FINAL_MAX_FRAMES:-64}"
    --tcot-segments "${TCOT_SEGMENTS:-4}"
    --max-selected-per-segment "${MAX_SELECTED_PER_SEGMENT:-6}"
    --selector-max-pixels "${SELECTOR_MAX_PIXELS:-50176}"
    --neighborhood-radius "${NEIGHBORHOOD_RADIUS:-1}"
    --selection-cache-dir "${SELECTION_CACHE_DIR}"
    --llm-model "${LLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
    --concurrency "${CONCURRENCY:-8}"
    --output "${OUTPUT_DIR}/predictions.jsonl"
    --selection-output "${OUTPUT_DIR}/selections.jsonl"
    --eval-output "${OUTPUT_DIR}/results.json"
)
if [[ "${HIERA_STRATEGY}" == "interval_pivot" ]]; then
    cmd+=(--proofpack "${PROOFPACK}")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    cmd+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi

echo "Run: ${RUN_NAME}"
echo "Strategy: ${HIERA_STRATEGY}; top-k: ${HIERA_TOP_K:-1}"
echo "Proposals: ${PROPOSALS}"
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
"${cmd[@]}"

python "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUTPUT_DIR}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id "${RUN_NAME}" \
    --output "${OUTPUT_DIR}/diagnostics.json"

for file in predictions.jsonl selections.jsonl results.json results_summary.json diagnostics.json; do
    cp "${OUTPUT_DIR}/${file}" "${ARCHIVE_DIR}/${file}"
done
LATEST_LOG="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
echo "Runtime seconds: $(($(date +%s) - RUN_START))"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ARCHIVE_DIR}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ARCHIVE_DIR}/slurm.err" || true
fi
