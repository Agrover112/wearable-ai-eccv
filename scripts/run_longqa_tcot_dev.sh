#!/bin/bash

# Shared execution body for Qwen Temporal Chain-of-Thought dev140 runs.

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="/CT/NDF/work/waw-26"
STARTER_KIT_DIR="${STARTER_KIT_DIR:-${ROOT}/data/wearable-ai/starter_kit}"
RUN_START_EPOCH="$(date +%s)"
RUN_DATE="$(date +%F)"
SUBSET_FILE="${SUBSET_FILE:-${ROOT}/configs/egolongqa_dev140_seed20260709.json}"

: "${TCOT_MODE:?TCOT_MODE must be set by the launcher}"
: "${RUN_STEM:?RUN_STEM must be set by the launcher}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX must be set by the launcher}"

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

SELECTION_CACHE_DIR="${TCOT_SELECTION_CACHE_DIR:-/scratch/inf0/user/agaur/wai-26/data/wearable-ai/tcot_selection_cache}"
RUN_NAME="${RUN_NAME:-${RUN_STEM}_${RUN_DATE}}"
OUTPUT_DIR="${OUTPUT_DIR:-output/egolongqa/${RUN_NAME}}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${ROOT}/runs/egolongqa/${RUN_NAME}}"
export VLLM_LOG_DIR="${STARTER_KIT_DIR}/${OUTPUT_DIR}"

mkdir -p \
    "${ROOT}/slurm_logs" \
    "${STARTER_KIT_DIR}/${OUTPUT_DIR}" \
    "${ARCHIVE_DIR}" \
    "${SELECTION_CACHE_DIR}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" \
    "${TRITON_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}"

cd "${STARTER_KIT_DIR}"
USER_SITE="$(python -c 'import site; print(site.USER_SITE)')"
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi

cmd=(
    python run_generate_longqa_tcot.py
    --video-folder ../egolongqa/val
    --subset-file "${SUBSET_FILE}"
    --mode "${TCOT_MODE}"
    --candidate-frames "${CANDIDATE_FRAMES:-128}"
    --segments "${TCOT_SEGMENTS:-1}"
    --max-selected-per-segment "${MAX_SELECTED_PER_SEGMENT:-12}"
    --selector-max-pixels "${SELECTOR_MAX_PIXELS:-50176}"
    --neighborhood-radius "${NEIGHBORHOOD_RADIUS:-2}"
    --selected-quota "${SELECTED_QUOTA:-64}"
    --uniform-quota "${UNIFORM_QUOTA:-0}"
    --max-frames "${FINAL_MAX_FRAMES:-64}"
    --llm-model "${LLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
    --concurrency "${CONCURRENCY:-4}"
    --selection-cache-dir "${SELECTION_CACHE_DIR}"
    --output "${OUTPUT_DIR}/predictions.jsonl"
    --eval-output "${OUTPUT_DIR}/results.json"
)
if [[ "${TCOT_MODE}" != "uniform_answer_cot" ]]; then
    cmd+=(--selection-output "${OUTPUT_DIR}/selections.jsonl")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    cmd+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi

echo "============================================================"
echo "Run: ${RUN_NAME}"
echo "TCoT mode: ${TCOT_MODE}"
echo "Candidates/segments: ${CANDIDATE_FRAMES:-128}/${TCOT_SEGMENTS:-1}"
echo "Final selected/uniform quota: ${SELECTED_QUOTA:-64}/${UNIFORM_QUOTA:-0}"
echo "Selector cache: ${SELECTION_CACHE_DIR}"
echo "Command: ${cmd[*]}"
echo "============================================================"
"${cmd[@]}"

python "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUTPUT_DIR}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id "${RUN_NAME}" \
    --output "${OUTPUT_DIR}/diagnostics.json"

for file in predictions.jsonl selections.jsonl results.json results_summary.json diagnostics.json; do
    if [[ -f "${OUTPUT_DIR}/${file}" ]]; then
        cp "${OUTPUT_DIR}/${file}" "${ARCHIVE_DIR}/${file}"
    fi
done
LATEST_VLLM_LOG="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -n "${LATEST_VLLM_LOG}" ]]; then
    cp "${LATEST_VLLM_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
fi
echo "Runtime seconds: $(($(date +%s) - RUN_START_EPOCH))"
echo "Archive: ${ARCHIVE_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ARCHIVE_DIR}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ARCHIVE_DIR}/slurm.err" || true
fi
