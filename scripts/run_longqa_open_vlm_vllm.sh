#!/bin/bash

set -euo pipefail

ROOT="/CT/NDF/work/waw-26"
STARTER_KIT_DIR="${ROOT}/data/wearable-ai/starter_kit"
PYTHON="/CT/NDF/work/miniforge3/envs/wearable-ai/bin/python"

: "${LLM_MODEL:?Set LLM_MODEL to a Hugging Face model ID}"
: "${MODEL_TAG:?Set MODEL_TAG to a filesystem-safe model name}"

RUN_DATE="${RUN_DATE:-$(date +%F)}"
RUN_NAME="${RUN_NAME:-${MODEL_TAG}_video${MAX_FRAMES:-64}_smoke5_${RUN_DATE}}"
MODEL_TYPE="${MODEL_TYPE:-generic}"
MAX_FRAMES="${MAX_FRAMES:-64}"
FRAMES_PER_INTERVAL="${FRAMES_PER_INTERVAL:-${MAX_FRAMES}}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-${MAX_SAMPLES}}"
SUBSET_FILE="${SUBSET_FILE:-${ROOT}/configs/egolongqa_dev140_seed20260709.json}"
OUTPUT_DIR="${STARTER_KIT_DIR}/output/egolongqa/${RUN_NAME}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${ROOT}/runs/egolongqa/${RUN_NAME}}"
SLURM_LOG_PREFIX="${SLURM_LOG_PREFIX:-${MODEL_TAG}_smoke5}"

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai

# vLLM is installed in the conda environment, while a few runtime packages
# (notably psutil) live in the user's site directory on this cluster.
USER_SITE="$("${PYTHON}" -c 'import site; print(site.getusersitepackages())')"
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi

SCRATCH_CACHE_ROOT="${SCRATCH_CACHE_ROOT:-/scratch/inf0/user/agaur/wai-26/cache}"
export HF_HOME="${HF_HOME:-${SCRATCH_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_CACHE_ROOT}/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_CACHE_ROOT}/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${SCRATCH_CACHE_ROOT}/cuda}"
export VLLM_LOG_DIR="${OUTPUT_DIR}"
export VLLM_QWEN_MEDIA_MODE=video
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

PYTHONNOUSERSITE=1 "${PYTHON}" -c "import psutil, vllm" >/dev/null

mkdir -p \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" \
    "${TRITON_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}" \
    "${OUTPUT_DIR}" \
    "${ARCHIVE_DIR}" \
    "${ROOT}/slurm_logs"

cmd=(
    "${PYTHON}" "${STARTER_KIT_DIR}/run_evaluation.py"
    --task longqa
    --model-type "${MODEL_TYPE}"
    --llm-model "${LLM_MODEL}"
    --backend vllm
    --tp 1
    --concurrency 1
    --num-gpus 1
    --batch-size 1
    --golden "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    --video-folder "${ROOT}/data/wearable-ai/egolongqa/val"
    --subset-file "${SUBSET_FILE}"
    --max-frames "${MAX_FRAMES}"
    --frames-per-interval "${FRAMES_PER_INTERVAL}"
    --uniform-sampling endpoint_inclusive
    --include-frame-timestamps
    --media-mode video
    --prompt-variant baseline
    --longqa-max-new-tokens 32
    --predictions "${OUTPUT_DIR}/predictions.jsonl"
    --eval-output "${OUTPUT_DIR}/results.json"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi

echo "============================================================"
echo "Open-VLM LongQA run: ${RUN_NAME}"
echo "Model: ${LLM_MODEL} (${MODEL_TYPE})"
echo "Frames: ${MAX_FRAMES}, endpoint-inclusive, native video payload"
echo "Subset: ${SUBSET_FILE}; samples: ${MAX_SAMPLES:-all}"
echo "Context: ${VLLM_MAX_MODEL_LEN}; GPU memory fraction: ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "Command: ${cmd[*]}"
echo "============================================================"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

RUN_START_EPOCH="$(date +%s)"
"${cmd[@]}"
RUN_SECONDS="$(( $(date +%s) - RUN_START_EPOCH ))"

cp "${OUTPUT_DIR}/predictions.jsonl" "${ARCHIVE_DIR}/predictions.jsonl"
cp "${OUTPUT_DIR}/results.json" "${ARCHIVE_DIR}/results.json"
if [[ -f "${OUTPUT_DIR}/results_summary.json" ]]; then
    cp "${OUTPUT_DIR}/results_summary.json" "${ARCHIVE_DIR}/results_summary.json"
fi
LATEST_VLLM_LOG="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -n "${LATEST_VLLM_LOG}" ]]; then
    cp "${LATEST_VLLM_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
fi

"${PYTHON}" "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${ARCHIVE_DIR}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id "${RUN_NAME}" \
    --output "${ARCHIVE_DIR}/diagnostics.json"

if [[ -n "${EXPECTED_SAMPLES}" ]]; then
    "${PYTHON}" "${ROOT}/scripts/check_open_vlm_smoke.py" \
        --predictions "${ARCHIVE_DIR}/predictions.jsonl" \
        --expected "${EXPECTED_SAMPLES}" \
        --latency-limit 300
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LOG_STEM="${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}"
    [[ -f "${ROOT}/slurm_logs/${LOG_STEM}.out" ]] && \
        cp "${ROOT}/slurm_logs/${LOG_STEM}.out" "${ARCHIVE_DIR}/slurm.out"
    [[ -f "${ROOT}/slurm_logs/${LOG_STEM}.err" ]] && \
        cp "${ROOT}/slurm_logs/${LOG_STEM}.err" "${ARCHIVE_DIR}/slurm.err"
fi

echo "Completed ${RUN_NAME} in ${RUN_SECONDS}s"
echo "Archived at ${ARCHIVE_DIR}"
