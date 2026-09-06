#!/bin/bash

# Shared execution and archival body for Minerva-Ego-inspired object hints.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER_KIT=${ROOT}/data/wearable-ai/starter_kit
RUN_DATE=$(date +%F)
RUN_START=$(date +%s)
: "${OBJECT_HINT_MODE:?OBJECT_HINT_MODE is required}"
: "${RUN_STEM:?RUN_STEM is required}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX is required}"

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai
SCRATCH_CACHE_ROOT=${SCRATCH_CACHE_ROOT:-/scratch/inf0/user/agaur/wai-26/cache}
export HF_HOME=${HF_HOME:-${SCRATCH_CACHE_ROOT}/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export TORCH_HOME=${TORCH_HOME:-${SCRATCH_CACHE_ROOT}/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${SCRATCH_CACHE_ROOT}/xdg}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${SCRATCH_CACHE_ROOT}/triton}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${SCRATCH_CACHE_ROOT}/cuda}
export QWEN_MIN_PIXELS=${QWEN_MIN_PIXELS:-784}
export QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-451584}
export VLLM_QWEN_MAX_MODEL_LEN=${VLLM_QWEN_MAX_MODEL_LEN:-49152}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.92}

RUN_NAME=${RUN_NAME:-${RUN_STEM}_${RUN_DATE}}
OUTPUT_DIR=output/egolongqa/${RUN_NAME}
ARCHIVE_DIR=${ROOT}/runs/egolongqa/${RUN_NAME}
SUBSET_FILE=${SUBSET_FILE:-${ROOT}/configs/egolongqa_dev140_seed20260709.json}
PRIMARY_RUN=${PRIMARY_RUN:-qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-12}
PRIMARY_DIR=${ROOT}/runs/egolongqa/${PRIMARY_RUN}
DETECTION_CACHE=${DETECTION_CACHE:-/scratch/inf0/user/agaur/wai-26/data/wearable-ai/object_hint_detection_cache}
export VLLM_LOG_DIR=${STARTER_KIT}/${OUTPUT_DIR}

for required in "${PRIMARY_DIR}/proofpack.jsonl" "${PRIMARY_DIR}/predictions.jsonl" "${SUBSET_FILE}"; do
    if [[ ! -s "${required}" ]]; then
        echo "ERROR: required input is missing or empty: ${required}" >&2
        exit 2
    fi
done

mkdir -p "${ROOT}/slurm_logs" "${STARTER_KIT}/${OUTPUT_DIR}" "${ARCHIVE_DIR}" \
    "${DETECTION_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}"

cd "${STARTER_KIT}"
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi

cmd=(
    python run_generate_longqa_object_hints.py
    --video-folder ../egolongqa/val
    --subset-file "${SUBSET_FILE}"
    --proofpack "${PRIMARY_DIR}/proofpack.jsonl"
    --proofpack-reference "${PRIMARY_DIR}/predictions.jsonl"
    --mode "${OBJECT_HINT_MODE}"
    --detection-cache-dir "${DETECTION_CACHE}"
    --detections-output "${OUTPUT_DIR}/detections.jsonl"
    --output "${OUTPUT_DIR}/predictions.jsonl"
    --eval-output "${OUTPUT_DIR}/results.json"
    --proofpack-detection-frames 24
    --uniform-detection-frames 8
    --detail-count 8
    --track-event-budget 16
    --max-frames 64
    --detector-batch-size 8
    --concurrency 1
)
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi
echo "Command: ${cmd[*]}"
"${cmd[@]}"

python "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUTPUT_DIR}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id "${RUN_NAME}" \
    --output "${OUTPUT_DIR}/diagnostics.json"

for file in predictions.jsonl detections.jsonl results.json results_summary.json diagnostics.json; do
    cp "${OUTPUT_DIR}/${file}" "${ARCHIVE_DIR}/${file}"
done
LATEST_LOG=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ARCHIVE_DIR}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ARCHIVE_DIR}/slurm.err" || true
fi
echo "Runtime seconds: $(($(date +%s) - RUN_START))"
echo "Archive: ${ARCHIVE_DIR}"
