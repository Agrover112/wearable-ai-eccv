#!/bin/bash

set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=/CT/NDF/work/waw-26
STARTER=${ROOT}/data/wearable-ai/starter_kit
RUN_START=$(date +%s)
: "${SPECIALIST_MODE:?SPECIALIST_MODE is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX is required}"
: "${SUBSET_FILE:?SUBSET_FILE is required}"

PRIMARY_RUN=${PRIMARY_RUN:-qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-29}
PROOFPACK_RUN=${PROOFPACK_RUN:-qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-12}
DETECTIONS_RUN=${DETECTIONS_RUN:-qwen3_vl_8b_vllm_object_crop_pairs_f56_c8_px451584_dev_2026-07-26}
PRIMARY_DIR=${ROOT}/runs/egolongqa/${PRIMARY_RUN}
PROOFPACK_DIR=${ROOT}/runs/egolongqa/${PROOFPACK_RUN}
DETECTIONS_DIR=${ROOT}/runs/egolongqa/${DETECTIONS_RUN}
OUTPUT_DIR=${STARTER}/output/egolongqa/${RUN_NAME}
ARCHIVE_DIR=${ROOT}/runs/egolongqa/${RUN_NAME}
for file in \
    "${PRIMARY_DIR}/predictions.jsonl" \
    "${PROOFPACK_DIR}/proofpack.jsonl" \
    "${PROOFPACK_DIR}/predictions.jsonl" \
    "${DETECTIONS_DIR}/detections.jsonl" \
    "${SUBSET_FILE}"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: missing specialist input: ${file}" >&2
        exit 2
    fi
done

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai
SCRATCH_CACHE_ROOT=${SCRATCH_CACHE_ROOT:-/scratch/inf0/user/agaur/wai-26/cache}
export HF_HOME=${HF_HOME:-${SCRATCH_CACHE_ROOT}/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export QWEN_MIN_PIXELS=784
export QWEN_MAX_PIXELS=451584
export VLLM_QWEN_MAX_MODEL_LEN=49152
export VLLM_GPU_MEMORY_UTILIZATION=0.92
export QWEN_ENABLE_THINKING=0
export VLLM_REASONING_PARSER=qwen3
export VLLM_GDN_PREFILL_BACKEND=triton
export VLLM_LOG_DIR=${OUTPUT_DIR}
mkdir -p "${ROOT}/slurm_logs" "${OUTPUT_DIR}" "${ARCHIVE_DIR}"
cd "${STARTER}"
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
[[ -d "${USER_SITE}" ]] && export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
cmd=(
    python run_generate_longqa_specialist_evidence.py
    --video-folder ../egolongqa/val
    --subset-file "${SUBSET_FILE}"
    --proofpack "${PROOFPACK_DIR}/proofpack.jsonl"
    --proofpack-reference "${PROOFPACK_DIR}/predictions.jsonl"
    --detections-input "${DETECTIONS_DIR}/detections.jsonl"
    --mode "${SPECIALIST_MODE}"
    --llm-model Qwen/Qwen3.5-9B
    --concurrency 1
    --max-frames 64
    --detail-count 12
    --output "${OUTPUT_DIR}/predictions.jsonl"
    --evidence-output "${OUTPUT_DIR}/evidence.jsonl"
    --eval-output "${OUTPUT_DIR}/results.json"
)
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
"${cmd[@]}"

python "${ROOT}/scripts/overlay_longqa_predictions.py" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --subset-file "${ROOT}/configs/egolongqa_dev140_seed20260709.json" \
    --fallback "${PRIMARY_DIR}/predictions.jsonl" \
    --override "${OUTPUT_DIR}/predictions.jsonl" \
    --output "${OUTPUT_DIR}/overlay_predictions.jsonl" \
    --eval-output "${OUTPUT_DIR}/overlay_results.json"

cp "${OUTPUT_DIR}"/*.json "${ARCHIVE_DIR}/"
cp "${OUTPUT_DIR}"/*.jsonl "${ARCHIVE_DIR}/"
LATEST_LOG=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
echo "Runtime seconds: $(($(date +%s) - RUN_START))"
echo "Archive: ${ARCHIVE_DIR}"
