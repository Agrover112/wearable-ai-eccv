#!/bin/bash

# Shared execution and archival body for semantic-ledger experiments.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER_KIT=${ROOT}/data/wearable-ai/starter_kit
RUN_DATE=$(date +%F)
RUN_START=$(date +%s)
: "${EXPERIMENT_MODE:?EXPERIMENT_MODE is required}"
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
DEV20=${ROOT}/configs/egolongqa_dev20_seed20260709.json
DEV140=${ROOT}/configs/egolongqa_dev140_seed20260709.json
PRIMARY_RUN=${PRIMARY_RUN:-qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_dev_2026-07-11}
SECONDARY_RUN=${SECONDARY_RUN:-qwen3_vl_8b_vllm_uniform64_px451584_dev_2026-07-11}
PRIMARY_DIR=${ROOT}/runs/egolongqa/${PRIMARY_RUN}
export VLLM_LOG_DIR=${STARTER_KIT}/${OUTPUT_DIR}
mkdir -p "${ROOT}/slurm_logs" "${STARTER_KIT}/${OUTPUT_DIR}" "${ARCHIVE_DIR}" \
    "${HUGGINGFACE_HUB_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" \
    "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}"

cd "${STARTER_KIT}"
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi

if [[ "${EXPERIMENT_MODE}" == event_ledger ]]; then
    LEDGER_CACHE=${LEDGER_CACHE:-/scratch/inf0/user/agaur/wai-26/data/wearable-ai/event_ledger_cache}
    mkdir -p "${LEDGER_CACHE}"
    cmd=(
        python run_generate_longqa_event_ledger.py
        --video-folder ../egolongqa/val
        --subset-file "${DEV20}"
        --proofpack "${PRIMARY_DIR}/proofpack.jsonl"
        --proofpack-reference "${PRIMARY_DIR}/predictions.jsonl"
        --ledger-cache-dir "${LEDGER_CACHE}"
        --ledger-frames 16
        --ledger-retrieval 8
        --visual-quota 44
        --max-frames 64
        --ledger-max-pixels 200704
        --concurrency 4
        --ledger-output "${OUTPUT_DIR}/event_ledger.jsonl"
        --output "${OUTPUT_DIR}/predictions.jsonl"
        --eval-output "${OUTPUT_DIR}/results.json"
    )
    artifacts=(predictions.jsonl event_ledger.jsonl results.json results_summary.json diagnostics.json)
elif [[ "${EXPERIMENT_MODE}" == event_ledger_detector ]]; then
    : "${LEDGER_RUN:?LEDGER_RUN must name a completed event-ledger run}"
    LEDGER_DIR=${ROOT}/runs/egolongqa/${LEDGER_RUN}
    DETECTION_CACHE=${DETECTION_CACHE:-/scratch/inf0/user/agaur/wai-26/data/wearable-ai/open_vocab_detection_cache}
    mkdir -p "${DETECTION_CACHE}"
    cmd=(
        python run_generate_longqa_event_ledger_detector.py
        --video-folder ../egolongqa/val
        --subset-file "${DEV20}"
        --ledger-input "${LEDGER_DIR}/event_ledger.jsonl"
        --proofpack "${PRIMARY_DIR}/proofpack.jsonl"
        --proofpack-reference "${PRIMARY_DIR}/predictions.jsonl"
        --detection-cache-dir "${DETECTION_CACHE}"
        --detections-output "${OUTPUT_DIR}/detections.jsonl"
        --output "${OUTPUT_DIR}/predictions.jsonl"
        --eval-output "${OUTPUT_DIR}/results.json"
    )
    artifacts=(predictions.jsonl detections.jsonl results.json results_summary.json diagnostics.json)
elif [[ "${EXPERIMENT_MODE}" == semantic_likelihood ]]; then
    SECONDARY_DIR=${ROOT}/runs/egolongqa/${SECONDARY_RUN}
    cmd=(
        python run_generate_longqa_semantic_likelihood.py
        --video-folder ../egolongqa/val
        --subset-file "${DEV140}"
        --primary-predictions "${PRIMARY_DIR}/predictions.jsonl"
        --secondary-predictions "${SECONDARY_DIR}/predictions.jsonl"
        --primary-proofpack "${PRIMARY_DIR}/proofpack.jsonl"
        --scores-output "${OUTPUT_DIR}/semantic_scores.jsonl"
        --output "${OUTPUT_DIR}/predictions.jsonl"
        --eval-output "${OUTPUT_DIR}/results.json"
    )
    artifacts=(predictions.jsonl semantic_scores.jsonl results.json results_summary.json diagnostics.json)
else
    echo "ERROR: unknown EXPERIMENT_MODE=${EXPERIMENT_MODE}" >&2
    exit 2
fi
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
for file in "${artifacts[@]}"; do
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
