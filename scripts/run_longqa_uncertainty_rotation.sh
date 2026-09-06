#!/bin/bash

# Score uncertainty-selected evidence and matched multi-view arbitration.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER_KIT=${ROOT}/data/wearable-ai/starter_kit
RUN_START=$(date +%s)
: "${RUN_NAME:?RUN_NAME is required}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX is required}"
: "${TERTIARY_RUN:?TERTIARY_RUN is required}"
: "${SELECTION_RUN:?SELECTION_RUN is required}"

PRIMARY_RUN=${PRIMARY_RUN:-qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-29}
SECONDARY_RUN=${SECONDARY_RUN:-qwen35_9b_vllm_uniform64_px451584_full_2026-07-30}
SUBSET_FILE=${SUBSET_FILE:-${ROOT}/configs/egolongqa_dev140_seed20260709.json}
PRIMARY_DIR=${ROOT}/runs/egolongqa/${PRIMARY_RUN}
SECONDARY_DIR=${ROOT}/runs/egolongqa/${SECONDARY_RUN}
TERTIARY_DIR=${ROOT}/runs/egolongqa/${TERTIARY_RUN}
SELECTION_DIR=${ROOT}/runs/egolongqa/${SELECTION_RUN}
OUTPUT_DIR=${STARTER_KIT}/output/egolongqa/${RUN_NAME}
ARCHIVE_DIR=${ROOT}/runs/egolongqa/${RUN_NAME}

for file in \
    "${PRIMARY_DIR}/predictions.jsonl" \
    "${PRIMARY_DIR}/proofpack.jsonl" \
    "${SECONDARY_DIR}/predictions.jsonl" \
    "${TERTIARY_DIR}/predictions.jsonl" \
    "${SELECTION_DIR}/selections.jsonl" \
    "${SUBSET_FILE}"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: required input is missing: ${file}" >&2
        exit 2
    fi
done

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai
SCRATCH_CACHE_ROOT=${SCRATCH_CACHE_ROOT:-/scratch/inf0/user/agaur/wai-26/cache}
export HF_HOME=${HF_HOME:-${SCRATCH_CACHE_ROOT}/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export QWEN_MIN_PIXELS=${QWEN_MIN_PIXELS:-784}
export QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-451584}
export VLLM_QWEN_MAX_MODEL_LEN=${VLLM_QWEN_MAX_MODEL_LEN:-49152}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.92}
export VLLM_LOG_DIR=${OUTPUT_DIR}
mkdir -p "${ROOT}/slurm_logs" "${OUTPUT_DIR}" "${ARCHIVE_DIR}"

python "${ROOT}/scripts/ensemble_longqa_predictions.py" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --subset-file "${SUBSET_FILE}" \
    --pred "q35_pivot=${PRIMARY_DIR}/predictions.jsonl" \
    --pred "q35_uniform=${SECONDARY_DIR}/predictions.jsonl" \
    --pred "candidate_c=${TERTIARY_DIR}/predictions.jsonl" \
    --mode majority_vote \
    --output "${OUTPUT_DIR}/majority_predictions.jsonl" \
    --eval-output "${OUTPUT_DIR}/majority_summary.json"

cd "${STARTER_KIT}"
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi
python -c 'import psutil, vllm'

cmd=(
    python run_score_longqa_disagreements.py
    --video-folder ../egolongqa/val
    --subset-file "${SUBSET_FILE}"
    --primary-predictions "${PRIMARY_DIR}/predictions.jsonl"
    --secondary-predictions "${SECONDARY_DIR}/predictions.jsonl"
    --tertiary-predictions "${TERTIARY_DIR}/predictions.jsonl"
    --prediction-labels q35_pivot q35_uniform candidate_c
    --proofpack "${PRIMARY_DIR}/proofpack.jsonl"
    --proofpack-reference "${PRIMARY_DIR}/predictions.jsonl"
    --uncertainty-selection "${SELECTION_DIR}/selections.jsonl"
    --option-rotations 4
    --score-views pivot uniform uncertainty
    --max-frames 64
    --llm-model "${LLM_MODEL:-Qwen/Qwen3.5-9B}"
    --concurrency 1
    --output "${OUTPUT_DIR}/features.jsonl"
)
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
"${cmd[@]}"

for policy in pivot uncertainty; do
    python "${ROOT}/scripts/evaluate_longqa_rotation_pivot.py" \
        --features "${OUTPUT_DIR}/features.jsonl" \
        --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
        --subset-file "${SUBSET_FILE}" \
        --majority-predictions "${OUTPUT_DIR}/majority_predictions.jsonl" \
        --view "${policy}" \
        --output "${OUTPUT_DIR}/rotation_${policy}_predictions.jsonl" \
        --summary-output "${OUTPUT_DIR}/rotation_${policy}_summary.json"
done
python "${ROOT}/scripts/evaluate_longqa_rotation_pivot.py" \
    --features "${OUTPUT_DIR}/features.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --subset-file "${SUBSET_FILE}" \
    --majority-predictions "${OUTPUT_DIR}/majority_predictions.jsonl" \
    --fusion-views pivot uniform uncertainty \
    --output "${OUTPUT_DIR}/rotation_multiview_predictions.jsonl" \
    --summary-output "${OUTPUT_DIR}/rotation_multiview_summary.json"

cp "${OUTPUT_DIR}"/*.json "${ARCHIVE_DIR}/"
cp "${OUTPUT_DIR}"/*.jsonl "${ARCHIVE_DIR}/"
LATEST_LOG=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
echo "Runtime seconds: $(($(date +%s) - RUN_START))"
echo "Archive: ${ARCHIVE_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ARCHIVE_DIR}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ARCHIVE_DIR}/slurm.err" || true
fi
