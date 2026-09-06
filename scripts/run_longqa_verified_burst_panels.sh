#!/bin/bash

set -euo pipefail

ROOT=/CT/NDF/work/waw-26
STARTER=${ROOT}/data/wearable-ai/starter_kit
: "${SUBSET_FILE:?SUBSET_FILE is required}"
: "${SELECTION_DIR:?SELECTION_DIR is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX is required}"
SELECTION=${SELECTION_DIR}/proofpack.jsonl
OUTPUT=${ROOT}/runs/egolongqa/${RUN_NAME}
[[ -s "${SELECTION}" ]] || {
    echo "ERROR: verified-burst selection is not complete: ${SELECTION}" >&2
    exit 2
}

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/scratch/inf0/user/agaur/wai-26/cache/huggingface}
export QWEN_MIN_PIXELS=784
export QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-451584}
export VLLM_QWEN_MAX_MODEL_LEN=${VLLM_QWEN_MAX_MODEL_LEN:-49152}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.95}
export QWEN_ENABLE_THINKING=0
export VLLM_GDN_PREFILL_BACKEND=triton
export VLLM_LOG_DIR=${OUTPUT}
mkdir -p "${ROOT}/slurm_logs" "${OUTPUT}"

cmd=(
    python "${STARTER}/run_answer_longqa_selection.py"
    --input "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    --video-folder "${ROOT}/data/wearable-ai/egolongqa/val"
    --subset-file "${SUBSET_FILE}"
    --selection "${SELECTION}"
    --indices-key final_indices
    --detail-panel-centers-key selection_meta.verified_centers
    --detail-panel-size 672
    --prompt-variant clause_complete
    --llm-model Qwen/Qwen3.5-27B
    --max-frames 64 --concurrency 1
    --output "${OUTPUT}/predictions.jsonl"
    --eval-output "${OUTPUT}/results.json"
)
if [[ "$#" -gt 0 ]]; then cmd+=("$@"); fi
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then exit 0; fi
"${cmd[@]}"
cp "${SELECTION}" "${OUTPUT}/proofpack.jsonl"
python "${ROOT}/scripts/evaluate_longqa_hard_iteration.py" \
    --subset "${SUBSET_FILE}" --predictions "${OUTPUT}/predictions.jsonl" \
    --output "${OUTPUT}/hard_diagnostics.json"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${OUTPUT}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${OUTPUT}/slurm.err" || true
fi
echo "Archive: ${OUTPUT}"
