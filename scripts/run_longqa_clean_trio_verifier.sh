#!/bin/bash

# Majority and terminal verification for three independent Qwen answer branches.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER=${ROOT}/data/wearable-ai/starter_kit
: "${RUN_NAME:?RUN_NAME is required}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX is required}"
: "${SUBSET_FILE:?SUBSET_FILE is required}"
: "${UNCERTAINTY_RUN:?UNCERTAINTY_RUN is required}"
: "${OPTION_QUOTA_RUN:?OPTION_QUOTA_RUN is required}"
: "${ENDPOINT_RUN:?ENDPOINT_RUN is required}"

U_DIR=${ROOT}/runs/egolongqa/${UNCERTAINTY_RUN}
O_DIR=${ROOT}/runs/egolongqa/${OPTION_QUOTA_RUN}
E_DIR=${ROOT}/runs/egolongqa/${ENDPOINT_RUN}
OUTPUT_DIR=${STARTER}/output/egolongqa/${RUN_NAME}
ARCHIVE_DIR=${ROOT}/runs/egolongqa/${RUN_NAME}
for file in \
    "${U_DIR}/predictions.jsonl" \
    "${U_DIR}/selections.jsonl" \
    "${O_DIR}/predictions.jsonl" \
    "${O_DIR}/proofpack.jsonl" \
    "${E_DIR}/predictions.jsonl" \
    "${SUBSET_FILE}"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: missing clean-trio input: ${file}" >&2
        exit 2
    fi
done

specialist_args=()
score_views=(pivot uniform uncertainty)
fusion_views=(pivot uniform uncertainty)
if [[ "${USE_SPECIALIST_EVIDENCE:-0}" == "1" ]]; then
    OCR_PATH=${OCR_PATH:-${ROOT}/runs/egolongqa/qwen35_9b_vllm_targeted_qwen_ocr_dev_2026-08-01/predictions.jsonl}
    REID_PATH=${REID_PATH:-${ROOT}/runs/egolongqa/qwen35_9b_vllm_siglip2_object_reid_dev_2026-08-01/predictions.jsonl}
    for file in "${OCR_PATH}" "${REID_PATH}"; do
        if [[ ! -s "${file}" ]]; then
            echo "ERROR: missing specialist evidence: ${file}" >&2
            exit 2
        fi
        specialist_args+=(--specialist-predictions "${file}")
    done
    score_views+=(specialist)
    fusion_views+=(specialist)
fi

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
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi
if ! python -c 'import psutil, vllm' >/dev/null 2>&1; then
    echo "ERROR: verifier environment cannot import psutil and vllm" >&2
    python -c 'import psutil, vllm'
fi

majority_cmd=(
    python "${ROOT}/scripts/ensemble_longqa_predictions.py"
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    --subset-file "${SUBSET_FILE}"
    --pred "uncertainty_pivot=${U_DIR}/predictions.jsonl"
    --pred "option_quota=${O_DIR}/predictions.jsonl"
    --pred "endpoint_uniform=${E_DIR}/predictions.jsonl"
    --mode majority_vote
    --output "${OUTPUT_DIR}/majority_predictions.jsonl"
    --eval-output "${OUTPUT_DIR}/majority_summary.json"
)
score_cmd=(
    python "${STARTER}/run_score_longqa_disagreements.py"
    --video-folder "${ROOT}/data/wearable-ai/egolongqa/val"
    --subset-file "${SUBSET_FILE}"
    --primary-predictions "${U_DIR}/predictions.jsonl"
    --secondary-predictions "${O_DIR}/predictions.jsonl"
    --tertiary-predictions "${E_DIR}/predictions.jsonl"
    --prediction-labels uncertainty_pivot option_quota endpoint_uniform
    --proofpack "${O_DIR}/proofpack.jsonl"
    --proofpack-reference "${O_DIR}/predictions.jsonl"
    --uncertainty-selection "${U_DIR}/selections.jsonl"
    --option-rotations 4
    --score-views "${score_views[@]}"
    "${specialist_args[@]}"
    --max-frames 64
    --llm-model Qwen/Qwen3.5-9B
    --concurrency 1
    --output "${OUTPUT_DIR}/features.jsonl"
)

echo "Majority command: ${majority_cmd[*]}"
echo "Verifier command: ${score_cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
"${majority_cmd[@]}"
"${score_cmd[@]}"

common_eval=(
    --features "${OUTPUT_DIR}/features.jsonl"
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    --subset-file "${SUBSET_FILE}"
    --majority-predictions "${OUTPUT_DIR}/majority_predictions.jsonl"
    --fusion-views "${fusion_views[@]}"
)
if [[ "${USE_SPECIALIST_EVIDENCE:-0}" == "1" ]]; then
    common_eval+=(--allow-missing-fusion-views)
fi
for policy in all_disagreements all_different_only conservative; do
    extra=()
    if [[ "${policy}" == "conservative" ]]; then
        extra+=(--two-one-min-margin 0.10 --two-one-min-view-support 2)
    fi
    python "${ROOT}/scripts/evaluate_longqa_rotation_pivot.py" \
        "${common_eval[@]}" \
        --agreement-policy "${policy}" \
        "${extra[@]}" \
        --output "${OUTPUT_DIR}/${policy}_predictions.jsonl" \
        --summary-output "${OUTPUT_DIR}/${policy}_summary.json"
done

cp "${OUTPUT_DIR}"/*.json "${ARCHIVE_DIR}/"
cp "${OUTPUT_DIR}"/*.jsonl "${ARCHIVE_DIR}/"
LATEST_LOG=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ARCHIVE_DIR}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ARCHIVE_DIR}/slurm.err" || true
fi
echo "Archive: ${ARCHIVE_DIR}"
