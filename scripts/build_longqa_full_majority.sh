#!/bin/bash

# Build a fixed full-validation majority once Qwen3.5 uniform is complete.

set -euo pipefail
ROOT=/CT/NDF/work/waw-26
RUN_DATE=${RUN_DATE:-$(date +%F)}
if [[ -z "${QWEN35_UNIFORM_RUN:-}" ]]; then
    QWEN35_UNIFORM_DIR=$(
        find "${ROOT}/runs/egolongqa" -mindepth 1 -maxdepth 1 -type d \
            -name 'qwen35_9b_vllm_uniform64_px451584_full_*' \
            -exec test -s '{}/predictions.jsonl' ';' -print |
            sort |
            tail -1
    )
    if [[ -z "${QWEN35_UNIFORM_DIR}" ]]; then
        echo "ERROR: no completed Qwen3.5 full uniform run was found" >&2
        exit 2
    fi
    QWEN35_UNIFORM_RUN=$(basename "${QWEN35_UNIFORM_DIR}")
fi
RUN_NAME=${RUN_NAME:-offline_majority_qwen35_pivot_qwen35_uniform_qwen3_verifier_${RUN_DATE}}
OUTPUT_DIR=${ROOT}/runs/egolongqa/${RUN_NAME}

inputs=(
    "${ROOT}/runs/egolongqa/qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-29/predictions.jsonl"
    "${ROOT}/runs/egolongqa/${QWEN35_UNIFORM_RUN}/predictions.jsonl"
    "${ROOT}/runs/egolongqa/qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-12/predictions.jsonl"
)
for input in "${inputs[@]}"; do
    if [[ ! -s "${input}" ]]; then
        echo "ERROR: required majority input is missing: ${input}" >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_DIR}"
python "${ROOT}/scripts/ensemble_longqa_predictions.py" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --pred "qwen35_pivot=${inputs[0]}" \
    --pred "qwen35_uniform=${inputs[1]}" \
    --pred "qwen3_verifier=${inputs[2]}" \
    --mode majority_vote \
    --output "${OUTPUT_DIR}/predictions.jsonl" \
    --eval-output "${OUTPUT_DIR}/diagnostics.json"

echo "Archive: ${OUTPUT_DIR}"
