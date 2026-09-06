#!/bin/bash

# Reproduce the label-free 582/700 ensemble and official upload file.
set -euo pipefail

ROOT=/CT/NDF/work/waw-26
PYTHON=/CT/NDF/work/miniforge3/envs/wearable-ai/bin/python
ANNOTATIONS=${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl
PLAIN=${ROOT}/runs/egolongqa/qwen35_27b_multievidence_candidate_blind_fixed_full_2026-08-05/predictions.jsonl
THINKING=${ROOT}/runs/egolongqa/qwen35_27b_multievidence_candidate_blind_thinking1024_full_2026-08-04/predictions.jsonl
ENDPOINT=${ROOT}/runs/egolongqa/qwen35_9b_vllm_uniform64_endpoint_px451584_full_2026-08-02/predictions.jsonl
RUN_NAME=${RUN_NAME:-qwen35_q27_thinking_endpoint_majority_rebuild_2026-08-06}
SUBMISSION_NAME=${SUBMISSION_NAME:-qwen35_27b_thinking_endpoint_majority_rebuild_2026-08-06}
RUN_DIR=${ROOT}/runs/egolongqa/${RUN_NAME}
SUBMISSION_DIR=${ROOT}/submissions/egolongqa/${SUBMISSION_NAME}

for file in "${ANNOTATIONS}" "${PLAIN}" "${THINKING}" "${ENDPOINT}"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: missing final-majority input: ${file}" >&2
        exit 2
    fi
done
mkdir -p "${RUN_DIR}" "${SUBMISSION_DIR}"
"${PYTHON}" "${ROOT}/scripts/ensemble_longqa_predictions.py" \
    --annotations "${ANNOTATIONS}" \
    --pred "plain27=${PLAIN}" \
    --pred "thinking27=${THINKING}" \
    --pred "endpoint9=${ENDPOINT}" \
    --mode majority_vote \
    --output "${RUN_DIR}/predictions.jsonl" \
    --eval-output "${RUN_DIR}/results.json"
"${PYTHON}" "${ROOT}/scripts/export_longqa_submission.py" \
    --predictions "${RUN_DIR}/predictions.jsonl" \
    --annotations "${ANNOTATIONS}" \
    --output "${SUBMISSION_DIR}/predictions.jsonl"
echo "Run artifact: ${RUN_DIR}"
echo "Upload artifact: ${SUBMISSION_DIR}/predictions.jsonl"
