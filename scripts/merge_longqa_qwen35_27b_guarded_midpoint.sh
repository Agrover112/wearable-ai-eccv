#!/bin/bash

set -euo pipefail
ROOT=/CT/NDF/work/waw-26
PYTHON=/CT/NDF/work/miniforge3/envs/wearable-ai/bin/python
ANNOTATIONS=${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl
DEV=${ROOT}/runs/egolongqa/qwen35_27b_vllm_endpoint_guarded_midpoint64_px451584_dev140_2026-08-08
VAL=${ROOT}/runs/egolongqa/qwen35_27b_vllm_endpoint_guarded_midpoint64_px451584_val560_2026-08-08
OUT=${ROOT}/runs/egolongqa/qwen35_27b_vllm_endpoint_guarded_midpoint64_px451584_full_2026-08-08
SUBMISSION=${ROOT}/submissions/egolongqa/qwen35_27b_endpoint_guarded_midpoint64_2026-08-08

mkdir -p "${OUT}" "${SUBMISSION}"
"${PYTHON}" "${ROOT}/scripts/merge_longqa_predictions.py" \
    --annotations "${ANNOTATIONS}" \
    --predictions "${DEV}/predictions.jsonl" "${VAL}/predictions.jsonl" \
    --output "${OUT}/predictions.jsonl"
"${PYTHON}" "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUT}/predictions.jsonl" \
    --annotations "${ANNOTATIONS}" \
    --run-id qwen35_27b_endpoint_guarded_midpoint64_full \
    --output "${OUT}/diagnostics.json"
"${PYTHON}" "${ROOT}/scripts/export_longqa_submission.py" \
    --predictions "${OUT}/predictions.jsonl" \
    --annotations "${ANNOTATIONS}" \
    --output "${SUBMISSION}/predictions.jsonl"
echo "Submission: ${SUBMISSION}/predictions.jsonl"
