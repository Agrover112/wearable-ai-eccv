#!/bin/bash
set -euo pipefail
ROOT=/CT/NDF/work/waw-26
DEV=${ROOT}/runs/egolongqa/qwen35_27b_vllm_uniform64_endpoint_px451584_dev140_2026-08-07
VAL=${ROOT}/runs/egolongqa/qwen35_27b_vllm_uniform64_endpoint_px451584_val560_2026-08-07
OUT=${ROOT}/runs/egolongqa/qwen35_27b_vllm_uniform64_endpoint_px451584_full_2026-08-07
mkdir -p "${OUT}"
python "${ROOT}/scripts/merge_longqa_predictions.py" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --predictions "${DEV}/predictions.jsonl" "${VAL}/predictions.jsonl" \
    --output "${OUT}/predictions.jsonl"
python "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUT}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id qwen35_27b_uniform64_endpoint_full \
    --output "${OUT}/diagnostics.json"

ROUTE=${ROOT}/runs/egolongqa/qwen35_27b_endpoint_global_route_full_2026-08-07
mkdir -p "${ROUTE}"
/CT/NDF/work/miniforge3/envs/wearable-ai/bin/python \
    "${ROOT}/scripts/evaluate_longqa_operator_route.py" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --primary "${ROOT}/runs/egolongqa/qwen35_q27_thinking_endpoint_majority_full_2026-08-05/predictions.jsonl" \
    --global-candidate "${OUT}/predictions.jsonl" \
    --output "${ROUTE}/predictions.jsonl" \
    --summary-output "${ROUTE}/summary.json"
