#!/bin/bash

# Reproduce the strongest fixed three-model dev140 majority vote without GPUs.

set -euo pipefail
ROOT=/CT/NDF/work/waw-26
RUN_DATE=${RUN_DATE:-$(date +%F)}
RUN_NAME=${RUN_NAME:-offline_majority_ug_router_qwen35_pivot_qwen35_uniform_${RUN_DATE}}
OUTPUT_DIR=${ROOT}/runs/egolongqa/${RUN_NAME}

mkdir -p "${OUTPUT_DIR}"
python "${ROOT}/scripts/ensemble_longqa_predictions.py" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --subset-file "${ROOT}/configs/egolongqa_dev140_seed20260709.json" \
    --pred "router=${ROOT}/runs/egolongqa/qwen3_vl_8b_vllm_ug_pivot_uniform_crop_router_px451584_dev_2026-07-29/predictions.jsonl" \
    --pred "qwen35_pivot=${ROOT}/runs/egolongqa/qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_dev_2026-07-29/predictions.jsonl" \
    --pred "qwen35_uniform=${ROOT}/runs/egolongqa/qwen35_9b_vllm_uniform64_px451584_dev_2026-07-29/predictions.jsonl" \
    --mode majority_vote \
    --output "${OUTPUT_DIR}/predictions.jsonl" \
    --eval-output "${OUTPUT_DIR}/diagnostics.json"

echo "Archive: ${OUTPUT_DIR}"
