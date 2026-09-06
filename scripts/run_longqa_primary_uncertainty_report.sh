#!/bin/bash

# Build the paired uncertainty report after the matched controls/judges finish.
set -euo pipefail
ROOT=/CT/NDF/work/waw-26
PRIMARY_RUN=${PRIMARY_RUN:-qwen35_9b_vllm_rotation_avg_pivot_full_2026-07-31}
TEXT_RUN=${TEXT_RUN:-qwen35_9b_vllm_text_only_full_$(date +%F)}
CENTRAL_RUN=${CENTRAL_RUN:-qwen35_9b_vllm_central_frame_px451584_full_$(date +%F)}
OUTPUT=${OUTPUT:-${ROOT}/analysis/egolongqa/primary_uncertainty_protocol_$(date +%F).json}

predictions=(
    --prediction "primary=${ROOT}/runs/egolongqa/${PRIMARY_RUN}/predictions.jsonl"
    --prediction "text_only=${ROOT}/runs/egolongqa/${TEXT_RUN}/predictions.jsonl"
    --prediction "central_frame=${ROOT}/runs/egolongqa/${CENTRAL_RUN}/predictions.jsonl"
)

exec /CT/NDF/work/miniforge3/envs/wearable-ai/bin/python \
    "${ROOT}/scripts/evaluate_longqa_uncertainty_protocol.py" \
    "${predictions[@]}" --bootstrap-repetitions 10000 --output "${OUTPUT}" "$@"
