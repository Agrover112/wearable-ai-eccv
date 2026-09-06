#!/bin/bash

set -euo pipefail
ROOT=/CT/NDF/work/waw-26
PYTHON=/CT/NDF/work/miniforge3/envs/wearable-ai/bin/python
ANNOTATIONS=${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl
DEV=${ROOT}/runs/egolongqa/qwen35_27b_vllm_endpoint_guarded_midpoint64_px451584_dev140_2026-08-08/predictions.jsonl
LIVE_VAL=${ROOT}/data/wearable-ai/starter_kit/output/egolongqa/qwen35_27b_vllm_endpoint_guarded_midpoint64_px451584_val560_2026-08-08/predictions.jsonl
VAL_SUBSET=${ROOT}/configs/egolongqa_val560_complement_dev140_seed20260709.json
MERGED_VAL=${ROOT}/runs/egolongqa/qwen35_27b_vllm_endpoint_guarded_midpoint64_px451584_val560_accelerated_2026-08-08
OUT=${ROOT}/runs/egolongqa/qwen35_27b_vllm_endpoint_guarded_midpoint64_px451584_full_2026-08-08
SUBMISSION=${ROOT}/submissions/egolongqa/qwen35_27b_endpoint_guarded_midpoint64_2026-08-08
SHARD0=${MERGED_VAL}/shard0_predictions.jsonl

mkdir -p "${MERGED_VAL}" "${OUT}" "${SUBMISSION}"
"${PYTHON}" "${ROOT}/scripts/merge_longqa_jsonl_shards.py" \
    --subset "${ROOT}/configs/egolongqa_val560_contiguous_shard0_of4_seed20260709.json" \
    --input "${LIVE_VAL}" --allow-extras --output "${SHARD0}"

merge_args=()
for shard in 1 2 3; do
    merge_args+=(--input "${ROOT}/runs/egolongqa/qwen35_27b_vllm_endpoint_guarded_midpoint64_px451584_val560_contig_shard${shard}_of4_2026-08-08/predictions.jsonl")
done
"${PYTHON}" "${ROOT}/scripts/merge_longqa_jsonl_shards.py" \
    --subset "${VAL_SUBSET}" --input "${SHARD0}" "${merge_args[@]}" \
    --output "${MERGED_VAL}/predictions.jsonl"

"${PYTHON}" "${ROOT}/scripts/merge_longqa_predictions.py" \
    --annotations "${ANNOTATIONS}" \
    --predictions "${DEV}" "${MERGED_VAL}/predictions.jsonl" \
    --output "${OUT}/predictions.jsonl"
"${PYTHON}" "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUT}/predictions.jsonl" --annotations "${ANNOTATIONS}" \
    --run-id qwen35_27b_endpoint_guarded_midpoint64_full \
    --output "${OUT}/diagnostics.json"
"${PYTHON}" "${ROOT}/scripts/export_longqa_submission.py" \
    --predictions "${OUT}/predictions.jsonl" --annotations "${ANNOTATIONS}" \
    --output "${SUBMISSION}/predictions.jsonl"
echo "Submission: ${SUBMISSION}/predictions.jsonl"
