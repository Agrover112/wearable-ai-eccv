#!/bin/bash

# Merge uncertainty shards, construct clean val560/full candidates, and vote.
set -euo pipefail
ROOT=/CT/NDF/work/waw-26
PYTHON=/CT/NDF/work/miniforge3/envs/wearable-ai/bin/python
ANNOTATIONS=${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl
DEV=${ROOT}/configs/egolongqa_dev140_seed20260709.json
VAL=${ROOT}/configs/egolongqa_val560_complement_dev140_seed20260709.json

U_DEV=${ROOT}/runs/egolongqa/qwen35_9b_vllm_ug_candidate_c_temporal_pivot_dev_2026-08-01
O_DEV=${ROOT}/runs/egolongqa/qwen35_9b_vllm_siglip2_option_quota_pivot_anc24_final64_px451584_dev_2026-08-01
E_DEV=${ROOT}/runs/egolongqa/qwen35_9b_vllm_uniform64_endpoint_px451584_dev_2026-08-01
U_VAL=${ROOT}/runs/egolongqa/qwen35_9b_vllm_uncertainty_pivot_val560_2026-08-02
O_VAL=${ROOT}/runs/egolongqa/qwen35_9b_vllm_siglip2_option_quota_pivot_anc24_final64_px451584_val560_2026-08-02
E_VAL=${ROOT}/runs/egolongqa/qwen35_9b_vllm_uniform64_endpoint_px451584_val560_2026-08-02
U_FULL=${ROOT}/runs/egolongqa/qwen35_9b_vllm_uncertainty_pivot_full_2026-08-02
O_FULL=${ROOT}/runs/egolongqa/qwen35_9b_vllm_siglip2_option_quota_pivot_anc24_final64_px451584_full_2026-08-02
E_FULL=${ROOT}/runs/egolongqa/qwen35_9b_vllm_uniform64_endpoint_px451584_full_2026-08-02
MAJ_VAL=${ROOT}/runs/egolongqa/qwen35_9b_clean_trio_majority_val560_2026-08-02
MAJ_FULL=${ROOT}/runs/egolongqa/qwen35_9b_clean_trio_majority_full_2026-08-02

shard_predictions=()
shard_selections=()
for shard in 0 1 2 3; do
    directory=${ROOT}/runs/egolongqa/qwen35_9b_vllm_uncertainty_pivot_val560_shard${shard}_of4_2026-08-02
    for file in predictions.jsonl selections.jsonl; do
        if [[ ! -s "${directory}/${file}" ]]; then
            echo "ERROR: incomplete uncertainty shard ${shard}: ${directory}/${file}" >&2
            exit 2
        fi
    done
    shard_predictions+=(--input "${directory}/predictions.jsonl")
    shard_selections+=(--input "${directory}/selections.jsonl")
done
for file in \
    "${U_DEV}/predictions.jsonl" "${U_DEV}/selections.jsonl" \
    "${O_DEV}/predictions.jsonl" "${O_DEV}/proofpack.jsonl" \
    "${E_DEV}/predictions.jsonl" \
    "${O_VAL}/predictions.jsonl" "${O_VAL}/proofpack.jsonl" \
    "${E_VAL}/predictions.jsonl"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: missing merge input: ${file}" >&2
        exit 2
    fi
done
mkdir -p "${U_VAL}" "${U_FULL}" "${O_FULL}" "${E_FULL}" "${MAJ_VAL}" "${MAJ_FULL}"

"${PYTHON}" "${ROOT}/scripts/merge_longqa_jsonl_shards.py" \
    --subset "${VAL}" "${shard_predictions[@]}" --output "${U_VAL}/predictions.jsonl"
"${PYTHON}" "${ROOT}/scripts/merge_longqa_jsonl_shards.py" \
    --subset "${VAL}" "${shard_selections[@]}" --output "${U_VAL}/selections.jsonl"
"${PYTHON}" "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${U_VAL}/predictions.jsonl" --annotations "${ANNOTATIONS}" \
    --run-id qwen35_uncertainty_pivot_val560 --output "${U_VAL}/diagnostics.json"

"${PYTHON}" "${ROOT}/scripts/ensemble_longqa_predictions.py" \
    --annotations "${ANNOTATIONS}" --subset-file "${VAL}" \
    --pred "uncertainty_pivot=${U_VAL}/predictions.jsonl" \
    --pred "option_quota=${O_VAL}/predictions.jsonl" \
    --pred "endpoint_uniform=${E_VAL}/predictions.jsonl" \
    --mode majority_vote --output "${MAJ_VAL}/predictions.jsonl" \
    --eval-output "${MAJ_VAL}/results.json"

"${PYTHON}" "${ROOT}/scripts/merge_longqa_jsonl_shards.py" --annotations "${ANNOTATIONS}" \
    --input "${U_DEV}/predictions.jsonl" --input "${U_VAL}/predictions.jsonl" \
    --output "${U_FULL}/predictions.jsonl"
"${PYTHON}" "${ROOT}/scripts/merge_longqa_jsonl_shards.py" --annotations "${ANNOTATIONS}" \
    --input "${U_DEV}/selections.jsonl" --input "${U_VAL}/selections.jsonl" \
    --output "${U_FULL}/selections.jsonl"
"${PYTHON}" "${ROOT}/scripts/merge_longqa_jsonl_shards.py" --annotations "${ANNOTATIONS}" \
    --input "${O_DEV}/predictions.jsonl" --input "${O_VAL}/predictions.jsonl" \
    --output "${O_FULL}/predictions.jsonl"
"${PYTHON}" "${ROOT}/scripts/merge_longqa_jsonl_shards.py" --annotations "${ANNOTATIONS}" \
    --input "${O_DEV}/proofpack.jsonl" --input "${O_VAL}/proofpack.jsonl" \
    --output "${O_FULL}/proofpack.jsonl"
"${PYTHON}" "${ROOT}/scripts/merge_longqa_jsonl_shards.py" --annotations "${ANNOTATIONS}" \
    --input "${E_DEV}/predictions.jsonl" --input "${E_VAL}/predictions.jsonl" \
    --output "${E_FULL}/predictions.jsonl"

"${PYTHON}" "${ROOT}/scripts/ensemble_longqa_predictions.py" \
    --annotations "${ANNOTATIONS}" \
    --pred "uncertainty_pivot=${U_FULL}/predictions.jsonl" \
    --pred "option_quota=${O_FULL}/predictions.jsonl" \
    --pred "endpoint_uniform=${E_FULL}/predictions.jsonl" \
    --mode majority_vote --output "${MAJ_FULL}/predictions.jsonl" \
    --eval-output "${MAJ_FULL}/results.json"
echo "Val560 majority: ${MAJ_VAL}/predictions.jsonl"
echo "Full majority: ${MAJ_FULL}/predictions.jsonl"
