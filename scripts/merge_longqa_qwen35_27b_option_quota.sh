#!/bin/bash
set -euo pipefail
ROOT=/CT/NDF/work/waw-26
PY=/CT/NDF/work/miniforge3/envs/wearable-ai/bin/python
ANN=${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl
DEV=${ROOT}/runs/egolongqa/qwen35_27b_option_quota_globalfix_dev140_2026-08-07
VAL=${ROOT}/runs/egolongqa/qwen35_27b_option_quota_globalfix_val560_2026-08-07
OUT=${ROOT}/runs/egolongqa/qwen35_27b_option_quota_globalfix_full_2026-08-07
PRIMARY=${ROOT}/runs/egolongqa/qwen35_q27_thinking_endpoint_majority_full_2026-08-05/predictions.jsonl
ENDPOINT=${ROOT}/runs/egolongqa/qwen35_27b_vllm_uniform64_endpoint_px451584_full_2026-08-07/predictions.jsonl
ENDPOINT_DEV=${ROOT}/runs/egolongqa/qwen35_27b_vllm_uniform64_endpoint_px451584_dev140_2026-08-07/predictions.jsonl
ENDPOINT_VAL=${ROOT}/runs/egolongqa/qwen35_27b_vllm_uniform64_endpoint_px451584_val560_2026-08-07/predictions.jsonl

for file in "${DEV}/predictions.jsonl" "${VAL}/predictions.jsonl"; do
    [[ -s "${file}" ]] || { echo "Missing option-quota merge input: ${file}" >&2; exit 2; }
done
mkdir -p "${OUT}"
"${PY}" "${ROOT}/scripts/merge_longqa_predictions.py" \
    --annotations "${ANN}" \
    --predictions "${DEV}/predictions.jsonl" "${VAL}/predictions.jsonl" \
    --output "${OUT}/predictions.jsonl"
"${PY}" "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUT}/predictions.jsonl" --annotations "${ANN}" \
    --run-id qwen35_27b_option_quota_globalfix_full \
    --output "${OUT}/diagnostics.json"

if [[ ! -s "${ENDPOINT}" && -s "${ENDPOINT_DEV}" && -s "${ENDPOINT_VAL}" ]]; then
    echo "Endpoint partitions are complete; building the endpoint full artifact first."
    bash "${ROOT}/scripts/merge_longqa_qwen35_27b_endpoint.sh"
fi

if [[ -s "${ENDPOINT}" ]]; then
    "${PY}" "${ROOT}/scripts/analyze_longqa_disagreements.py" \
        --annotations "${ANN}" --first "${ENDPOINT}" --second "${OUT}/predictions.jsonl" \
        --first-label endpoint27 --second-label option_quota27 \
        --output-json "${OUT}/endpoint_complementarity.json" \
        --output-jsonl "${OUT}/endpoint_disagreements.jsonl"
    ENSEMBLE=${ROOT}/runs/egolongqa/qwen35_27b_endpoint_optionquota_primary_majority_full_2026-08-07
    mkdir -p "${ENSEMBLE}"
    "${PY}" "${ROOT}/scripts/ensemble_longqa_predictions.py" \
        --annotations "${ANN}" \
        --pred endpoint27="${ENDPOINT}" \
        --pred option_quota27="${OUT}/predictions.jsonl" \
        --pred primary="${PRIMARY}" \
        --mode majority_vote \
        --output "${ENSEMBLE}/predictions.jsonl" \
        --eval-output "${ENSEMBLE}/results.json"
    TEMPORAL_ROUTE=${ROOT}/runs/egolongqa/qwen35_27b_endpoint_optionquota_temporal_route_full_2026-08-07
    mkdir -p "${TEMPORAL_ROUTE}"
    "${PY}" "${ROOT}/scripts/evaluate_longqa_temporal_candidate_route.py" \
        --annotations "${ANN}" \
        --default-candidate "${ENDPOINT}" \
        --temporal-candidate "${OUT}/predictions.jsonl" \
        --operators BEFORE MULTI_TIME \
        --output "${TEMPORAL_ROUTE}/predictions.jsonl" \
        --summary-output "${TEMPORAL_ROUTE}/summary.json"
else
    echo "Endpoint full output is not ready; rerun this merge after endpoint merging to build the frozen three-way ensemble."
fi
echo "Option-quota full output: ${OUT}/predictions.jsonl"
