#!/bin/bash

# Merge the frozen dev and held-out val policy after both runs complete.
set -euo pipefail
ROOT=/CT/NDF/work/waw-26
PYTHON=/CT/NDF/work/miniforge3/envs/wearable-ai/bin/python
: "${JUDGE_POLICY:?Set JUDGE_POLICY to the policy promoted on dev140}"
DEV=${ROOT}/runs/egolongqa/qwen35_27b_multievidence_primary_judge_dev140_2026-08-03
VAL=${ROOT}/runs/egolongqa/qwen35_27b_multievidence_primary_judge_val560_2026-08-03
OUT=${ROOT}/runs/egolongqa/qwen35_27b_multievidence_primary_judge_${JUDGE_POLICY}_full_2026-08-03
for file in "${DEV}/${JUDGE_POLICY}_predictions.jsonl" "${VAL}/${JUDGE_POLICY}_predictions.jsonl"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: missing judge merge input: ${file}" >&2
        exit 2
    fi
done
mkdir -p "${OUT}"
"${PYTHON}" "${ROOT}/scripts/merge_longqa_jsonl_shards.py" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --input "${DEV}/raw_judge_predictions.jsonl" \
    --input "${VAL}/raw_judge_predictions.jsonl" \
    --output "${OUT}/raw_judge_predictions.jsonl"
"${PYTHON}" "${ROOT}/scripts/merge_longqa_jsonl_shards.py" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --input "${DEV}/${JUDGE_POLICY}_predictions.jsonl" \
    --input "${VAL}/${JUDGE_POLICY}_predictions.jsonl" \
    --output "${OUT}/predictions.jsonl"
"${PYTHON}" "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUT}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id "qwen35_27b_multievidence_primary_judge_${JUDGE_POLICY}_full" \
    --output "${OUT}/results.json"
echo "Full judge predictions: ${OUT}/predictions.jsonl"
