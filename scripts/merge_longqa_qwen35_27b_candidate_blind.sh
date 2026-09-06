#!/bin/bash

# Merge candidate-blind dev140 and val560 predictions into one 700-row candidate.
set -euo pipefail
ROOT=/CT/NDF/work/waw-26
PYTHON=/CT/NDF/work/miniforge3/envs/wearable-ai/bin/python
DEV=${ROOT}/runs/egolongqa/qwen35_27b_multievidence_candidate_blind_dev140_2026-08-03
VAL=${ROOT}/runs/egolongqa/qwen35_27b_multievidence_candidate_blind_val560_2026-08-03
OUT=${ROOT}/runs/egolongqa/qwen35_27b_multievidence_candidate_blind_full_2026-08-03
for file in "${DEV}/raw_judge_predictions.jsonl" "${VAL}/raw_judge_predictions.jsonl"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: missing candidate-blind merge input: ${file}" >&2
        exit 2
    fi
done
mkdir -p "${OUT}"
"${PYTHON}" "${ROOT}/scripts/merge_longqa_jsonl_shards.py" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --input "${DEV}/raw_judge_predictions.jsonl" \
    --input "${VAL}/raw_judge_predictions.jsonl" \
    --output "${OUT}/predictions.jsonl"
"${PYTHON}" "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUT}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id qwen35_27b_multievidence_candidate_blind_full \
    --output "${OUT}/results.json"
echo "Full candidate-blind predictions: ${OUT}/predictions.jsonl"
