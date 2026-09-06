#!/bin/bash

# CPU-only grouped OOF test over independent candidates and the larger judge.
set -euo pipefail
ROOT=/CT/NDF/work/waw-26
: "${JUDGE_POLICY:?Set JUDGE_POLICY to the policy promoted on dev140}"
JUDGE=${ROOT}/runs/egolongqa/qwen35_27b_multievidence_primary_judge_${JUDGE_POLICY}_full_2026-08-03/raw_judge_predictions.jsonl
PRIMARY=${ROOT}/runs/egolongqa/qwen35_9b_independent_five_majority_full_2026-08-03/predictions.jsonl
OUT=${ROOT}/analysis/egolongqa/qwen35_27b_confidence_router_${JUDGE_POLICY}_2026-08-03
for file in "${JUDGE}" "${PRIMARY}"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: missing router input: ${file}" >&2
        exit 2
    fi
done
mkdir -p "${OUT}"
source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai
python "${ROOT}/scripts/evaluate_longqa_multicandidate_router.py" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --fallback-predictions "${PRIMARY}" \
    --judge-predictions "${JUDGE}" \
    --output "${OUT}/router_oof.jsonl" \
    --summary-output "${OUT}/router_oof_summary.json"
