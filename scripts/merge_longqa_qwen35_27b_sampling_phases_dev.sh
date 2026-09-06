#!/bin/bash
set -euo pipefail

ROOT=/CT/NDF/work/waw-26
PY=/CT/NDF/work/miniforge3/envs/wearable-ai/bin/python
ANN=${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl
DEV=${ROOT}/configs/egolongqa_dev140_seed20260709.json
ENDPOINT=${ROOT}/runs/egolongqa/qwen35_27b_vllm_uniform64_endpoint_px451584_dev140_2026-08-07/predictions.jsonl
LEGACY=${ROOT}/runs/egolongqa/qwen35_27b_vllm_uniform64_legacy_px451584_dev140_2026-08-07/predictions.jsonl
MIDPOINT=${ROOT}/runs/egolongqa/qwen35_27b_vllm_uniform64_midpoint_px451584_dev140_2026-08-07/predictions.jsonl
OUT=${ROOT}/runs/egolongqa/qwen35_27b_uniform64_three_phase_majority_dev140_2026-08-07

for file in "${ENDPOINT}" "${LEGACY}" "${MIDPOINT}"; do
    [[ -s "${file}" ]] || { echo "Missing phase candidate: ${file}" >&2; exit 2; }
done
mkdir -p "${OUT}"

"${PY}" "${ROOT}/scripts/ensemble_longqa_predictions.py" \
    --annotations "${ANN}" --subset-file "${DEV}" \
    --pred endpoint="${ENDPOINT}" \
    --pred legacy="${LEGACY}" \
    --pred midpoint="${MIDPOINT}" \
    --mode majority_vote \
    --output "${OUT}/predictions.jsonl" \
    --eval-output "${OUT}/diagnostics.json"

"${PY}" "${ROOT}/scripts/analyze_longqa_candidate_pool.py" \
    --annotations "${ANN}" --subset-file "${DEV}" \
    --pred endpoint="${ENDPOINT}" \
    --pred legacy="${LEGACY}" \
    --pred midpoint="${MIDPOINT}" \
    --output "${OUT}/candidate_pool_summary.json"

echo "Three-phase dev analysis: ${OUT}"
