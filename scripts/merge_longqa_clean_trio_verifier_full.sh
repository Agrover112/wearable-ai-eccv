#!/bin/bash

# Combine the frozen dev140 verifier policy with its val560 evaluation output.
set -euo pipefail
ROOT=/CT/NDF/work/waw-26
PYTHON=/CT/NDF/work/miniforge3/envs/wearable-ai/bin/python
POLICY=${POLICY:-conservative}
DEV_RUN=${DEV_RUN:-qwen35_9b_clean_trio_multiview_verifier_dev_2026-08-02}
VAL_RUN=${VAL_RUN:-qwen35_9b_clean_trio_multiview_verifier_val560_2026-08-02}
OUTPUT_RUN=${OUTPUT_RUN:-qwen35_9b_clean_trio_multiview_verifier_${POLICY}_full_2026-08-02}
DEV_PATH=${ROOT}/runs/egolongqa/${DEV_RUN}/${POLICY}_predictions.jsonl
VAL_PATH=${ROOT}/runs/egolongqa/${VAL_RUN}/${POLICY}_predictions.jsonl
OUTPUT_DIR=${ROOT}/runs/egolongqa/${OUTPUT_RUN}
for file in "${DEV_PATH}" "${VAL_PATH}"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: missing verifier merge input: ${file}" >&2
        exit 2
    fi
done
mkdir -p "${OUTPUT_DIR}"
"${PYTHON}" "${ROOT}/scripts/merge_longqa_jsonl_shards.py" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --input "${DEV_PATH}" --input "${VAL_PATH}" \
    --output "${OUTPUT_DIR}/predictions.jsonl"
"${PYTHON}" "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUTPUT_DIR}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id "${OUTPUT_RUN}" --output "${OUTPUT_DIR}/results.json"
echo "Full verifier predictions: ${OUTPUT_DIR}/predictions.jsonl"
