#!/bin/bash

set -euo pipefail
ROOT=/CT/NDF/work/waw-26
RUNS=${ROOT}/runs/egolongqa
OUTPUT=${ROOT}/analysis/egolongqa/candidate_c_replacement_audit_dev140_2026-08-01.json
PIVOT=${RUNS}/qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-29/predictions.jsonl
UNIFORM=${RUNS}/qwen35_9b_vllm_uniform64_px451584_full_2026-07-30/predictions.jsonl
CURRENT_C=${RUNS}/qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-12/predictions.jsonl
FINAL=${RUNS}/qwen35_9b_vllm_rotation_avg_pivot_full_2026-07-31/predictions.jsonl

cmd=(
    /CT/NDF/work/miniforge3/envs/wearable-ai/bin/python
    "${ROOT}/scripts/audit_longqa_candidate_replacement.py"
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    --subset-file "${ROOT}/configs/egolongqa_dev140_seed20260709.json"
    --primary "${PIVOT}"
    --secondary "${UNIFORM}"
    --current-c "${CURRENT_C}"
    --final-predictions "${FINAL}"
    --output "${OUTPUT}"
)
for named_run in \
    "qwen_embedding=qwen35_9b_vllm_qwen3vl_embed2b_temporal_pivot_anc24_final64_px451584_dev_2026-08-01" \
    "endpoint_uniform=qwen35_9b_vllm_uniform64_endpoint_px451584_dev_2026-08-01"; do
    name=${named_run%%=*}
    run=${named_run#*=}
    path=${RUNS}/${run}/predictions.jsonl
    if [[ -s "${path}" ]]; then
        cmd+=(--alternative "${name}=${path}")
    fi
done
mkdir -p "$(dirname "${OUTPUT}")"
echo "Command: ${cmd[*]}"
"${cmd[@]}"
