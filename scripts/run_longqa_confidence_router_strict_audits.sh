#!/bin/bash

# CPU-only strict holdout, view ablations, repeated grouped CV, and deployment file.
set -euo pipefail

ROOT=/CT/NDF/work/waw-26
FEATURE_RUN=qwen35_9b_vllm_pivot_uniform_mixed_confidence_router_full_2026-07-30
MAJORITY_RUN=offline_majority_qwen35_pivot_qwen35_uniform_qwen3_verifier_2026-07-30
FEATURES=${ROOT}/runs/egolongqa/${FEATURE_RUN}/features.jsonl
MAJORITY=${ROOT}/runs/egolongqa/${MAJORITY_RUN}/predictions.jsonl
ANNOTATIONS=${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl
DEV140=${ROOT}/configs/egolongqa_dev140_seed20260709.json
OUT=${ROOT}/runs/egolongqa/qwen35_confidence_router_strict_audits_2026-07-31

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai
mkdir -p "${OUT}/repeated" "${OUT}/ablations"

python "${ROOT}/scripts/evaluate_longqa_confidence_router_holdout.py" \
    --features "${FEATURES}" \
    --annotations "${ANNOTATIONS}" \
    --majority-predictions "${MAJORITY}" \
    --train-subset "${DEV140}" \
    --output "${OUT}/holdout_val560.jsonl" \
    --summary-output "${OUT}/holdout_val560_summary.json" \
    --model-output "${OUT}/router_model_dev140.json"

for views in all visual blind; do
    case "${views}" in
        all) view_args=(pivot uniform mixed blind) ;;
        visual) view_args=(pivot uniform mixed) ;;
        blind) view_args=(blind) ;;
    esac
    python "${ROOT}/scripts/evaluate_longqa_confidence_router.py" \
        --features "${FEATURES}" \
        --annotations "${ANNOTATIONS}" \
        --majority-predictions "${MAJORITY}" \
        --views "${view_args[@]}" \
        --outer-salt "ablation-${views}" \
        --inner-salt "ablation-${views}" \
        --output "${OUT}/ablations/${views}.jsonl" \
        --summary-output "${OUT}/ablations/${views}_summary.json"
done

repeat_summaries=()
for seed in $(seq 0 9); do
    summary="${OUT}/repeated/seed${seed}_summary.json"
    python "${ROOT}/scripts/evaluate_longqa_confidence_router.py" \
        --features "${FEATURES}" \
        --annotations "${ANNOTATIONS}" \
        --majority-predictions "${MAJORITY}" \
        --outer-salt "strict-repeat-${seed}" \
        --inner-salt "strict-repeat-${seed}" \
        --output "${OUT}/repeated/seed${seed}.jsonl" \
        --summary-output "${summary}"
    repeat_summaries+=("${summary}")
done

python "${ROOT}/scripts/summarize_longqa_router_repeats.py" \
    --inputs "${repeat_summaries[@]}" \
    --output "${OUT}/repeated_summary.json"

python "${ROOT}/scripts/apply_longqa_confidence_router.py" \
    --features "${FEATURES}" \
    --majority-predictions "${MAJORITY}" \
    --model "${OUT}/router_model_dev140.json" \
    --annotations "${ANNOTATIONS}" \
    --output "${OUT}/predictions.jsonl" \
    --summary-output "${OUT}/deployment_summary.json"

echo "Strict confidence-router audits: ${OUT}"
