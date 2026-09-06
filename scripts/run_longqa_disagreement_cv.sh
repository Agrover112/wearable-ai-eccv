#!/bin/bash

set -euo pipefail
ROOT=/CT/NDF/work/waw-26
OUT="${ROOT}/analysis/egolongqa/disagreement_router_2026-07-16"
mkdir -p "${OUT}"
source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai

python "${ROOT}/scripts/build_longqa_disagreement_dataset.py" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --primary "${ROOT}/runs/egolongqa/qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-12/predictions.jsonl" \
    --secondary "${ROOT}/runs/egolongqa/qwen3_vl_8b_vllm_uniform64_px451584_full_2026-07-11/predictions.jsonl" \
    --primary-proofpack "${ROOT}/runs/egolongqa/qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-12/proofpack.jsonl" \
    --verifier "${ROOT}/runs/egolongqa/qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-12/predictions.jsonl" \
    --likelihood "${ROOT}/runs/egolongqa/qwen3_vl_8b_vllm_pivot_letterlogp_blindwm01_px451584_val560_2026-07-15/predictions.jsonl" \
    --output-jsonl "${OUT}/disagreements.jsonl" \
    --output-summary "${OUT}/cv_summary.json"
