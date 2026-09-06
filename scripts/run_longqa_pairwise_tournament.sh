#!/bin/bash

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER_KIT_DIR="${ROOT}/data/wearable-ai/starter_kit"
RUN_DATE="$(date +%F)"
SUBSET_FILE="${SUBSET_FILE:-${ROOT}/configs/egolongqa_dev140_seed20260709.json}"
RUN_NAME="${RUN_NAME:-qwen35_9b_pairwise_candidate_tournament_dev_${RUN_DATE}}"
OUTPUT_DIR="${STARTER_KIT_DIR}/output/egolongqa/${RUN_NAME}"
ARCHIVE_DIR="${ROOT}/runs/egolongqa/${RUN_NAME}"

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai
export HF_HOME="${HF_HOME:-/scratch/inf0/user/agaur/wai-26/cache/huggingface}"
export VLLM_LOG_DIR="${OUTPUT_DIR}"
export QWEN_ENABLE_THINKING=0
export VLLM_REASONING_PARSER=qwen3
export VLLM_GDN_PREFILL_BACKEND=triton
export QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-451584}"
export VLLM_QWEN_MAX_MODEL_LEN="${VLLM_QWEN_MAX_MODEL_LEN:-49152}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"

PIVOT_DEV="${ROOT}/runs/egolongqa/qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_dev_2026-07-29"
UNIFORM_DEV="${ROOT}/runs/egolongqa/qwen35_9b_vllm_uniform64_px451584_dev_2026-07-29"
QWEN3_VERIFIER_DEV="${ROOT}/runs/egolongqa/qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_dev_2026-07-12"
FALLBACK_FULL="${ROOT}/runs/egolongqa/qwen35_9b_vllm_rotation_avg_pivot_full_2026-07-31"

mkdir -p "${OUTPUT_DIR}" "${ARCHIVE_DIR}" "${ROOT}/slurm_logs"
cd "${STARTER_KIT_DIR}"
USER_SITE="$(python -c 'import site; print(site.USER_SITE)')"
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi
python -c "import psutil, vllm" >/dev/null
python run_generate_longqa_pairwise_tournament.py \
    --video-folder ../egolongqa/val \
    --subset-file "${SUBSET_FILE}" \
    --candidate-predictions "${PIVOT_DEV}/predictions.jsonl" \
    --candidate-predictions "${UNIFORM_DEV}/predictions.jsonl" \
    --candidate-predictions "${QWEN3_VERIFIER_DEV}/predictions.jsonl" \
    --fallback-predictions "${FALLBACK_FULL}/predictions.jsonl" \
    --proofpack "${PIVOT_DEV}/proofpack.jsonl" \
    --llm-model Qwen/Qwen3.5-9B \
    --max-frames 64 \
    --output "${OUTPUT_DIR}/predictions.jsonl" \
    --eval-output "${OUTPUT_DIR}/results.json" \
    "$@"

python "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUTPUT_DIR}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id "${RUN_NAME}" \
    --output "${OUTPUT_DIR}/diagnostics.json"
cp "${OUTPUT_DIR}/predictions.jsonl" "${ARCHIVE_DIR}/predictions.jsonl"
cp "${OUTPUT_DIR}/results.json" "${ARCHIVE_DIR}/results.json"
cp "${OUTPUT_DIR}/results_summary.json" "${ARCHIVE_DIR}/results_summary.json"
cp "${OUTPUT_DIR}/diagnostics.json" "${ARCHIVE_DIR}/diagnostics.json"
LATEST_VLLM_LOG="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -n "${LATEST_VLLM_LOG}" ]]; then
    cp "${LATEST_VLLM_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/qwen35_pairwise_tournament_dev_${SLURM_JOB_ID}.out" "${ARCHIVE_DIR}/slurm.out" || true
    cp "${ROOT}/slurm_logs/qwen35_pairwise_tournament_dev_${SLURM_JOB_ID}.err" "${ARCHIVE_DIR}/slurm.err" || true
fi
