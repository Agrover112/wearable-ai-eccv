#!/bin/bash

# Score full candidate answer text on dev140 fixed-ensemble disagreements.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER_KIT=${ROOT}/data/wearable-ai/starter_kit
PRIMARY_RUN=qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-29
SECONDARY_RUN=qwen35_9b_vllm_uniform64_px451584_full_2026-07-30
TERTIARY_RUN=qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-12
MAJORITY_RUN=offline_majority_qwen35_pivot_qwen35_uniform_qwen3_verifier_2026-07-30
RUN_NAME=qwen35_9b_candidate_text_likelihood_disagreements_dev_2026-07-31
OUTPUT_DIR=output/egolongqa/${RUN_NAME}
ARCHIVE_DIR=${ROOT}/runs/egolongqa/${RUN_NAME}
DEV140=${ROOT}/configs/egolongqa_dev140_seed20260709.json
ANNOTATIONS=${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl
RUN_START=$(date +%s)

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai
SCRATCH_CACHE_ROOT=${SCRATCH_CACHE_ROOT:-/scratch/inf0/user/agaur/wai-26/cache}
export HF_HOME=${HF_HOME:-${SCRATCH_CACHE_ROOT}/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export QWEN_MIN_PIXELS=${QWEN_MIN_PIXELS:-784}
# Prompt-logprob scoring materializes logits for every prompt token. Keep the
# 64-frame timeline, but use the established 448px budget and reserve headroom
# beyond vLLM's KV cache for the full-vocabulary log-softmax.
export QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-200704}
export VLLM_QWEN_MAX_MODEL_LEN=${VLLM_QWEN_MAX_MODEL_LEN:-32768}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.80}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_LOG_DIR=${STARTER_KIT}/${OUTPUT_DIR}
mkdir -p "${ROOT}/slurm_logs" "${STARTER_KIT}/${OUTPUT_DIR}" "${ARCHIVE_DIR}"

cd "${STARTER_KIT}"
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi
python -c 'import psutil'

cmd=(
    python run_score_longqa_candidate_text_disagreements.py
    --video-folder ../egolongqa/val \
    --subset-file "${DEV140}" \
    --primary-predictions "${ROOT}/runs/egolongqa/${PRIMARY_RUN}/predictions.jsonl" \
    --secondary-predictions "${ROOT}/runs/egolongqa/${SECONDARY_RUN}/predictions.jsonl" \
    --tertiary-predictions "${ROOT}/runs/egolongqa/${TERTIARY_RUN}/predictions.jsonl" \
    --proofpack "${ROOT}/runs/egolongqa/${PRIMARY_RUN}/proofpack.jsonl" \
    --proofpack-reference "${ROOT}/runs/egolongqa/${PRIMARY_RUN}/predictions.jsonl" \
    --max-frames 64 \
    --proofpack-quota 32 \
    --llm-model Qwen/Qwen3.5-9B \
    --concurrency 1 \
    --output "${OUTPUT_DIR}/scores.jsonl"
)
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
"${cmd[@]}"

python "${ROOT}/scripts/evaluate_longqa_candidate_text_scores.py" \
    --scores "${OUTPUT_DIR}/scores.jsonl" \
    --annotations "${ANNOTATIONS}" \
    --subset-file "${DEV140}" \
    --majority-predictions "${ROOT}/runs/egolongqa/${MAJORITY_RUN}/predictions.jsonl" \
    --output "${OUTPUT_DIR}/summary.json"

cp "${OUTPUT_DIR}/scores.jsonl" "${ARCHIVE_DIR}/scores.jsonl"
cp "${OUTPUT_DIR}/summary.json" "${ARCHIVE_DIR}/summary.json"
LATEST_LOG=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/qwen35_candidate_text_likelihood_dev_${SLURM_JOB_ID}.out" "${ARCHIVE_DIR}/slurm.out" || true
    cp "${ROOT}/slurm_logs/qwen35_candidate_text_likelihood_dev_${SLURM_JOB_ID}.err" "${ARCHIVE_DIR}/slurm.err" || true
fi
echo "Runtime seconds: $(($(date +%s) - RUN_START))"
echo "Archive: ${ARCHIVE_DIR}"
