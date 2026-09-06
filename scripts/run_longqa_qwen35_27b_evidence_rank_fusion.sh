#!/bin/bash

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER=${ROOT}/data/wearable-ai/starter_kit
ANNOTATIONS=${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl
VIDEO_FOLDER=${ROOT}/data/wearable-ai/egolongqa/val
ENDPOINT=${ENDPOINT:-${ROOT}/runs/egolongqa/qwen35_27b_vllm_uniform64_endpoint_px451584_full_2026-08-07/predictions.jsonl}
OPTION=${OPTION:-${ROOT}/runs/egolongqa/qwen35_27b_option_quota_globalfix_full_2026-08-07/predictions.jsonl}
REFERENCE=${REFERENCE:-${ROOT}/runs/egolongqa/qwen35_27b_endpoint_optionquota_temporal_route_full_2026-08-07/predictions.jsonl}
LLM_MODEL=${LLM_MODEL:-Qwen/Qwen3.5-27B}
: "${RUN_NAME:?RUN_NAME must be set}"
: "${SUBSET_FILE:?SUBSET_FILE must be set}"
: "${OPTION_PROOFPACK:?OPTION_PROOFPACK must be set}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX must be set}"

OUTPUT=${STARTER}/output/egolongqa/${RUN_NAME}
ARCHIVE=${ROOT}/runs/egolongqa/${RUN_NAME}
mkdir -p "${ROOT}/slurm_logs" "${OUTPUT}" "${ARCHIVE}"

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai
SCRATCH_CACHE_ROOT=${SCRATCH_CACHE_ROOT:-/scratch/inf0/user/agaur/wai-26/cache}
export HF_HOME=${HF_HOME:-${SCRATCH_CACHE_ROOT}/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export QWEN_MIN_PIXELS=784
export QWEN_MAX_PIXELS=451584
export VLLM_QWEN_MAX_MODEL_LEN=49152
export VLLM_GPU_MEMORY_UTILIZATION=0.95
export VLLM_MAX_LOGPROBS=100
export QWEN_ENABLE_THINKING=0
export VLLM_GDN_PREFILL_BACKEND=triton
export VLLM_LOG_DIR=${OUTPUT}
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
[[ -d "${USER_SITE}" ]] && export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
python -c 'import psutil, vllm' >/dev/null

cmd=(
    python "${STARTER}/run_score_longqa_evidence_rank_fusion.py"
    --input "${ANNOTATIONS}"
    --video-folder "${VIDEO_FOLDER}"
    --subset-file "${SUBSET_FILE}"
    --baseline-predictions "${ENDPOINT}"
    --challenger-predictions "${OPTION}"
    --option-proofpack "${OPTION_PROOFPACK}"
    --output "${OUTPUT}/features.jsonl"
    --llm-model "${LLM_MODEL}"
    --concurrency 1
)
if [[ "$#" -gt 0 ]]; then cmd+=("$@"); fi
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then exit 0; fi
"${cmd[@]}"

python "${ROOT}/scripts/evaluate_longqa_evidence_rank_fusion.py" \
    --annotations "${ANNOTATIONS}" \
    --subset-file "${SUBSET_FILE}" \
    --features "${OUTPUT}/features.jsonl" \
    --reference-predictions "${REFERENCE}" \
    --output-dir "${OUTPUT}"

cp "${OUTPUT}"/*.json "${ARCHIVE}/"
cp "${OUTPUT}"/*.jsonl "${ARCHIVE}/"
LATEST_LOG=$(find "${OUTPUT}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ARCHIVE}/vllm_server.log"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ARCHIVE}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ARCHIVE}/slurm.err" || true
fi
echo "Archive: ${ARCHIVE}"
