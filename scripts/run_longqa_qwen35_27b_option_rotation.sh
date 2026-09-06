#!/bin/bash

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER=${ROOT}/data/wearable-ai/starter_kit
PRIMARY=${ROOT}/runs/egolongqa/qwen35_27b_multievidence_candidate_blind_fixed_full_2026-08-05/predictions.jsonl
ENDPOINT=${ROOT}/runs/egolongqa/qwen35_9b_vllm_uniform64_endpoint_px451584_full_2026-08-02/predictions.jsonl
ANNOTATIONS=${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl
: "${RUN_NAME:?RUN_NAME must be set}"
: "${SUBSET_FILE:?SUBSET_FILE must be set}"
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
export QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-451584}
export VLLM_QWEN_MAX_MODEL_LEN=${VLLM_QWEN_MAX_MODEL_LEN:-49152}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.95}
export VLLM_MAX_LOGPROBS=100
export QWEN_ENABLE_THINKING=0
export VLLM_GDN_PREFILL_BACKEND=triton
export VLLM_LOG_DIR=${OUTPUT}
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
[[ -d "${USER_SITE}" ]] && export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
python -c 'import psutil, vllm' >/dev/null

cmd=(
    python "${STARTER}/run_score_longqa_option_rotation.py"
    --video-folder "${ROOT}/data/wearable-ai/egolongqa/val"
    --subset-file "${SUBSET_FILE}"
    --evidence-predictions "${PRIMARY}"
    --llm-model Qwen/Qwen3.5-27B
    --max-frames 64 --option-rotations 4 --concurrency 1
    --output "${OUTPUT}/features.jsonl"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd+=(--max-samples "${MAX_SAMPLES}"); fi
if [[ "$#" -gt 0 ]]; then cmd+=("$@"); fi
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then exit 0; fi
"${cmd[@]}"

eval_subset=${SUBSET_FILE}
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    python "${ROOT}/scripts/create_longqa_subset_prefix.py" \
        --source "${SUBSET_FILE}" --count "${MAX_SAMPLES}" \
        --output "${OUTPUT}/evaluation_subset.json"
    eval_subset=${OUTPUT}/evaluation_subset.json
fi
python "${ROOT}/scripts/evaluate_longqa_option_rotation.py" \
    --annotations "${ANNOTATIONS}" --subset-file "${eval_subset}" \
    --features "${OUTPUT}/features.jsonl" --primary "${PRIMARY}" \
    --endpoint "${ENDPOINT}" --output-dir "${OUTPUT}"

cp "${OUTPUT}"/*.json "${ARCHIVE}/"
cp "${OUTPUT}"/*.jsonl "${ARCHIVE}/"
LATEST_LOG=$(find "${OUTPUT}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ARCHIVE}/vllm_server.log"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ARCHIVE}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ARCHIVE}/slurm.err" || true
fi
echo "Archive: ${ARCHIVE}"
