#!/bin/bash

# Compare the final majority against the direct 27B answer on their 25 disagreements.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER=${ROOT}/data/wearable-ai/starter_kit
: "${RUN_NAME:?RUN_NAME is required}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX is required}"

ANNOTATIONS=${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl
BASELINE=${ROOT}/runs/egolongqa/qwen35_q27_thinking_endpoint_majority_full_2026-08-05/predictions.jsonl
CHALLENGER=${ROOT}/runs/egolongqa/qwen35_27b_multievidence_candidate_blind_fixed_full_2026-08-05/predictions.jsonl
EVIDENCE=${CHALLENGER}
OUTPUT_DIR=${STARTER}/output/egolongqa/${RUN_NAME}
ARCHIVE_DIR=${ROOT}/runs/egolongqa/${RUN_NAME}
for file in "${ANNOTATIONS}" "${BASELINE}" "${CHALLENGER}" "${EVIDENCE}"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: missing pairwise input: ${file}" >&2
        exit 2
    fi
done

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
export VLLM_REASONING_PARSER=qwen3
export VLLM_GDN_PREFILL_BACKEND=triton
export VLLM_LOG_DIR=${OUTPUT_DIR}
mkdir -p "${ROOT}/slurm_logs" "${OUTPUT_DIR}" "${ARCHIVE_DIR}"
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi
python -c 'import psutil, vllm' >/dev/null

subset_args=()
if [[ -n "${SUBSET_FILE:-}" ]]; then
    subset_args+=(--subset-file "${SUBSET_FILE}")
fi
cmd=(
    python "${STARTER}/run_score_longqa_full_evidence_pairwise.py"
    --input "${ANNOTATIONS}"
    --video-folder "${ROOT}/data/wearable-ai/egolongqa/val"
    --baseline-predictions "${BASELINE}"
    --challenger-predictions "${CHALLENGER}"
    --evidence-predictions "${EVIDENCE}"
    --llm-model Qwen/Qwen3.5-27B
    --tp 1 --concurrency 1
    --output "${OUTPUT_DIR}/features.jsonl"
    "${subset_args[@]}"
)
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
"${cmd[@]}"

for policy in conservative consensus; do
    python "${ROOT}/scripts/evaluate_longqa_full_evidence_pairwise.py" \
        --annotations "${ANNOTATIONS}" \
        "${subset_args[@]}" \
        --features "${OUTPUT_DIR}/features.jsonl" \
        --policy "${policy}" \
        --output "${OUTPUT_DIR}/${policy}_predictions.jsonl" \
        --summary-output "${OUTPUT_DIR}/${policy}_summary.json"
done

cp "${OUTPUT_DIR}"/*.json "${ARCHIVE_DIR}/"
cp "${OUTPUT_DIR}"/*.jsonl "${ARCHIVE_DIR}/"
LATEST_LOG=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ARCHIVE_DIR}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ARCHIVE_DIR}/slurm.err" || true
fi
echo "Archive: ${ARCHIVE_DIR}"
