#!/bin/bash

# Shared execution body for disagreement-verifier launchers.

set -euo pipefail
ROOT="/CT/NDF/work/waw-26"
STARTER_KIT="${ROOT}/data/wearable-ai/starter_kit"
RUN_DATE="$(date +%F)"
RUN_START="$(date +%s)"
: "${PRIMARY_RUN:?PRIMARY_RUN is required}"
: "${SECONDARY_RUN:?SECONDARY_RUN is required}"
: "${RUN_STEM:?RUN_STEM is required}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX is required}"

PRIMARY_DIR="${ROOT}/runs/egolongqa/${PRIMARY_RUN}"
SECONDARY_DIR="${ROOT}/runs/egolongqa/${SECONDARY_RUN}"
for file in "${PRIMARY_DIR}/predictions.jsonl" "${PRIMARY_DIR}/proofpack.jsonl" "${SECONDARY_DIR}/predictions.jsonl"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: required verifier input is missing: ${file}" >&2
        exit 2
    fi
done

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai
SCRATCH_CACHE_ROOT="${SCRATCH_CACHE_ROOT:-/scratch/inf0/user/agaur/wai-26/cache}"
export HF_HOME="${HF_HOME:-${SCRATCH_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-784}"
export QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-451584}"
export VLLM_QWEN_MAX_MODEL_LEN="${VLLM_QWEN_MAX_MODEL_LEN:-49152}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"

RUN_NAME="${RUN_NAME:-${RUN_STEM}_${RUN_DATE}}"
OUTPUT_DIR="output/egolongqa/${RUN_NAME}"
ARCHIVE_DIR="${ROOT}/runs/egolongqa/${RUN_NAME}"
export VLLM_LOG_DIR="${STARTER_KIT}/${OUTPUT_DIR}"
mkdir -p "${ROOT}/slurm_logs" "${STARTER_KIT}/${OUTPUT_DIR}" "${ARCHIVE_DIR}"

cd "${STARTER_KIT}"
USER_SITE="$(python -c 'import site; print(site.USER_SITE)')"
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi
cmd=(
    python run_generate_longqa_verifier.py
    --video-folder ../egolongqa/val
    --primary-predictions "${PRIMARY_DIR}/predictions.jsonl"
    --secondary-predictions "${SECONDARY_DIR}/predictions.jsonl"
    --primary-proofpack "${PRIMARY_DIR}/proofpack.jsonl"
    --max-frames 64
    --proofpack-quota 32
    --llm-model "${LLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
    --backend vllm
    --tp 1
    --concurrency 1
    --blind-weight "${BLIND_WEIGHT:--0.1}"
    --output "${OUTPUT_DIR}/predictions.jsonl"
    --eval-output "${OUTPUT_DIR}/results.json"
)
if [[ -n "${VERIFY_OPERATORS:-}" ]]; then
    read -r -a verify_operators <<< "${VERIFY_OPERATORS}"
    cmd+=(--verify-operators "${verify_operators[@]}")
fi
if [[ -n "${VERIFIER_PROMPT:-}" ]]; then
    cmd+=(--verifier-prompt "${VERIFIER_PROMPT}")
fi
if [[ "${FULL_DATASET:-0}" != "1" ]]; then
    cmd+=(--subset-file "${ROOT}/configs/egolongqa_dev140_seed20260709.json")
fi
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
"${cmd[@]}"

python "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUTPUT_DIR}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id "${RUN_NAME}" \
    --output "${OUTPUT_DIR}/diagnostics.json"
python "${ROOT}/scripts/analyze_longqa_disagreements.py" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --first "${PRIMARY_DIR}/predictions.jsonl" \
    --second "${SECONDARY_DIR}/predictions.jsonl" \
    --first-label primary \
    --second-label secondary \
    --output-json "${OUTPUT_DIR}/candidate_disagreements.json" \
    --output-jsonl "${OUTPUT_DIR}/candidate_disagreements.jsonl"

for file in predictions.jsonl results.json results_summary.json diagnostics.json candidate_disagreements.json candidate_disagreements.jsonl; do
    cp "${OUTPUT_DIR}/${file}" "${ARCHIVE_DIR}/${file}"
done
LATEST_LOG="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
echo "Runtime seconds: $(($(date +%s) - RUN_START))"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ARCHIVE_DIR}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ARCHIVE_DIR}/slurm.err" || true
fi
