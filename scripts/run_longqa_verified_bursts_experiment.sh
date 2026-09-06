#!/bin/bash

set -euo pipefail

ROOT=/CT/NDF/work/waw-26
STARTER=${ROOT}/data/wearable-ai/starter_kit
: "${QUERY_MODE:?QUERY_MODE must be events_only or events_options}"
: "${SUBSET_FILE:?SUBSET_FILE is required}"
: "${SELECTION_NAME:?SELECTION_NAME is required}"
: "${ANSWER_NAME:?ANSWER_NAME is required}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX is required}"

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai

export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/scratch/inf0/user/agaur/wai-26/cache/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export TORCH_HOME=${TORCH_HOME:-/scratch/inf0/user/agaur/wai-26/cache/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/scratch/inf0/user/agaur/wai-26/cache/xdg}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/scratch/inf0/user/agaur/wai-26/cache/triton}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/scratch/inf0/user/agaur/wai-26/cache/cuda}
export QWEN_MIN_PIXELS=784
export QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-451584}
export VLLM_QWEN_MAX_MODEL_LEN=${VLLM_QWEN_MAX_MODEL_LEN:-49152}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.95}
export QWEN_ENABLE_THINKING=0
export VLLM_GDN_PREFILL_BACKEND=triton

SELECTION_DIR=${ROOT}/runs/egolongqa/${SELECTION_NAME}
ANSWER_DIR=${ROOT}/runs/egolongqa/${ANSWER_NAME}
mkdir -p \
    "${ROOT}/slurm_logs" "${SELECTION_DIR}" "${ANSWER_DIR}" \
    "${HUGGINGFACE_HUB_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" \
    "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}"

selection_cmd=(
    python "${STARTER}/run_generate_longqa_verified_bursts.py"
    --input "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    --video-folder "${ROOT}/data/wearable-ai/egolongqa/val"
    --subset-file "${SUBSET_FILE}"
    --output "${SELECTION_DIR}/proofpack.jsonl"
    --query-mode "${QUERY_MODE}"
    --candidate-frames "${CANDIDATE_FRAMES:-256}"
    --coarse-windows "${COARSE_WINDOWS:-32}"
    --coarse-frames-per-window "${COARSE_FRAMES_PER_WINDOW:-4}"
    --max-event-centers "${MAX_EVENT_CENTERS:-6}"
    --fine-verification-frames "${FINE_VERIFICATION_FRAMES:-3}"
    --fine-verification-span-seconds "${FINE_VERIFICATION_SPAN_SECONDS:-2}"
    --burst-frames "${BURST_FRAMES:-5}"
    --burst-span-seconds "${BURST_SPAN_SECONDS:-4}"
    --global-frames "${GLOBAL_FRAMES:-32}"
    --final-frames "${FINAL_MAX_FRAMES:-64}"
    --option-contrast-weight "${OPTION_CONTRAST_WEIGHT:-0.5}"
    --qualification-delta "${QUALIFICATION_DELTA:-0.10}"
    --reranker-model "${RERANKER_MODEL:-Qwen/Qwen3-VL-Reranker-2B}"
    --reranker-revision "${RERANKER_REVISION:-93eac850736c677b682c67fc0302b03e552a7b16}"
    --reranker-batch-size "${RERANKER_BATCH_SIZE:-4}"
    --reranker-max-pixels "${RERANKER_MAX_PIXELS:-100352}"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    selection_cmd+=(--max-samples "${MAX_SAMPLES}")
fi

answer_cmd=(
    python "${STARTER}/run_answer_longqa_selection.py"
    --input "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    --video-folder "${ROOT}/data/wearable-ai/egolongqa/val"
    --subset-file "${SUBSET_FILE}"
    --selection "${SELECTION_DIR}/proofpack.jsonl"
    --indices-key final_indices
    --prompt-variant "${PROMPT_VARIANT:-clause_complete}"
    --llm-model "${LLM_MODEL:-Qwen/Qwen3.5-27B}"
    --max-frames "${FINAL_MAX_FRAMES:-64}"
    --concurrency 1
    --output "${ANSWER_DIR}/predictions.jsonl"
    --eval-output "${ANSWER_DIR}/results.json"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    answer_cmd+=(--max-samples "${MAX_SAMPLES}")
fi

echo "============================================================"
echo "Verified event bursts: ${ANSWER_NAME}"
echo "Mode: ${QUERY_MODE}; subset: ${SUBSET_FILE}; max samples: ${MAX_SAMPLES:-all}"
echo "Selection: ${selection_cmd[*]}"
echo "Answer: ${answer_cmd[*]}"
echo "============================================================"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

START=$(date +%s)
"${selection_cmd[@]}"
export VLLM_LOG_DIR=${ANSWER_DIR}
"${answer_cmd[@]}"
cp "${SELECTION_DIR}/proofpack.jsonl" "${ANSWER_DIR}/proofpack.jsonl"
python "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${ANSWER_DIR}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id "${ANSWER_NAME}" \
    --output "${ANSWER_DIR}/diagnostics.json"
if [[ "${HARD_DIAGNOSTIC:-0}" == "1" ]]; then
    hard_cmd=(
        python "${ROOT}/scripts/evaluate_longqa_hard_iteration.py"
        --subset "${SUBSET_FILE}"
        --predictions "${ANSWER_DIR}/predictions.jsonl"
        --output "${ANSWER_DIR}/hard_diagnostics.json"
    )
    if [[ -n "${MAX_SAMPLES:-}" ]]; then hard_cmd+=(--allow-partial); fi
    "${hard_cmd[@]}"
fi
LATEST_LOG=$(find "${ANSWER_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ANSWER_DIR}/vllm_server.log"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ANSWER_DIR}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ANSWER_DIR}/slurm.err" || true
fi
echo "Runtime seconds: $(($(date +%s) - START))"
echo "Selection archive: ${SELECTION_DIR}"
echo "Answer archive: ${ANSWER_DIR}"
