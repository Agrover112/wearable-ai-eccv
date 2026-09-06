#!/bin/bash

set -euo pipefail

ROOT=/CT/NDF/work/waw-26
STARTER=${ROOT}/data/wearable-ai/starter_kit
: "${RERANK_MODE:?RERANK_MODE must be balanced or temporal}"
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
GROUNDING_CACHE_DIR=${GROUNDING_CACHE_DIR:-/scratch/inf0/user/agaur/wai-26/data/wearable-ai/grounder_features}
SELECTION_DIR=${ROOT}/runs/egolongqa/${SELECTION_NAME}
ANSWER_DIR=${ROOT}/runs/egolongqa/${ANSWER_NAME}

mkdir -p \
    "${ROOT}/slurm_logs" \
    "${SELECTION_DIR}" \
    "${ANSWER_DIR}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${GROUNDING_CACHE_DIR}"

selection_cmd=(
    python "${STARTER}/run_generate_longqa_qwen_rerank.py"
    --input "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    --video-folder "${ROOT}/data/wearable-ai/egolongqa/val"
    --subset-file "${SUBSET_FILE}"
    --output "${SELECTION_DIR}/proofpack.jsonl"
    --mode "${RERANK_MODE}"
    --candidate-frames "${CANDIDATE_FRAMES:-128}"
    --shortlist-windows "${SHORTLIST_WINDOWS:-24}"
    --final-windows "${FINAL_WINDOWS:-8}"
    --window-radius "${WINDOW_RADIUS:-1}"
    --global-frames "${GLOBAL_FRAMES:-40}"
    --final-frames "${FINAL_MAX_FRAMES:-64}"
    --contrast-weight "${CONTRAST_WEIGHT:-0.5}"
    --nms-seconds "${TEMPORAL_NMS_SECONDS:-8}"
    --grounder-cache-dir "${GROUNDING_CACHE_DIR}"
    --grounder-batch-size "${GROUNDER_BATCH_SIZE:-32}"
    --reranker-model "${RERANKER_MODEL:-Qwen/Qwen3-VL-Reranker-2B}"
    --reranker-revision "${RERANKER_REVISION:-93eac850736c677b682c67fc0302b03e552a7b16}"
    --reranker-batch-size "${RERANKER_BATCH_SIZE:-4}"
    --reranker-max-pixels "${RERANKER_MAX_PIXELS:-100352}"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    selection_cmd+=(--max-samples "${MAX_SAMPLES}")
fi

echo "============================================================"
echo "Qwen reranker selection: ${SELECTION_NAME}"
echo "Mode: ${RERANK_MODE}; subset: ${SUBSET_FILE}; max samples: ${MAX_SAMPLES:-all}"
echo "Command: ${selection_cmd[*]}"
echo "============================================================"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Dry run: selection and Qwen answer stages were not executed."
    exit 0
fi
"${selection_cmd[@]}"

export STRATEGY=qwen_window_rerank
export RETRIEVAL_QUERY_MODE=token_safe_balanced
export RUN_STEM=${ANSWER_NAME}
export RUN_NAME=${ANSWER_NAME}
export FIXED_PROOFPACK_INPUT=${SELECTION_DIR}/proofpack.jsonl
export FINAL_MAX_FRAMES=${FINAL_MAX_FRAMES:-64}
export QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-451584}
export LLM_MODEL=${LLM_MODEL:-Qwen/Qwen3.5-27B}
export QWEN_ENABLE_THINKING=0
export VLLM_GDN_PREFILL_BACKEND=triton
export VLLM_QWEN_MAX_MODEL_LEN=${VLLM_QWEN_MAX_MODEL_LEN:-49152}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.95}
export CONCURRENCY=1

bash "${ROOT}/scripts/run_longqa_proofpack_dev.sh"

if [[ "${HARD_DIAGNOSTIC:-0}" == "1" ]]; then
    diagnostic_cmd=(
        python "${ROOT}/scripts/evaluate_longqa_hard_iteration.py"
        --subset "${SUBSET_FILE}"
        --predictions "${ANSWER_DIR}/predictions.jsonl"
        --output "${ANSWER_DIR}/hard_diagnostics.json"
    )
    if [[ -n "${MAX_SAMPLES:-}" ]]; then
        diagnostic_cmd+=(--allow-partial)
    fi
    "${diagnostic_cmd[@]}"
fi

echo "Selection archive: ${SELECTION_DIR}"
echo "Answer archive: ${ANSWER_DIR}"
