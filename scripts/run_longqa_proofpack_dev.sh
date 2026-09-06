#!/bin/bash

# Shared execution body for proof-pack Slurm launchers.

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="/CT/NDF/work/waw-26"
STARTER_KIT_DIR="${STARTER_KIT_DIR:-${ROOT}/data/wearable-ai/starter_kit}"
RUN_START_EPOCH="$(date +%s)"
RUN_DATE="$(date +%F)"
SUBSET_FILE="${SUBSET_FILE:-${ROOT}/configs/egolongqa_dev140_seed20260709.json}"
if [[ "${SUBSET_FILE}" != /* ]]; then
    SUBSET_FILE="${ROOT}/${SUBSET_FILE}"
fi

: "${STRATEGY:?STRATEGY must be set by the launcher}"
: "${RUN_STEM:?RUN_STEM must be set by the launcher}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX must be set by the launcher}"

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai

SCRATCH_CACHE_ROOT="${SCRATCH_CACHE_ROOT:-/scratch/inf0/user/agaur/wai-26/cache}"
export HF_HOME="${HF_HOME:-${SCRATCH_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_CACHE_ROOT}/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_CACHE_ROOT}/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${SCRATCH_CACHE_ROOT}/cuda}"
export QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-784}"
export QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-200704}"
export VLLM_QWEN_MAX_MODEL_LEN="${VLLM_QWEN_MAX_MODEL_LEN:-32768}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"

GROUNDING_CACHE_DIR="${GROUNDING_CACHE_DIR:-/scratch/inf0/user/agaur/wai-26/data/wearable-ai/grounder_features}"
RUN_NAME="${RUN_NAME:-${RUN_STEM}_${RUN_DATE}}"
OUTPUT_DIR="${OUTPUT_DIR:-output/egolongqa/${RUN_NAME}}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${ROOT}/runs/egolongqa/${RUN_NAME}}"
export VLLM_LOG_DIR="${STARTER_KIT_DIR}/${OUTPUT_DIR}"

mkdir -p \
    "${ROOT}/slurm_logs" \
    "${STARTER_KIT_DIR}/${OUTPUT_DIR}" \
    "${ARCHIVE_DIR}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" \
    "${TRITON_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}" \
    "${GROUNDING_CACHE_DIR}"

cd "${STARTER_KIT_DIR}"
USER_SITE="$(python -c 'import site; print(site.USER_SITE)')"
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi

cmd=(
    python run_generate_longqa_proofpack.py
    --video-folder ../egolongqa/val
    --strategy "${STRATEGY}"
    --candidate-frames "${CANDIDATE_FRAMES:-128}"
    --event-centers "${EVENT_CENTERS:-8}"
    --centers-per-option "${CENTERS_PER_OPTION:-4}"
    --pivot-centers "${PIVOT_CENTERS:-2}"
    --target-centers "${TARGET_CENTERS:-8}"
    --eventlet-radius "${EVENTLET_RADIUS:-1}"
    --anchor-k "${ANCHOR_K:-32}"
    --boundary-k "${BOUNDARY_K:-8}"
    --bridge-k "${BRIDGE_K:-8}"
    --global-uniform-frames "${GLOBAL_UNIFORM_FRAMES:-64}"
    --multi-event-centers "${MULTI_EVENT_CENTERS:-2}"
    --max-event-queries "${MAX_EVENT_QUERIES:-3}"
    --qca-segments "${QCA_SEGMENTS:-16}"
    --qca-alpha "${QCA_ALPHA:-0.5}"
    --qca-beta "${QCA_BETA:-0.5}"
    --qca-temperature "${QCA_TEMPERATURE:-0.5}"
    --qca-relevance-threshold "${QCA_RELEVANCE_THRESHOLD:-0.7}"
    --selection-seed "${SELECTION_SEED:-42}"
    --adaq-var-scale "${ADAQ_VAR_SCALE:-0.5}"
    --adaq-p-threshold "${ADAQ_P_THRESHOLD:-0.95}"
    --focus-arms "${FOCUS_ARMS:-16}"
    --focus-zoom-ratio "${FOCUS_ZOOM_RATIO:-0.25}"
    --focus-extra-samples "${FOCUS_EXTRA_SAMPLES:-2}"
    --focus-top-ratio "${FOCUS_TOP_RATIO:-0.2}"
    --focus-temperature "${FOCUS_TEMPERATURE:-0.06}"
    --mixed-high-frames "${MIXED_HIGH_FRAMES:-4}"
    --mixed-medium-frames "${MIXED_MEDIUM_FRAMES:-8}"
    --mixed-low-frames "${MIXED_LOW_FRAMES:-32}"
    --mixed-high-pixels "${MIXED_HIGH_PIXELS:-451584}"
    --mixed-medium-pixels "${MIXED_MEDIUM_PIXELS:-200704}"
    --mixed-low-pixels "${MIXED_LOW_PIXELS:-50176}"
    --qframe-temperature "${QFRAME_TEMPERATURE:-0.1}"
    --mmr-retrieval-k "${MMR_RETRIEVAL_K:-16}"
    --mmr-lambda "${MMR_LAMBDA:-0.7}"
    --rrf-constant "${RRF_CONSTANT:-60}"
    --final-max-frames "${FINAL_MAX_FRAMES:-64}"
    --temporal-nms-seconds "${TEMPORAL_NMS_SECONDS:-10}"
    --fill-mode "${FILL_MODE:-semantic_boundary}"
    --grounder-model "${GROUNDER_MODEL:-google/siglip2-so400m-patch14-384}"
    --grounder-device cuda
    --grounder-batch-size "${GROUNDER_BATCH_SIZE:-16}"
    --grounder-dtype bfloat16
    --grounder-max-pixels "${GROUNDER_MAX_PIXELS:-200704}"
    --grounder-cache-dir "${GROUNDING_CACHE_DIR}"
    --retrieval-query-mode "${RETRIEVAL_QUERY_MODE:-legacy_joint}"
    --prompt-variant "${PROMPT_VARIANT:-baseline}"
    --model-type qwen
    --llm-model "${LLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
    --backend vllm
    --tp "${TP:-1}"
    --concurrency "${CONCURRENCY:-1}"
    --output "${OUTPUT_DIR}/predictions.jsonl"
    --eval-output "${OUTPUT_DIR}/results.json"
    --grounding-output "${OUTPUT_DIR}/proofpack.jsonl"
)
if [[ -n "${FIXED_PROOFPACK_INPUT:-}" ]]; then
    cmd+=(--fixed-grounding-input "${FIXED_PROOFPACK_INPUT}")
fi
if [[ "${STRUCTURED_EVIDENCE:-0}" == "1" ]]; then
    cmd+=(--structured-evidence)
fi
if [[ "${SELECTION_ONLY:-0}" == "1" ]]; then
    cmd+=(--selection-only)
fi
if [[ "${FULL_DATASET:-0}" != "1" ]]; then
    cmd+=(--subset-file "${SUBSET_FILE}")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    cmd+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi

echo "============================================================"
echo "Run: ${RUN_NAME}"
echo "Strategy: ${STRATEGY}"
echo "Structured evidence: ${STRUCTURED_EVIDENCE:-0}"
echo "Subset: ${SUBSET_FILE}"
echo "Grounder cache: ${GROUNDING_CACHE_DIR}"
echo "Command: ${cmd[*]}"
echo "============================================================"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
"${cmd[@]}"

if [[ "${SELECTION_ONLY:-0}" == "1" ]]; then
    cp "${OUTPUT_DIR}/proofpack.jsonl" "${ARCHIVE_DIR}/proofpack.jsonl"
    echo "Runtime seconds: $(($(date +%s) - RUN_START_EPOCH))"
    echo "Selection-only archive: ${ARCHIVE_DIR}"
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ARCHIVE_DIR}/slurm.out" || true
        cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ARCHIVE_DIR}/slurm.err" || true
    fi
    exit 0
fi

python "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUTPUT_DIR}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id "${RUN_NAME}" \
    --output "${OUTPUT_DIR}/diagnostics.json"

for file in predictions.jsonl proofpack.jsonl results.json results_summary.json diagnostics.json; do
    cp "${OUTPUT_DIR}/${file}" "${ARCHIVE_DIR}/${file}"
done
LATEST_VLLM_LOG="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -n "${LATEST_VLLM_LOG}" ]]; then
    cp "${LATEST_VLLM_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
fi
echo "Runtime seconds: $(($(date +%s) - RUN_START_EPOCH))"
echo "Archive: ${ARCHIVE_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ARCHIVE_DIR}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ARCHIVE_DIR}/slurm.err" || true
fi
