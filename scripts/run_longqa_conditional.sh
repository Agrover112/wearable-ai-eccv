#!/bin/bash

# Shared execution and archival body for targeted disagreement interventions.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER_KIT=${ROOT}/data/wearable-ai/starter_kit
RUN_DATE=${RUN_DATE:-$(date +%F)}
RUN_START=$(date +%s)

: "${CONDITIONAL_MODE:?CONDITIONAL_MODE is required}"
: "${RUN_STEM:?RUN_STEM is required}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX is required}"

PRIMARY_RUN=${PRIMARY_RUN:-qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-29}
SECONDARY_RUN=${SECONDARY_RUN:-qwen35_9b_vllm_uniform64_px451584_full_2026-07-30}
TERTIARY_RUN=${TERTIARY_RUN:-qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-12}
DETECTION_RUN=${DETECTION_RUN:-qwen3_vl_8b_vllm_object_crop_pairs_f56_c8_px451584_dev_2026-07-26}
PRIMARY_DIR=${ROOT}/runs/egolongqa/${PRIMARY_RUN}
SECONDARY_DIR=${ROOT}/runs/egolongqa/${SECONDARY_RUN}
TERTIARY_DIR=${ROOT}/runs/egolongqa/${TERTIARY_RUN}
DETECTION_DIR=${ROOT}/runs/egolongqa/${DETECTION_RUN}

for file in \
    "${PRIMARY_DIR}/predictions.jsonl" \
    "${PRIMARY_DIR}/proofpack.jsonl" \
    "${SECONDARY_DIR}/predictions.jsonl" \
    "${TERTIARY_DIR}/predictions.jsonl"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: required conditional input is missing: ${file}" >&2
        exit 2
    fi
done
if [[ "${CONDITIONAL_MODE}" == "delta_crops" && ! -s "${DETECTION_DIR}/detections.jsonl" ]]; then
    echo "ERROR: required detection cache is missing: ${DETECTION_DIR}/detections.jsonl" >&2
    exit 2
fi

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai

SCRATCH_CACHE_ROOT=${SCRATCH_CACHE_ROOT:-/scratch/inf0/user/agaur/wai-26/cache}
export HF_HOME=${HF_HOME:-${SCRATCH_CACHE_ROOT}/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export TORCH_HOME=${TORCH_HOME:-${SCRATCH_CACHE_ROOT}/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${SCRATCH_CACHE_ROOT}/xdg}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${SCRATCH_CACHE_ROOT}/triton}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${SCRATCH_CACHE_ROOT}/cuda}
export QWEN_MIN_PIXELS=${QWEN_MIN_PIXELS:-784}
export QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-451584}
export VLLM_QWEN_MAX_MODEL_LEN=${VLLM_QWEN_MAX_MODEL_LEN:-49152}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.92}
export VLLM_MAX_LOGPROBS=${VLLM_MAX_LOGPROBS:-100}

RUN_NAME=${RUN_NAME:-${RUN_STEM}_${RUN_DATE}}
OUTPUT_DIR=output/egolongqa/${RUN_NAME}
ARCHIVE_DIR=${ROOT}/runs/egolongqa/${RUN_NAME}
SCORE_CACHE_DIR=${SCORE_CACHE_DIR:-/scratch/inf0/user/agaur/wai-26/data/wearable-ai/conditional_uncertainty_cache}
TCOT_CACHE_DIR=${TCOT_CACHE_DIR:-/scratch/inf0/user/agaur/wai-26/data/wearable-ai/conditional_tcot_cache}
export VLLM_LOG_DIR=${STARTER_KIT}/${OUTPUT_DIR}

mkdir -p \
    "${ROOT}/slurm_logs" \
    "${STARTER_KIT}/${OUTPUT_DIR}" \
    "${ARCHIVE_DIR}" \
    "${SCORE_CACHE_DIR}" \
    "${TCOT_CACHE_DIR}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" \
    "${TRITON_CACHE_DIR}" \
    "${CUDA_CACHE_PATH}"

cd "${STARTER_KIT}"
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi

cmd=(
    python run_generate_longqa_conditional.py
    --video-folder ../egolongqa/val
    --mode "${CONDITIONAL_MODE}"
    --disagreement-scope "${DISAGREEMENT_SCOPE:-all_different}"
    --primary-predictions "${PRIMARY_DIR}/predictions.jsonl"
    --secondary-predictions "${SECONDARY_DIR}/predictions.jsonl"
    --tertiary-predictions "${TERTIARY_DIR}/predictions.jsonl"
    --proofpack "${PRIMARY_DIR}/proofpack.jsonl"
    --proofpack-reference "${PRIMARY_DIR}/predictions.jsonl"
    --max-frames "${FINAL_MAX_FRAMES:-64}"
    --proofpack-quota "${PROOFPACK_QUOTA:-32}"
    --candidate-frames "${CANDIDATE_FRAMES:-128}"
    --segments "${SEGMENTS:-16}"
    --uncertainty-per-segment "${UNCERTAINTY_PER_SEGMENT:-4}"
    --scoring-max-pixels "${SCORING_MAX_PIXELS:-50176}"
    --top-logprobs "${TOP_LOGPROBS:-100}"
    --tcot-segments "${TCOT_SEGMENTS:-4}"
    --max-selected-per-segment "${MAX_SELECTED_PER_SEGMENT:-6}"
    --selector-max-pixels "${SELECTOR_MAX_PIXELS:-50176}"
    --neighborhood-radius "${NEIGHBORHOOD_RADIUS:-1}"
    --selected-quota "${SELECTED_QUOTA:-48}"
    --uniform-quota "${UNIFORM_QUOTA:-16}"
    --crop-candidates "${CROP_CANDIDATES:-24}"
    --detail-count "${DETAIL_COUNT:-8}"
    --min-entropy-reduction "${MIN_ENTROPY_REDUCTION:-0.05}"
    --llm-model "${LLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
    --concurrency "${CONCURRENCY:-1}"
    --score-cache-dir "${SCORE_CACHE_DIR}"
    --tcot-cache-dir "${TCOT_CACHE_DIR}"
    --output "${OUTPUT_DIR}/predictions.jsonl"
    --eval-output "${OUTPUT_DIR}/results.json"
    --audit-output "${OUTPUT_DIR}/audit.jsonl"
)
if [[ "${CONDITIONAL_MODE}" == "delta_crops" ]]; then
    cmd+=(--detections-input "${DETECTION_DIR}/detections.jsonl")
fi
if [[ "${FULL_DATASET:-0}" != "1" ]]; then
    cmd+=(--subset-file "${ROOT}/configs/egolongqa_dev140_seed20260709.json")
fi
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi

echo "============================================================"
echo "Run: ${RUN_NAME}"
echo "Mode: ${CONDITIONAL_MODE}"
echo "Scope: ${DISAGREEMENT_SCOPE:-all_different}"
echo "Primary: ${PRIMARY_RUN}"
echo "Secondary: ${SECONDARY_RUN}"
echo "Tertiary: ${TERTIARY_RUN}"
echo "Command: ${cmd[*]}"
echo "============================================================"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
"${cmd[@]}"

python "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUTPUT_DIR}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id "${RUN_NAME}" \
    --output "${OUTPUT_DIR}/diagnostics.json"

for file in predictions.jsonl audit.jsonl results.json results_summary.json diagnostics.json; do
    cp "${OUTPUT_DIR}/${file}" "${ARCHIVE_DIR}/${file}"
done
LATEST_LOG=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
echo "Runtime seconds: $(($(date +%s) - RUN_START))"
echo "Archive: ${ARCHIVE_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ARCHIVE_DIR}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ARCHIVE_DIR}/slurm.err" || true
fi
