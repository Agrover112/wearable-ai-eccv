#!/bin/bash

# Score the three fixed ensemble views, then run nested grouped CV on CPU.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER_KIT=${ROOT}/data/wearable-ai/starter_kit
RUN_DATE=${RUN_DATE:-$(date +%F)}
RUN_START=$(date +%s)
: "${RUN_STEM:?RUN_STEM is required}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX is required}"

PRIMARY_RUN=${PRIMARY_RUN:-qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-29}
SECONDARY_RUN=${SECONDARY_RUN:-qwen35_9b_vllm_uniform64_px451584_full_2026-07-30}
TERTIARY_RUN=${TERTIARY_RUN:-qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-12}
MAJORITY_RUN=${MAJORITY_RUN:-offline_majority_qwen35_pivot_qwen35_uniform_qwen3_verifier_2026-07-30}
PRIMARY_DIR=${ROOT}/runs/egolongqa/${PRIMARY_RUN}
SECONDARY_DIR=${ROOT}/runs/egolongqa/${SECONDARY_RUN}
TERTIARY_DIR=${ROOT}/runs/egolongqa/${TERTIARY_RUN}
MAJORITY_DIR=${ROOT}/runs/egolongqa/${MAJORITY_RUN}
for file in \
    "${PRIMARY_DIR}/predictions.jsonl" \
    "${PRIMARY_DIR}/proofpack.jsonl" \
    "${SECONDARY_DIR}/predictions.jsonl" \
    "${TERTIARY_DIR}/predictions.jsonl" \
    "${MAJORITY_DIR}/predictions.jsonl"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: required router input is missing: ${file}" >&2
        exit 2
    fi
done

source /CT/NDF/work/miniforge3/etc/profile.d/conda.sh
conda activate wearable-ai
SCRATCH_CACHE_ROOT=${SCRATCH_CACHE_ROOT:-/scratch/inf0/user/agaur/wai-26/cache}
export HF_HOME=${HF_HOME:-${SCRATCH_CACHE_ROOT}/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export QWEN_MIN_PIXELS=${QWEN_MIN_PIXELS:-784}
export QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-451584}
export VLLM_QWEN_MAX_MODEL_LEN=${VLLM_QWEN_MAX_MODEL_LEN:-49152}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.92}

RUN_NAME=${RUN_NAME:-${RUN_STEM}_${RUN_DATE}}
OUTPUT_DIR=output/egolongqa/${RUN_NAME}
ARCHIVE_DIR=${ROOT}/runs/egolongqa/${RUN_NAME}
export VLLM_LOG_DIR=${STARTER_KIT}/${OUTPUT_DIR}
mkdir -p "${ROOT}/slurm_logs" "${STARTER_KIT}/${OUTPUT_DIR}" "${ARCHIVE_DIR}"

cd "${STARTER_KIT}"
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi
python -c 'import psutil' || {
    echo "ERROR: vLLM dependency psutil is unavailable after PYTHONPATH setup" >&2
    exit 2
}
cmd=(
    python run_score_longqa_disagreements.py
    --video-folder ../egolongqa/val
    --primary-predictions "${PRIMARY_DIR}/predictions.jsonl"
    --secondary-predictions "${SECONDARY_DIR}/predictions.jsonl"
    --tertiary-predictions "${TERTIARY_DIR}/predictions.jsonl"
    --proofpack "${PRIMARY_DIR}/proofpack.jsonl"
    --proofpack-reference "${PRIMARY_DIR}/predictions.jsonl"
    --max-frames 64
    --proofpack-quota 32
    --llm-model "${LLM_MODEL:-Qwen/Qwen3.5-9B}"
    --concurrency 1
    --output "${OUTPUT_DIR}/features.jsonl"
)
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
"${cmd[@]}"

eval_cmd=(
    python "${ROOT}/scripts/evaluate_longqa_confidence_router.py"
    --features "${OUTPUT_DIR}/features.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --majority-predictions "${MAJORITY_DIR}/predictions.jsonl" \
    --output "${OUTPUT_DIR}/router_oof.jsonl" \
    --summary-output "${OUTPUT_DIR}/router_oof_summary.json"
)
if [[ -n "${ROUTER_EVAL_SUBSET:-}" ]]; then
    eval_cmd+=(--subset-file "${ROUTER_EVAL_SUBSET}")
fi
"${eval_cmd[@]}"

archive_files=(features.jsonl router_oof.jsonl router_oof_summary.json)
if [[ "${OPTION_ROTATION_ANALYSIS:-0}" == "1" ]]; then
    python "${ROOT}/scripts/analyze_longqa_option_rotation_scores.py" \
        --features "${OUTPUT_DIR}/features.jsonl" \
        --output "${OUTPUT_DIR}/rotation_summary.json"
    archive_files+=(rotation_summary.json)
fi

for file in "${archive_files[@]}"; do
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
