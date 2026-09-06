#!/bin/bash

# Run Qwen3.5-27B only where the validated primary candidates disagree.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER=${ROOT}/data/wearable-ai/starter_kit
: "${RUN_NAME:?RUN_NAME is required}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX is required}"
: "${SUBSET_FILE:?SUBSET_FILE is required}"

PIVOT=${PIVOT_RUN_DIR:-${ROOT}/runs/egolongqa/qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-29}
UNIFORM=${UNIFORM_RUN_DIR:-${ROOT}/runs/egolongqa/qwen35_9b_vllm_uniform64_px451584_full_2026-07-30}
UNCERTAINTY=${UNCERTAINTY_RUN_DIR:-${ROOT}/runs/egolongqa/qwen35_9b_vllm_uncertainty_pivot_full_2026-08-02}
OPTION=${OPTION_RUN_DIR:-${ROOT}/runs/egolongqa/qwen35_9b_vllm_siglip2_option_quota_pivot_anc24_final64_px451584_full_2026-08-02}
ENDPOINT=${ENDPOINT_RUN_DIR:-${ROOT}/runs/egolongqa/qwen35_9b_vllm_uniform64_endpoint_px451584_full_2026-08-02}
PROOFPACK=${PROOFPACK_RUN_DIR:-${PIVOT}}
FALLBACK=${FALLBACK_RUN_DIR:-${ROOT}/runs/egolongqa/qwen35_9b_independent_five_majority_full_2026-08-03}
OUTPUT_DIR=${STARTER}/output/egolongqa/${RUN_NAME}
ARCHIVE_DIR=${ROOT}/runs/egolongqa/${RUN_NAME}
for file in \
    "${PIVOT}/predictions.jsonl" "${PROOFPACK}/predictions.jsonl" \
    "${PROOFPACK}/proofpack.jsonl" \
    "${UNIFORM}/predictions.jsonl" "${UNCERTAINTY}/predictions.jsonl" \
    "${OPTION}/predictions.jsonl" "${ENDPOINT}/predictions.jsonl" \
    "${FALLBACK}/predictions.jsonl" "${UNCERTAINTY}/selections.jsonl" \
    "${SUBSET_FILE}"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: missing 27B judge input: ${file}" >&2
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
export QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-451584}
export VLLM_QWEN_MAX_MODEL_LEN=${VLLM_QWEN_MAX_MODEL_LEN:-49152}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.95}
export QWEN_ENABLE_THINKING=${QWEN_ENABLE_THINKING:-0}
export VLLM_REASONING_PARSER=qwen3
export VLLM_GDN_PREFILL_BACKEND=triton
export VLLM_LOG_DIR=${OUTPUT_DIR}
mkdir -p "${ROOT}/slurm_logs" "${OUTPUT_DIR}" "${ARCHIVE_DIR}"
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi
python -c 'import psutil, vllm' >/dev/null

cmd=(
    python "${STARTER}/run_generate_longqa_multicandidate_judge.py"
    --video-folder "${ROOT}/data/wearable-ai/egolongqa/val"
    --subset-file "${SUBSET_FILE}"
    --candidate-predictions "${PIVOT}/predictions.jsonl"
    --candidate-predictions "${UNIFORM}/predictions.jsonl"
    --candidate-predictions "${UNCERTAINTY}/predictions.jsonl"
    --candidate-predictions "${OPTION}/predictions.jsonl"
    --candidate-predictions "${ENDPOINT}/predictions.jsonl"
    --candidate-labels q35_pivot q35_uniform uncertainty option_quota endpoint
    --fallback-predictions "${FALLBACK}/predictions.jsonl"
    --proofpack "${PROOFPACK}/proofpack.jsonl"
    --proofpack-reference "${PROOFPACK}/predictions.jsonl"
    --uncertainty-selection "${UNCERTAINTY}/selections.jsonl"
    --llm-model "${LLM_MODEL:-Qwen/Qwen3.5-27B}"
    --backend vllm --tp 1 --concurrency 1
    --max-frames "${MAX_FRAMES:-64}"
    --pivot-quota "${PIVOT_QUOTA:-24}"
    --uniform-quota "${UNIFORM_QUOTA:-24}"
    --uncertainty-quota "${UNCERTAINTY_QUOTA:-16}"
    --evidence-strategy "${EVIDENCE_STRATEGY:-mixed}"
    --sparse-global-quota "${SPARSE_GLOBAL_QUOTA:-8}"
    --sparse-neighbor-radius "${SPARSE_NEIGHBOR_RADIUS:-1}"
    --max-new-tokens "${JUDGE_MAX_NEW_TOKENS:-16}"
    --output "${OUTPUT_DIR}/raw_judge_predictions.jsonl"
    --no-eval
)
if [[ "${JUDGE_ALL:-0}" == "1" ]]; then
    cmd+=(--judge-all)
fi
if [[ "${HIDE_CANDIDATE_SUGGESTIONS:-0}" == "1" ]]; then
    cmd+=(--hide-candidate-suggestions)
fi
if [[ "${JUDGE_TIES_ONLY:-0}" == "1" ]]; then
    cmd+=(--judge-plurality-ties-only)
fi
if [[ -n "${THINKING_TOKEN_BUDGET:-}" ]]; then
    export VLLM_THINKING_TOKEN_BUDGET=${THINKING_TOKEN_BUDGET}
    cmd+=(--thinking-token-budget "${THINKING_TOKEN_BUDGET}")
fi
if [[ "${REQUIRE_FINAL_ANSWER_MARKER:-0}" == "1" ]]; then
    cmd+=(--require-final-answer-marker)
fi
if [[ "${INCLUDE_FRAME_TIMESTAMPS:-0}" == "1" ]]; then
    cmd+=(--include-frame-timestamps)
fi
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
"${cmd[@]}"

read -ra policies <<< "${JUDGE_POLICIES:-all_disagreements candidate_supported all_different_only}"
for policy in "${policies[@]}"; do
    python "${ROOT}/scripts/evaluate_longqa_multicandidate_judge.py" \
        --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
        --subset-file "${SUBSET_FILE}" \
        --fallback-predictions "${FALLBACK}/predictions.jsonl" \
        --judge-predictions "${OUTPUT_DIR}/raw_judge_predictions.jsonl" \
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
