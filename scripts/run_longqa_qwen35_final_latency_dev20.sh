#!/bin/bash

# Run the complete final ensemble path on dev20 with fresh selection caches.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
SUBSET=${ROOT}/configs/egolongqa_dev20_seed20260709.json
TAG=${SLURM_JOB_ID:-manual_$(date +%s)}
PREFIX=latency_qwen35_final_dev20_${TAG}
SCRATCH_ROOT=/scratch/inf0/user/agaur/wai-26/latency_smoke/${PREFIX}
PIVOT_NAME=${PREFIX}_pivot
UG_NAME=${PREFIX}_uncertainty
ENDPOINT_NAME=${PREFIX}_endpoint
PLAIN_NAME=${PREFIX}_plain27
THINKING_NAME=${PREFIX}_thinking27
FINAL_NAME=${PREFIX}_majority
PIVOT_DIR=${ROOT}/runs/egolongqa/${PIVOT_NAME}
UG_DIR=${ROOT}/runs/egolongqa/${UG_NAME}
ENDPOINT_DIR=${ROOT}/runs/egolongqa/${ENDPOINT_NAME}
PLAIN_DIR=${ROOT}/runs/egolongqa/${PLAIN_NAME}
THINKING_DIR=${ROOT}/runs/egolongqa/${THINKING_NAME}
FINAL_DIR=${ROOT}/runs/egolongqa/${FINAL_NAME}
mkdir -p "${SCRATCH_ROOT}" "${FINAL_DIR}"

stages=()
run_stage() {
    local name=$1
    shift
    local begun ended
    begun=$(date +%s)
    echo "===== Stage ${name} started at $(date --iso-8601=seconds) ====="
    "$@"
    ended=$(date +%s)
    stages+=("${name}=$((ended - begun))")
    echo "===== Stage ${name} finished in $((ended - begun)) seconds ====="
}

run_stage siglip2_pivot env \
    RUN_NAME=${PIVOT_NAME} \
    OUTPUT_DIR=output/egolongqa/${PIVOT_NAME} \
    ARCHIVE_DIR=${PIVOT_DIR} \
    STRATEGY=temporal_pivot \
    RUN_STEM=${PIVOT_NAME} \
    SLURM_LOG_PREFIX=qwen35_final_latency \
    SUBSET_FILE=${SUBSET} \
    LLM_MODEL=Qwen/Qwen3.5-9B \
    QWEN_ENABLE_THINKING=0 \
    QWEN_MAX_PIXELS=451584 \
    VLLM_QWEN_MAX_MODEL_LEN=49152 \
    CANDIDATE_FRAMES=128 \
    PIVOT_CENTERS=2 \
    TARGET_CENTERS=8 \
    EVENTLET_RADIUS=1 \
    ANCHOR_K=24 \
    BRIDGE_K=8 \
    FINAL_MAX_FRAMES=64 \
    FILL_MODE=semantic_boundary \
    GROUNDING_CACHE_DIR=${SCRATCH_ROOT}/siglip2 \
    bash "${ROOT}/scripts/run_longqa_proofpack_dev.sh" \
    --no-resume-grounding --no-resume-predictions

run_stage uncertainty env \
    RUN_NAME=${UG_NAME} \
    RUN_STEM=${UG_NAME} \
    SLURM_LOG_PREFIX=qwen35_final_latency \
    SUBSET_FILE=${SUBSET} \
    UNCERTAINTY_MODE=temporal_pivot \
    LLM_MODEL=Qwen/Qwen3.5-9B \
    QWEN_ENABLE_THINKING=0 \
    CANDIDATE_FRAMES=128 \
    UNIFORM_QUOTA=0 \
    FINAL_MAX_FRAMES=64 \
    QWEN_MAX_PIXELS=451584 \
    VLLM_QWEN_MAX_MODEL_LEN=49152 \
    CONCURRENCY=8 \
    SCORE_CACHE_DIR=${SCRATCH_ROOT}/uncertainty \
    bash "${ROOT}/scripts/run_longqa_uncertainty.sh" --no-resume

run_stage endpoint_uniform env \
    RUN_NAME=${ENDPOINT_NAME} \
    ARCHIVE_DIR=${ENDPOINT_DIR} \
    SLURM_LOG_PREFIX=qwen35_final_latency \
    SUBSET_FILE=${SUBSET} \
    LLM_MODEL=Qwen/Qwen3.5-9B \
    QWEN_ENABLE_THINKING=0 \
    MAX_FRAMES=64 \
    FRAMES_PER_INTERVAL=64 \
    QWEN_MAX_PIXELS=451584 \
    VLLM_QWEN_MAX_MODEL_LEN=49152 \
    CONCURRENCY=1 \
    bash "${ROOT}/slurm_longqa_qwen3_resolution_dev.sh" \
    --uniform-sampling endpoint_inclusive --no-resume-predictions

COMMON_JUDGE_ENV=(
    RUN_NAME=${PLAIN_NAME}
    SLURM_LOG_PREFIX=qwen35_final_latency
    SUBSET_FILE=${SUBSET}
    JUDGE_ALL=1
    HIDE_CANDIDATE_SUGGESTIONS=1
    JUDGE_POLICIES=all_disagreements
    PIVOT_RUN_DIR=${PIVOT_DIR}
    PROOFPACK_RUN_DIR=${PIVOT_DIR}
    UNCERTAINTY_RUN_DIR=${UG_DIR}
    ENDPOINT_RUN_DIR=${ENDPOINT_DIR}
)
run_stage direct27 env "${COMMON_JUDGE_ENV[@]}" \
    bash "${ROOT}/scripts/run_longqa_qwen35_27b_primary_judge.sh" \
    --no-resume

COMMON_JUDGE_ENV[0]="RUN_NAME=${THINKING_NAME}"
run_stage thinking27 env "${COMMON_JUDGE_ENV[@]}" \
    QWEN_ENABLE_THINKING=1 \
    THINKING_TOKEN_BUDGET=1024 \
    JUDGE_MAX_NEW_TOKENS=1152 \
    REQUIRE_FINAL_ANSWER_MARKER=1 \
    bash "${ROOT}/scripts/run_longqa_qwen35_27b_primary_judge.sh" \
    --no-resume

run_stage majority \
    /CT/NDF/work/miniforge3/envs/wearable-ai/bin/python \
    "${ROOT}/scripts/ensemble_longqa_predictions.py" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --subset-file "${SUBSET}" \
    --pred "plain27=${PLAIN_DIR}/all_disagreements_predictions.jsonl" \
    --pred "thinking27=${THINKING_DIR}/all_disagreements_predictions.jsonl" \
    --pred "endpoint9=${ENDPOINT_DIR}/predictions.jsonl" \
    --mode majority_vote \
    --output "${FINAL_DIR}/predictions.jsonl" \
    --eval-output "${FINAL_DIR}/results.json"

summary_cmd=(
    /CT/NDF/work/miniforge3/envs/wearable-ai/bin/python
    "${ROOT}/scripts/write_longqa_latency_summary.py"
    --samples 20
)
for stage in "${stages[@]}"; do
    summary_cmd+=(--stage "${stage}")
done
summary_cmd+=(--output "${FINAL_DIR}/latency_summary.json")
"${summary_cmd[@]}"
echo "Final latency smoke: ${FINAL_DIR}"
