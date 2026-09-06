#!/bin/bash

# Shared execution body for primary-aware hypothesis verification on dev140.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER_KIT=${ROOT}/data/wearable-ai/starter_kit
: "${HYPOTHESIS_MODE:?HYPOTHESIS_MODE is required}"
: "${RUN_STEM:?RUN_STEM is required}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX is required}"

PIVOT_RUN=${PIVOT_RUN:-qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-29}
UNIFORM_RUN=${UNIFORM_RUN:-qwen35_9b_vllm_uniform64_px451584_full_2026-07-30}
TERTIARY_RUN=${TERTIARY_RUN:-qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-12}
FALLBACK_RUN=${FALLBACK_RUN:-qwen35_9b_vllm_rotation_avg_pivot_full_2026-07-31}
PIVOT_DIR=${PIVOT_DIR:-${ROOT}/runs/egolongqa/${PIVOT_RUN}}
UNIFORM_DIR=${UNIFORM_DIR:-${ROOT}/runs/egolongqa/${UNIFORM_RUN}}
TERTIARY_DIR=${TERTIARY_DIR:-${ROOT}/runs/egolongqa/${TERTIARY_RUN}}
FALLBACK_DIR=${FALLBACK_DIR:-${ROOT}/runs/egolongqa/${FALLBACK_RUN}}
SUBSET_FILE=${SUBSET_FILE:-${ROOT}/configs/egolongqa_dev140_seed20260709.json}
PROOFPACK_DIR=${PROOFPACK_DIR:-${PIVOT_DIR}}
LLM_MODEL=${LLM_MODEL:-Qwen/Qwen3.5-9B}

for file in \
    "${PIVOT_DIR}/predictions.jsonl" "${PROOFPACK_DIR}/proofpack.jsonl" \
    "${UNIFORM_DIR}/predictions.jsonl" "${TERTIARY_DIR}/predictions.jsonl" \
    "${FALLBACK_DIR}/predictions.jsonl"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: required input is missing: ${file}" >&2
        exit 2
    fi
done

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
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
export QWEN_ENABLE_THINKING=0
export VLLM_REASONING_PARSER=qwen3
export VLLM_GDN_PREFILL_BACKEND=triton

RUN_NAME=${RUN_NAME:-${RUN_STEM}_$(date +%F)}
OUTPUT_DIR=output/egolongqa/${RUN_NAME}
ARCHIVE_DIR=${ROOT}/runs/egolongqa/${RUN_NAME}
GROUNDER_CACHE_DIR=${GROUNDER_CACHE_DIR:-/scratch/inf0/user/agaur/wai-26/data/wearable-ai/grounder_features}
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    GROUNDER_CACHE_DIR=/tmp/wai_siglip2_c128_dry_run
fi
export VLLM_LOG_DIR=${STARTER_KIT}/${OUTPUT_DIR}
mkdir -p "${ROOT}/slurm_logs" "${STARTER_KIT}/${OUTPUT_DIR}" "${ARCHIVE_DIR}" \
    "${GROUNDER_CACHE_DIR}" "${HUGGINGFACE_HUB_CACHE}" "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}"

cd "${STARTER_KIT}"
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi
python -c 'import psutil, vllm'

cmd=(
    python run_generate_longqa_hypothesis_judge.py
    --video-folder ../egolongqa/val
    --subset-file "${SUBSET_FILE}"
    --mode "${HYPOTHESIS_MODE}"
    --primary-predictions "${PIVOT_DIR}/predictions.jsonl"
    --secondary-predictions "${UNIFORM_DIR}/predictions.jsonl"
    --tertiary-predictions "${TERTIARY_DIR}/predictions.jsonl"
    --fallback-predictions "${FALLBACK_DIR}/predictions.jsonl"
    --proofpack "${PROOFPACK_DIR}/proofpack.jsonl"
    --proofpack-reference "${PROOFPACK_DIR}/predictions.jsonl"
    --grounder-model google/siglip2-so400m-patch14-384
    --grounder-cache-dir "${GROUNDER_CACHE_DIR}"
    --candidate-frames 128
    --centers-per-candidate 4
    --temporal-nms-seconds 10
    --neighborhood-radius 1
    --max-evidence-refinements "${MAX_EVIDENCE_REFINEMENTS:-0}"
    --refinement-frame-budget 12
    --refinement-local-candidate-frames 64
    --refinement-full-candidate-frames 128
    --refinement-around-seconds 30
    --max-frames 64
    --max-pixels 451584
    --llm-model "${LLM_MODEL}"
    --concurrency 1
    --output "${OUTPUT_DIR}/predictions.jsonl"
    --audit-output "${OUTPUT_DIR}/audit.jsonl"
    --eval-output "${OUTPUT_DIR}/results.json"
)
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
START=$(date +%s)
"${cmd[@]}"

python "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${OUTPUT_DIR}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id "${RUN_NAME}" --output "${OUTPUT_DIR}/diagnostics.json"
for file in predictions.jsonl audit.jsonl results.json results_summary.json diagnostics.json; do
    cp "${OUTPUT_DIR}/${file}" "${ARCHIVE_DIR}/${file}"
done
cp -r "${OUTPUT_DIR}/hypothesis_stages" "${ARCHIVE_DIR}/"
LATEST_LOG=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ARCHIVE_DIR}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ARCHIVE_DIR}/slurm.err" || true
fi
echo "Runtime seconds: $(($(date +%s) - START))"
echo "Archive: ${ARCHIVE_DIR}"
