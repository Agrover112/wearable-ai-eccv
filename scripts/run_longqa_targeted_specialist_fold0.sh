#!/bin/bash

set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=/CT/NDF/work/waw-26
STARTER=${ROOT}/data/wearable-ai/starter_kit
: "${SPECIALIST_MODE:?SPECIALIST_MODE is required}"
: "${SPECIALIST_GATE:?SPECIALIST_GATE is required}"
: "${RUN_NAME:?RUN_NAME is required}"

SUBSET_FILE=${SUBSET_FILE:-${ROOT}/configs/egolongqa_balanced_fold0_of5_20260808.json}
PIVOT_DIR=${PIVOT_DIR:-${ROOT}/runs/egolongqa/qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-29}
DETECTIONS_DIR=${DETECTIONS_DIR:-${ROOT}/runs/egolongqa/qwen35_27b_targeted_object_ledger_fold0_2026-08-08}
FALLBACK_DIR=${FALLBACK_DIR:-${ROOT}/runs/egolongqa/qwen35_27b_endpoint_optionquota_evidence_rank_mean_probability_full_2026-08-08}
OUTPUT=${STARTER}/output/egolongqa/${RUN_NAME}
ARCHIVE=${ROOT}/runs/egolongqa/${RUN_NAME}

for file in \
    "${SUBSET_FILE}" "${PIVOT_DIR}/proofpack.jsonl" \
    "${PIVOT_DIR}/predictions.jsonl" "${DETECTIONS_DIR}/detections.jsonl" \
    "${FALLBACK_DIR}/predictions.jsonl"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: missing targeted-specialist input: ${file}" >&2
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
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.92}
export QWEN_ENABLE_THINKING=0
export VLLM_GDN_PREFILL_BACKEND=triton
export VLLM_LOG_DIR=${OUTPUT}
mkdir -p "${ROOT}/slurm_logs" "${OUTPUT}" "${ARCHIVE}"
cd "${STARTER}"
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
[[ -d "${USER_SITE}" ]] && export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
python -c 'import psutil, vllm' >/dev/null

cmd=(
    python run_generate_longqa_specialist_evidence.py
    --video-folder ../egolongqa/val
    --subset-file "${SUBSET_FILE}"
    --proofpack "${PIVOT_DIR}/proofpack.jsonl"
    --proofpack-reference "${PIVOT_DIR}/predictions.jsonl"
    --detections-input "${DETECTIONS_DIR}/detections.jsonl"
    --fallback-predictions "${FALLBACK_DIR}/predictions.jsonl"
    --mode "${SPECIALIST_MODE}"
    --gate "${SPECIALIST_GATE}"
    --llm-model Qwen/Qwen3.5-27B
    --concurrency 1
    --max-frames 64
    --detail-count 12
    --reid-batch-size 16
    --reid-threshold 0.72
    --occurrence-gap-seconds 8
    --detections-per-frame 2
    --max-reid-detections 64
    --max-tracks 6
    --output "${OUTPUT}/predictions.jsonl"
    --evidence-output "${OUTPUT}/evidence.jsonl"
    --eval-output "${OUTPUT}/results.json"
)
if [[ "$#" -gt 0 ]]; then cmd+=("$@"); fi
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then exit 0; fi
START=$(date +%s)
"${cmd[@]}"

cp "${OUTPUT}"/*.json "${ARCHIVE}/"
cp "${OUTPUT}"/*.jsonl "${ARCHIVE}/"
LATEST_LOG=$(find "${OUTPUT}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ARCHIVE}/vllm_server.log"
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_LOG_PREFIX:-}" ]]; then
    if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        LOG_STEM=${SLURM_LOG_PREFIX}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
    else
        LOG_STEM=${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}
    fi
    [[ -f "${ROOT}/slurm_logs/${LOG_STEM}.out" ]] && \
        cp "${ROOT}/slurm_logs/${LOG_STEM}.out" "${ARCHIVE}/slurm.out"
    [[ -f "${ROOT}/slurm_logs/${LOG_STEM}.err" ]] && \
        cp "${ROOT}/slurm_logs/${LOG_STEM}.err" "${ARCHIVE}/slurm.err"
fi
echo "Runtime seconds: $(($(date +%s) - START))"
echo "Archive: ${ARCHIVE}"
