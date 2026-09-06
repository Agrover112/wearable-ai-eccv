#!/bin/bash

# Score independent LongQA candidates with the rubric-tuned VideoJudge-7B.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=/CT/NDF/work/waw-26
STARTER=${ROOT}/data/wearable-ai/starter_kit
: "${RUN_NAME:?RUN_NAME is required}"
: "${SLURM_LOG_PREFIX:?SLURM_LOG_PREFIX is required}"
: "${SUBSET_FILE:?SUBSET_FILE is required}"

PIVOT=${ROOT}/runs/egolongqa/qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-29
UNIFORM=${ROOT}/runs/egolongqa/qwen35_9b_vllm_uniform64_px451584_full_2026-07-30
UNCERTAINTY=${ROOT}/runs/egolongqa/qwen35_9b_vllm_uncertainty_pivot_full_2026-08-02
OPTION=${ROOT}/runs/egolongqa/qwen35_9b_vllm_siglip2_option_quota_pivot_anc24_final64_px451584_full_2026-08-02
ENDPOINT=${ROOT}/runs/egolongqa/qwen35_9b_vllm_uniform64_endpoint_px451584_full_2026-08-02
PRIMARY=${ROOT}/runs/egolongqa/qwen35_27b_multievidence_candidate_blind_full_2026-08-03
OUTPUT_DIR=${STARTER}/output/egolongqa/${RUN_NAME}
ARCHIVE_DIR=${ROOT}/runs/egolongqa/${RUN_NAME}

for file in \
    "${PIVOT}/predictions.jsonl" "${PIVOT}/proofpack.jsonl" \
    "${UNIFORM}/predictions.jsonl" \
    "${UNCERTAINTY}/predictions.jsonl" "${UNCERTAINTY}/selections.jsonl" \
    "${OPTION}/predictions.jsonl" "${OPTION}/proofpack.jsonl" \
    "${ENDPOINT}/predictions.jsonl" "${PRIMARY}/predictions.jsonl" \
    "${SUBSET_FILE}"; do
    if [[ ! -s "${file}" ]]; then
        echo "ERROR: missing VideoJudge input: ${file}" >&2
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
export QWEN_MAX_PIXELS=${QWEN_MAX_PIXELS:-50176}
export VLLM_QWEN_MAX_MODEL_LEN=${VLLM_QWEN_MAX_MODEL_LEN:-32768}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.92}
export VLLM_QWEN_MEDIA_MODE=video
export VLLM_LOG_DIR=${OUTPUT_DIR}
mkdir -p "${ROOT}/slurm_logs" "${OUTPUT_DIR}" "${ARCHIVE_DIR}"
USER_SITE=$(python -c 'import site; print(site.USER_SITE)')
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi
python -c 'import psutil, vllm' >/dev/null

cmd=(
    python "${STARTER}/run_generate_longqa_videojudge.py"
    --video-folder "${ROOT}/data/wearable-ai/egolongqa/val"
    --subset-file "${SUBSET_FILE}"
    --candidate-predictions "${PIVOT}/predictions.jsonl"
    --candidate-predictions "${UNIFORM}/predictions.jsonl"
    --candidate-predictions "${UNCERTAINTY}/predictions.jsonl"
    --candidate-predictions "${OPTION}/predictions.jsonl"
    --candidate-predictions "${ENDPOINT}/predictions.jsonl"
    --candidate-labels pivot uniform uncertainty option_quota endpoint
    --primary-predictions "${PRIMARY}/predictions.jsonl"
    --pivot-proofpack "${PIVOT}/proofpack.jsonl"
    --option-proofpack "${OPTION}/proofpack.jsonl"
    --uncertainty-selection "${UNCERTAINTY}/selections.jsonl"
    --llm-model "${VIDEOJUDGE_MODEL:-VideoJudge/Qwen2.5-VL-7B-Instruct-VideoJudgeWithRubric-RS-20K}"
    --backend vllm --tp 1 --concurrency 1
    --max-frames "${MAX_FRAMES:-64}"
    --evidence-mode "${EVIDENCE_MODE:-primary}"
    --candidate-mode "${CANDIDATE_MODE:-unique}"
    --max-new-tokens "${VIDEOJUDGE_MAX_NEW_TOKENS:-1024}"
    --output "${OUTPUT_DIR}/predictions.jsonl"
    --eval-output "${OUTPUT_DIR}/results.json"
)
if [[ "${JUDGE_ALL:-0}" == "1" ]]; then
    cmd+=(--judge-all)
fi
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
"${cmd[@]}"

cp "${OUTPUT_DIR}"/*.json "${ARCHIVE_DIR}/"
cp "${OUTPUT_DIR}"/*.jsonl "${ARCHIVE_DIR}/"
LATEST_LOG=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'vllm_server_*.log' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
[[ -n "${LATEST_LOG}" ]] && cp "${LATEST_LOG}" "${ARCHIVE_DIR}/vllm_server.log"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.out" "${ARCHIVE_DIR}/slurm.out" || true
    cp "${ROOT}/slurm_logs/${SLURM_LOG_PREFIX}_${SLURM_JOB_ID}.err" "${ARCHIVE_DIR}/slurm.err" || true
fi
echo "Archive: ${ARCHIVE_DIR}"
