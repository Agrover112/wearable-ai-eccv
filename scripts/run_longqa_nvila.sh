#!/bin/bash

set -euo pipefail
ROOT=/CT/NDF/work/waw-26
ENV_PREFIX=/scratch/inf0/user/agaur/wai-26/envs/nvila-autogaze
PYTHON="${ENV_PREFIX}/bin/python"
READY_MARKER="${ENV_PREFIX}/.autogaze_ready"
: "${RUN_NAME:?Set RUN_NAME}"

if [[ ! -x "${PYTHON}" ]] || \
   [[ "$(cat "${READY_MARKER}" 2>/dev/null || true)" != "nvila-autogaze-v2" ]]; then
    echo "NVILA environment is incomplete. Run: sbatch slurm_setup_nvila_autogaze.sh" >&2
    exit 2
fi

MAX_SAMPLES="${MAX_SAMPLES:-5}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-${MAX_SAMPLES}}"
SUBSET_FILE="${SUBSET_FILE:-${ROOT}/configs/egolongqa_dev140_seed20260709.json}"
OUTPUT_DIR="${ROOT}/data/wearable-ai/starter_kit/output/egolongqa/${RUN_NAME}"
ARCHIVE_DIR="${ROOT}/runs/egolongqa/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}" "${ARCHIVE_DIR}" "${ROOT}/slurm_logs"

export HF_HOME="${HF_HOME:-/scratch/inf0/user/agaur/wai-26/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PYTHONNOUSERSITE=1

cmd=(
    "${PYTHON}" "${ROOT}/data/wearable-ai/starter_kit/run_generate_longqa_nvila.py"
    --input "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    --video-folder "${ROOT}/data/wearable-ai/egolongqa/val"
    --output "${OUTPUT_DIR}/predictions.jsonl"
    --subset-file "${SUBSET_FILE}"
    --num-video-frames 128
    --num-thumbnail-frames 64
    --max-tiles-video 48
    --max-new-tokens 32
)
if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
fi
echo "Command: ${cmd[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi
"${cmd[@]}"

"${PYTHON}" "${ROOT}/data/wearable-ai/starter_kit/run_evaluation.py" \
    --task longqa --eval-only \
    --golden "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --predictions "${OUTPUT_DIR}/predictions.jsonl" \
    --eval-output "${OUTPUT_DIR}/results.json"
cp "${OUTPUT_DIR}/predictions.jsonl" "${ARCHIVE_DIR}/predictions.jsonl"
cp "${OUTPUT_DIR}/results.json" "${ARCHIVE_DIR}/results.json"
"${PYTHON}" "${ROOT}/scripts/eval_longqa_diagnostics.py" \
    --predictions "${ARCHIVE_DIR}/predictions.jsonl" \
    --annotations "${ROOT}/data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl" \
    --run-id "${RUN_NAME}" --output "${ARCHIVE_DIR}/diagnostics.json"
if [[ -n "${EXPECTED_SAMPLES}" ]]; then
    "${PYTHON}" "${ROOT}/scripts/check_open_vlm_smoke.py" \
        --predictions "${ARCHIVE_DIR}/predictions.jsonl" \
        --expected "${EXPECTED_SAMPLES}" --latency-limit 300
fi
