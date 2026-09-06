#!/bin/bash

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/inf0/user/agaur/wai-26}
RECOVERY_ROOT=${RECOVERY_ROOT:-${ROOT}/archives/submission_recovery_2026-09-01}
ARTIFACT_ROOT=${SCRATCH_ROOT}/test_submission/qwen35_27b_dual_view_fusion_v2/artifacts
FINAL_ARCHIVE=${ARTIFACT_ROOT}/kth-saar-qwen35-dual-view-v3.tar
FINAL_SIDECAR=${FINAL_ARCHIVE}.sha256
FINAL_MARKER=${RECOVERY_ROOT}/validation/CONTAINER_SMOKE_PASSED_V3
FINAL_REFERENCE=${RECOVERY_ROOT}/artifacts/ecr_immutable_reference_v3.txt
EXPECTED_BYTES=67720555008
EXPECTED_SHA=d467137d5e3bfcd41dba817aa2e5f8643d72ff54fe39b5a3b764e63735f8db0d
MODE=${1:---dry-run}
RECORD_DIR=${ROOT}/documentation/cleanup_records
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--apply" ]]; then
    echo "Usage: $0 [--dry-run|--apply]" >&2
    exit 2
fi

mkdir -p "${RECORD_DIR}"

require_file() {
    [[ -f "$1" ]] || { echo "Protected artifact is missing: $1" >&2; exit 1; }
}

require_file "${FINAL_ARCHIVE}"
require_file "${FINAL_SIDECAR}"
require_file "${FINAL_MARKER}"
require_file "${FINAL_REFERENCE}"
[[ "$(stat -c %s "${FINAL_ARCHIVE}")" == "${EXPECTED_BYTES}" ]] || {
    echo "Final archive size does not match the recorded receipt" >&2
    exit 1
}
grep -q "${EXPECTED_SHA}" "${FINAL_SIDECAR}" || {
    echo "Final archive sidecar does not contain the recorded SHA-256" >&2
    exit 1
}

if command -v squeue >/dev/null 2>&1 && squeue -h -u "$(id -un)" | grep -q .; then
    echo "Refusing cleanup while Slurm jobs are queued or running" >&2
    exit 1
fi

TARGETS=(
    "${SCRATCH_ROOT}/test_submission/qwen35_27b_dual_view_fusion_v1"
    "${SCRATCH_ROOT}/test_submission/qwen35_27b_dual_view_fusion_v2/starter_kit"
    "${ARTIFACT_ROOT}/kth-saar-qwen35-dual-view-v2.tar"
    "${ARTIFACT_ROOT}/crane-cache"
    "${ARTIFACT_ROOT}/apptainer"
    "${ARTIFACT_ROOT}/link_test"
    "${SCRATCH_ROOT}/envs/nvila-autogaze"
    "${SCRATCH_ROOT}/cache/huggingface/transformers"
    "${SCRATCH_ROOT}/cache/huggingface/hub/models--OpenGVLab--InternVL3_5-30B-A3B-Flash"
    "${SCRATCH_ROOT}/cache/huggingface/hub/models--Qwen--Qwen3.5-9B"
    "${SCRATCH_ROOT}/cache/huggingface/hub/models--Qwen--Qwen3.8-27B"
    "${SCRATCH_ROOT}/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking"
    "${SCRATCH_ROOT}/cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B"
    "${SCRATCH_ROOT}/cache/huggingface/hub/models--Qwen--Qwen3-VL-Reranker-2B"
    "${SCRATCH_ROOT}/cache/huggingface/hub/models--VideoJudge--Qwen2.5-VL-7B-Instruct-VideoJudgeWithRubric-RS-20K"
    "${SCRATCH_ROOT}/cache/huggingface/hub/models--allenai--Molmo2-8B"
    "${SCRATCH_ROOT}/cache/huggingface/hub/models--google--siglip-base-patch16-224"
    "${SCRATCH_ROOT}/cache/huggingface/hub/models--openbmb--MiniCPM-V-4_5"
    "${SCRATCH_ROOT}/cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
    "${SCRATCH_ROOT}/cache/triton"
    "${SCRATCH_ROOT}/cache/flashinfer_workspace"
    "${SCRATCH_ROOT}/cache/cuda"
    "/tmp/wai-docker-config"
    "/tmp/wai-registry-tools"
    "/tmp/wai-ecr-layer-url"
)

record_targets() {
    local output=$1
    printf 'path\tbytes\texists\n' > "${output}"
    for path in "${TARGETS[@]}"; do
        if [[ -e "${path}" || -L "${path}" ]]; then
            printf '%s\t%s\tyes\n' "${path}" "$(du -s -B1 "${path}" 2>/dev/null | cut -f1 || echo unknown)" >> "${output}"
        else
            printf '%s\t0\tno\n' "${path}" >> "${output}"
        fi
    done
}

record_targets "${RECORD_DIR}/cleanup_${STAMP}_before.tsv"

if [[ "${MODE}" == "--dry-run" ]]; then
    cat "${RECORD_DIR}/cleanup_${STAMP}_before.tsv"
    echo "Dry run only. Protected v3 archive: ${FINAL_ARCHIVE}"
    exit 0
fi

for path in "${TARGETS[@]}"; do
    [[ "${path}" != "${FINAL_ARCHIVE}" ]] || exit 1
    rm -rf --one-file-system -- "${path}"
done

find "${ROOT}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
find "${ROOT}" -type d -name __pycache__ -prune -exec rm -rf -- {} +

record_targets "${RECORD_DIR}/cleanup_${STAMP}_after.tsv"
{
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'protected_archive=%s\n' "${FINAL_ARCHIVE}"
    printf 'protected_archive_bytes=%s\n' "$(stat -c %s "${FINAL_ARCHIVE}")"
    printf 'protected_archive_sha256_record=%s\n' "${EXPECTED_SHA}"
    printf 'protected_ecr_reference=%s\n' "$(cat "${FINAL_REFERENCE}")"
} > "${RECORD_DIR}/cleanup_${STAMP}_receipt.txt"

echo "Cleanup complete. Receipt: ${RECORD_DIR}/cleanup_${STAMP}_receipt.txt"
