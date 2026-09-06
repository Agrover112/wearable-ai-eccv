#!/bin/bash

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/inf0/user/agaur/wai-26}
BEFORE=${ROOT}/documentation/cleanup_records/scratch_deletion_20260901_before.tsv
RESTORE_DOC=${ROOT}/documentation/SCRATCH_ASSET_RESTORATION.md
RECOVERY=${ROOT}/archives/submission_recovery_2026-09-01
EXPECTED_ARCHIVE_SHA=d467137d5e3bfcd41dba817aa2e5f8643d72ff54fe39b5a3b764e63735f8db0d

[[ "${SCRATCH_ROOT}" == /scratch/inf0/user/agaur/wai-26 ]] || {
    echo "Refusing unexpected scratch root: ${SCRATCH_ROOT}" >&2
    exit 1
}
[[ -d "${SCRATCH_ROOT}" ]] || {
    echo "Scratch tree is already absent: ${SCRATCH_ROOT}" >&2
    exit 1
}
for path in \
    "${BEFORE}" \
    "${RESTORE_DOC}" \
    "${ROOT}/documentation/reproducibility_manifests/dataset-inventory.tsv" \
    "${ROOT}/documentation/reproducibility_manifests/huggingface-model-snapshots.tsv" \
    "${RECOVERY}/artifacts/kth-saar-qwen35-dual-view-v3.tar.sha256" \
    "${RECOVERY}/validation/CONTAINER_SMOKE_PASSED_V3" \
    "${RECOVERY}/artifacts/ecr_immutable_reference_v3.txt"; do
    [[ -f "${path}" ]] || { echo "Required record is missing: ${path}" >&2; exit 1; }
done
grep -q "${EXPECTED_ARCHIVE_SHA}" \
    "${RECOVERY}/artifacts/kth-saar-qwen35-dual-view-v3.tar.sha256" || {
    echo "Preserved image checksum record is incorrect" >&2
    exit 1
}
sha256sum -c "${RECOVERY}/SHA256SUMS"

if command -v squeue >/dev/null 2>&1 && squeue -h -u "$(id -u)" | grep -q .; then
    echo "Refusing deletion while Slurm jobs are queued or running" >&2
    exit 1
fi

recorded_bytes=$(awk -F '\t' -v path="${SCRATCH_ROOT}" '$1 == path {print $2}' "${BEFORE}")
[[ "${recorded_bytes}" == 392437898240 ]] || {
    echo "Scratch inventory does not match the reviewed deletion target" >&2
    exit 1
}

rm -rf --one-file-system -- "${SCRATCH_ROOT}"
[[ ! -e "${SCRATCH_ROOT}" ]] || {
    echo "Scratch deletion did not complete" >&2
    exit 1
}

cat > "${ROOT}/documentation/cleanup_records/scratch_deletion_20260901_receipt.txt" <<EOF
deleted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
deleted_path=${SCRATCH_ROOT}
recorded_bytes=${recorded_bytes}
restoration_document=documentation/SCRATCH_ASSET_RESTORATION.md
restoration_script=scripts/maintenance/restore_wearable_ai_assets.sh
final_image_sha256=${EXPECTED_ARCHIVE_SHA}
final_image_local_archive=deleted
EOF

echo "Deleted ${SCRATCH_ROOT}; restoration record preserved in the workspace"
