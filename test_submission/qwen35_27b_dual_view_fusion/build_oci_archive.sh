#!/bin/bash

set -euo pipefail

BUILDCTL=${BUILDCTL:-/tmp/wai-buildkit/bin/buildctl}
BUILDKIT_ADDR=${BUILDKIT_ADDR:-unix:///tmp/wai-buildkit/buildkitd4.sock}
CONTEXT=${1:-/scratch/inf0/user/agaur/wai-26/test_submission/qwen35_27b_dual_view_fusion_v1/starter_kit}
OUTPUT=${2:-/scratch/inf0/user/agaur/wai-26/test_submission/qwen35_27b_dual_view_fusion_v1/kth-saar-qwen35-dual-view-v1.oci.tar}

if [[ ! -x "${BUILDCTL}" ]]; then
    echo "buildctl is unavailable: ${BUILDCTL}" >&2
    exit 1
fi
if [[ ! -f "${CONTEXT}/my_weights/qwen35-27b/.snapshot_complete" ]] || \
   [[ ! -f "${CONTEXT}/my_weights/siglip2-so400m-patch14-384/.snapshot_complete" ]]; then
    echo "Pinned snapshots are not completely staged; run prepare_build_context.sh first." >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"
"${BUILDCTL}" --addr "${BUILDKIT_ADDR}" build \
    --frontend dockerfile.v0 \
    --local context="${CONTEXT}" \
    --local dockerfile="${CONTEXT}/test_phase" \
    --opt filename=Containerfile \
    --opt platform=linux/amd64 \
    --output "type=oci,name=kth-saar-qwen35-dual-view:v1,dest=${OUTPUT},compression=zstd,compression-level=3,force-compression=true"

sha256sum "${OUTPUT}" > "${OUTPUT}.sha256"
ls -lh "${OUTPUT}" "${OUTPUT}.sha256"
echo "Local OCI archive created. It has not been uploaded."
