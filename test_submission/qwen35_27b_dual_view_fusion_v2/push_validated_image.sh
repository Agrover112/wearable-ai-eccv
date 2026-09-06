#!/bin/bash

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RECOVERY_ROOT=${RECOVERY_ROOT:-/CT/HyperAvatar/work/wearable-ai-submission-recovery}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/scratch/inf0/user/agaur/wai-26/test_submission/qwen35_27b_dual_view_fusion_v2/artifacts}
CRANE=${CRANE:-/tmp/wai-registry-tools/crane}
export DOCKER_CONFIG=${DOCKER_CONFIG:-/tmp/wai-docker-config}
IMAGE=${IMAGE:-${ARTIFACT_ROOT}/kth-saar-qwen35-dual-view-v3.tar}
DESTINATION=${DESTINATION:-510788268643.dkr.ecr.us-east-2.amazonaws.com/wearable-ai-2026/kth-saar:qwen35-27b-dual-view-fusion-v3}

if [[ ! -f "${IMAGE}" ]]; then
    echo "Validated image tarball is missing: ${IMAGE}" >&2
    exit 1
fi
MARKER=${RECOVERY_ROOT}/validation/CONTAINER_SMOKE_PASSED_V3
if [[ ! -f "${MARKER}" ]]; then
    echo "Compiler-repaired container smoke-test marker is missing. Refusing to push." >&2
    exit 1
fi
VALIDATED_SHA=$(sed -n 's/^image_sha256=//p' "${MARKER}")
CURRENT_SHA=$(sha256sum "${IMAGE}" | awk '{print $1}')
if [[ -z "${VALIDATED_SHA}" || "${VALIDATED_SHA}" != "${CURRENT_SHA}" ]]; then
    echo "The image tarball does not match the container smoke-test marker. Refusing to push." >&2
    exit 1
fi
if [[ ! -x "${CRANE}" ]]; then
    echo "crane is unavailable: ${CRANE}" >&2
    exit 1
fi

python "${ROOT}/test_submission/qwen35_27b_dual_view_fusion_v2/validate_compiler_repair_tarball.py" \
    "${IMAGE}"

PUSH_OUTPUT=$("${CRANE}" push "${IMAGE}" "${DESTINATION}")
printf '%s\n' "${PUSH_OUTPUT}"
PUSH_DIGEST=$(printf '%s\n' "${PUSH_OUTPUT}" | grep -Eo 'sha256:[0-9a-f]{64}' | tail -1 || true)
REMOTE_DIGEST=$("${CRANE}" digest "${DESTINATION}")
if [[ -z "${PUSH_DIGEST}" || "${REMOTE_DIGEST}" != "${PUSH_DIGEST}" ]]; then
    echo "Remote tag digest does not match crane's pushed digest." >&2
    exit 1
fi
"${CRANE}" validate --remote "${DESTINATION}" --fast

REFERENCE_FILE=${RECOVERY_ROOT}/artifacts/ecr_immutable_reference_v3.txt
printf '%s@%s\n' "${DESTINATION%:*}" "${REMOTE_DIGEST}" > "${REFERENCE_FILE}"
cat "${REFERENCE_FILE}"
