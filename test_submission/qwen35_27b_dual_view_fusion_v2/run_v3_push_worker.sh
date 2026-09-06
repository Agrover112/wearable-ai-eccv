#!/bin/bash

set -u

ROOT=/CT/NDF/work/waw-26
LOG=/CT/HyperAvatar/work/wearable-ai-submission-recovery/logs/push_v3.log

mkdir -p "$(dirname "${LOG}")"
cd "${ROOT}"

set +e
bash test_submission/qwen35_27b_dual_view_fusion_v2/push_validated_image.sh \
    > "${LOG}" 2>&1
status=$?
printf 'push_exit_status=%s\nfinished_utc=%s\n' \
    "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG}"
exit "${status}"
