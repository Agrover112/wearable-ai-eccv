#!/bin/bash

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RECOVERY_ROOT=${RECOVERY_ROOT:-/CT/HyperAvatar/work/wearable-ai-submission-recovery}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/scratch/inf0/user/agaur/wai-26/test_submission/qwen35_27b_dual_view_fusion_v2/artifacts}
CRANE=${CRANE:-/tmp/wai-registry-tools/crane}
export DOCKER_CONFIG=${DOCKER_CONFIG:-/tmp/wai-docker-config}
REPOSITORY=${REPOSITORY:-510788268643.dkr.ecr.us-east-2.amazonaws.com/wearable-ai-2026/kth-saar}
BASE_DIGEST=sha256:5f8fad0c6994534b117613127d2f11b14f36551bc10866f30a909a32738fbe73
BASE=${BASE:-${REPOSITORY}@${BASE_DIGEST}}
BASE_LAYER_DIGEST=sha256:3686bf29ed06f77b282f002dcee87ec24603d51c8bf7cf609caba4faa547b453
OUTPUT=${OUTPUT:-${ARTIFACT_ROOT}/kth-saar-qwen35-dual-view-v2.tar}
CACHE=${CACHE:-${ARTIFACT_ROOT}/crane-cache}
LAYER=${LAYER:-/tmp/kth-saar-qwen35-dual-view-v2-layer.tar}
PARTIAL=${OUTPUT}.partial
URL_FILE=/tmp/wai-ecr-layer-url

if [[ ! -x "${CRANE}" ]]; then
    echo "crane is unavailable: ${CRANE}" >&2
    exit 1
fi
if [[ ! -f "${DOCKER_CONFIG}/config.json" ]]; then
    echo "ECR authentication is unavailable: ${DOCKER_CONFIG}/config.json" >&2
    exit 1
fi

LAYER_ROOT=$(mktemp -d /tmp/wai-v2-layer.XXXXXX)
cleanup() {
    rm -rf "${LAYER_ROOT}"
    rm -f "${URL_FILE}" "${PARTIAL}"
}
trap cleanup EXIT

mkdir -p "${LAYER_ROOT}/app" "${ARTIFACT_ROOT}" "${CACHE}" \
    "${RECOVERY_ROOT}/artifacts"
cp "${ROOT}/data/wearable-ai/starter_kit/model.py" "${LAYER_ROOT}/app/model.py"
cp "${ROOT}/data/wearable-ai/starter_kit/run_evaluation.py" "${LAYER_ROOT}/app/run_evaluation.py"
cp "${ROOT}/data/wearable-ai/starter_kit/longqa_dual_view_fusion.py" "${LAYER_ROOT}/app/longqa_dual_view_fusion.py"
cp "${ROOT}/test_submission/qwen35_27b_dual_view_fusion_v2/BUILD_PROVENANCE.json" \
   "${LAYER_ROOT}/app/SUBMISSION_BUILD_PROVENANCE.json"

tar --sort=name --mtime='UTC 2026-08-27' --owner=0 --group=0 --numeric-owner \
    -C "${LAYER_ROOT}" -cf "${LAYER}" app

# ECR redirects blob downloads to a signed backing-object URL. Capture that URL
# privately, then use ranged transfers so the 67 GB immutable layer is fetched
# once at useful bandwidth. The final SHA-256 check is the trust boundary.
REGISTRY=${REPOSITORY%%/*}
REPOSITORY_PATH=${REPOSITORY#*/}
export REGISTRY REPOSITORY_PATH BASE_LAYER_DIGEST URL_FILE
python - <<'PY'
import json
import os
import pathlib
import urllib.error
import urllib.request

registry = os.environ["REGISTRY"]
repository = os.environ["REPOSITORY_PATH"]
digest = os.environ["BASE_LAYER_DIGEST"]
docker_config = pathlib.Path(os.environ["DOCKER_CONFIG"]) / "config.json"
auth = json.loads(docker_config.read_text())["auths"][registry]["auth"]

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

request = urllib.request.Request(
    f"https://{registry}/v2/{repository}/blobs/{digest}",
    headers={"Authorization": f"Basic {auth}"},
)
try:
    urllib.request.build_opener(NoRedirect).open(request, timeout=60)
except urllib.error.HTTPError as error:
    if error.code not in (301, 302, 303, 307, 308):
        raise
    location = error.headers.get("Location")
    if not location:
        raise RuntimeError("ECR blob redirect did not include a Location header")
else:
    raise RuntimeError("ECR did not redirect the model-layer request")

path = pathlib.Path(os.environ["URL_FILE"])
path.write_text(location + "\n")
path.chmod(0o600)
PY

CACHE_BLOB=${CACHE}/${BASE_LAYER_DIGEST}
aria2c --continue=true --max-connection-per-server=16 --split=16 \
    --min-split-size=64M --file-allocation=none --auto-file-renaming=false \
    --allow-overwrite=true --show-console-readout=false --summary-interval=300 \
    --dir "${CACHE}" --out "${BASE_LAYER_DIGEST}" "$(cat "${URL_FILE}")"

EXPECTED_BLOB_SHA=${BASE_LAYER_DIGEST#sha256:}
ACTUAL_BLOB_SHA=$(sha256sum "${CACHE_BLOB}" | awk '{print $1}')
if [[ "${ACTUAL_BLOB_SHA}" != "${EXPECTED_BLOB_SHA}" ]]; then
    echo "Immutable base-layer digest mismatch" >&2
    exit 1
fi

# Push by digest only. This reuses the existing ECR base blob and uploads just
# the correction layer/config; it does not create a tag or a submission.
MUTATE_OUTPUT=$("${CRANE}" mutate "${BASE}" \
    --append "${LAYER}" \
    --env WAI_QWEN_TP_SIZE=1 \
    --env WAI_SIGLIP_GPU_INDEX=1 \
    --env VLLM_QWEN_MAX_MODEL_LEN=32768 \
    --env VLLM_GPU_MEMORY_UTILIZATION=0.90 \
    --label org.opencontainers.image.revision=qwen35-27b-dual-view-fusion-v2-recovery \
    --repo "${REPOSITORY}")
CORRECTED_DIGEST=$(printf '%s\n' "${MUTATE_OUTPUT}" | grep -Eo 'sha256:[0-9a-f]{64}' | tail -1)
if [[ -z "${CORRECTED_DIGEST}" ]]; then
    echo "Could not determine corrected image digest from crane output" >&2
    exit 1
fi
CORRECTED_REFERENCE=${REPOSITORY}@${CORRECTED_DIGEST}
printf '%s\n' "${CORRECTED_REFERENCE}" \
    > "${RECOVERY_ROOT}/artifacts/ecr_untagged_candidate_reference.txt"

rm -f "${PARTIAL}"
"${CRANE}" pull --cache_path "${CACHE}" "${CORRECTED_REFERENCE}" "${PARTIAL}"
mv "${PARTIAL}" "${OUTPUT}"

"${CRANE}" validate --remote "${CORRECTED_REFERENCE}" --fast
"${CRANE}" validate --tarball "${OUTPUT}"
python "${ROOT}/test_submission/qwen35_27b_dual_view_fusion_v2/validate_local_tarball.py" \
    "${OUTPUT}"

sha256sum "${LAYER}" > "${OUTPUT}.layer.sha256"
sha256sum "${OUTPUT}" > "${OUTPUT}.sha256"
ls -lh "${OUTPUT}" "${OUTPUT}.sha256" "${OUTPUT}.layer.sha256"
echo "Untagged candidate: ${CORRECTED_REFERENCE}"
