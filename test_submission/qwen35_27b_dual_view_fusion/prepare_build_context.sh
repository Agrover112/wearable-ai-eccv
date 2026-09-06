#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_KIT="${ROOT}/data/wearable-ai/starter_kit"
SUBMISSION="${ROOT}/test_submission/qwen35_27b_dual_view_fusion"
DEST_ROOT="${1:-/scratch/inf0/user/agaur/wai-26/test_submission/qwen35_27b_dual_view_fusion}"
DEST_KIT="${DEST_ROOT}/starter_kit"

QWEN_SNAPSHOT="${QWEN_SNAPSHOT:-/scratch/inf0/user/agaur/wai-26/cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654}"
SIGLIP_SNAPSHOT="${SIGLIP_SNAPSHOT:-/scratch/inf0/user/agaur/wai-26/cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f}"

for path in "${SOURCE_KIT}" "${QWEN_SNAPSHOT}" "${SIGLIP_SNAPSHOT}"; do
    if [[ ! -d "${path}" ]]; then
        echo "Required directory is missing: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${DEST_KIT}/test_phase" "${DEST_KIT}/my_weights"
find "${DEST_KIT}" -mindepth 1 -maxdepth 1 \
    ! -name my_weights ! -name test_phase -exec rm -rf {} +

cp "${SOURCE_KIT}"/*.py "${DEST_KIT}/"
cp "${SOURCE_KIT}/requirements.txt" "${DEST_KIT}/"
cp "${SOURCE_KIT}/LICENSE" "${DEST_KIT}/"
cp "${SOURCE_KIT}/README.md" "${DEST_KIT}/STARTER_KIT_README.md"
cp "${SUBMISSION}/test_phase/"* "${DEST_KIT}/test_phase/"

stage_snapshot() {
    local source="$1"
    local destination="$2"
    local marker="${destination}/.snapshot_complete"
    local revision
    revision="$(basename "${source}")"
    if [[ -f "${marker}" ]] && grep -qx "${revision}" "${marker}"; then
        echo "Reusing staged snapshot: ${destination}"
        return
    fi
    rm -rf "${destination}"
    mkdir -p "${destination}"
    # Prefer dereferenced hard links on the shared scratch filesystem. Some
    # striped filesystems reject links for particular inodes, so fall back to a
    # byte copy per file while preserving a complete, symlink-free context.
    for item in "${source}"/*; do
        name="$(basename "${item}")"
        resolved="$(readlink -f "${item}")"
        if [[ -d "${resolved}" ]]; then
            cp -aL "${resolved}" "${destination}/${name}"
        elif ! ln "${resolved}" "${destination}/${name}" 2>/dev/null; then
            cp -aL "${resolved}" "${destination}/${name}"
        fi
    done
    printf '%s\n' "${revision}" > "${marker}"
}

stage_snapshot "${QWEN_SNAPSHOT}" "${DEST_KIT}/my_weights/qwen35-27b"
stage_snapshot "${SIGLIP_SNAPSHOT}" "${DEST_KIT}/my_weights/siglip2-so400m-patch14-384"

chmod +x "${DEST_KIT}/test_phase/build_image.sh" "${DEST_KIT}/test_phase/validate_image.sh"

echo "Build context: ${DEST_KIT}"
du -sh "${DEST_KIT}" "${DEST_KIT}/my_weights/"*
echo "Build with:"
echo "  cd ${DEST_KIT}/test_phase"
echo "  ./build_image.sh --tag kth-saar-qwen35-dual-view:v1"
