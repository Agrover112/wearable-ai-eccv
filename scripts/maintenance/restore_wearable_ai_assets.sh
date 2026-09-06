#!/bin/bash

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/inf0/user/agaur/wai-26}
HF_HOME=${HF_HOME:-${SCRATCH_ROOT}/cache/huggingface}
DATA_LOCAL_DIR=${SCRATCH_ROOT}/data/wearable-ai
DATA_LINK=${ROOT}/data/wearable-ai/egolongqa
EXPECTED_ANNOTATION_SHA=e42d1d3de86dd15de73ff7e303652000b156f384c0f28700b49116043803436f

usage() {
    cat <<'EOF'
Usage: restore_wearable_ai_assets.sh OPTION [OPTION ...]

Options:
  --dataset       Restore EgoLongQA annotations and 700 validation videos.
  --core-models   Restore Qwen3.5-27B and SigLIP2 used by the final pipeline.
  --baselines     Restore the three retained baseline/grounding checkpoints.
  --autogaze      Restore the exact AutoGaze source revision used in exploration.
  --all           Restore dataset, core models, baselines, and AutoGaze.

The final OCI archive is not downloaded by this script. Its build and recovery
lineage is documented in documentation/SCRATCH_ASSET_RESTORATION.md.
EOF
}

if [[ $# -eq 0 ]]; then
    usage
    exit 2
fi

command -v hf >/dev/null 2>&1 || {
    echo "The Hugging Face CLI is required. Install huggingface_hub first." >&2
    exit 1
}

restore_dataset=0
restore_core=0
restore_baselines=0
restore_autogaze=0
for option in "$@"; do
    case "${option}" in
        --dataset) restore_dataset=1 ;;
        --core-models) restore_core=1 ;;
        --baselines) restore_baselines=1 ;;
        --autogaze) restore_autogaze=1 ;;
        --all)
            restore_dataset=1
            restore_core=1
            restore_baselines=1
            restore_autogaze=1
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: ${option}" >&2; usage >&2; exit 2 ;;
    esac
done

export HF_HOME
mkdir -p "${HF_HOME}"

download_model() {
    local model=$1
    local revision=$2
    echo "Restoring ${model}@${revision}"
    hf download "${model}" --revision "${revision}"
}

if [[ "${restore_dataset}" == 1 ]]; then
    mkdir -p "${DATA_LOCAL_DIR}"
    hf download facebook/wearable-ai \
        --repo-type dataset \
        --include 'egolongqa/**' 'README.md' 'LICENSE' \
        --local-dir "${DATA_LOCAL_DIR}"

    annotations=${DATA_LOCAL_DIR}/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl
    [[ -f "${annotations}" ]] || {
        echo "EgoLongQA annotations were not downloaded" >&2
        exit 1
    }
    actual_sha=$(sha256sum "${annotations}" | awk '{print $1}')
    [[ "${actual_sha}" == "${EXPECTED_ANNOTATION_SHA}" ]] || {
        echo "Annotation checksum differs from the recorded validation set" >&2
        echo "Expected: ${EXPECTED_ANNOTATION_SHA}" >&2
        echo "Actual:   ${actual_sha}" >&2
        exit 1
    }
    video_count=$(find "${DATA_LOCAL_DIR}/egolongqa/val" -maxdepth 1 -type f -name '*.mp4' | wc -l)
    [[ "${video_count}" == 700 ]] || {
        echo "Expected 700 EgoLongQA videos, found ${video_count}" >&2
        exit 1
    }

    if [[ -L "${DATA_LINK}" ]]; then
        rm "${DATA_LINK}"
    elif [[ -d "${DATA_LINK}" ]]; then
        extra=$(find "${DATA_LINK}" -mindepth 1 ! -name README.md -print -quit)
        [[ -z "${extra}" ]] || {
            echo "Refusing to replace non-empty data path: ${DATA_LINK}" >&2
            exit 1
        }
        rm -rf "${DATA_LINK}"
    fi
    ln -s "${DATA_LOCAL_DIR}/egolongqa" "${DATA_LINK}"
fi

if [[ "${restore_core}" == 1 ]]; then
    download_model Qwen/Qwen3.5-27B \
        fc05daec18b0a78c049392ed2e771dde82bdf654
    download_model google/siglip2-so400m-patch14-384 \
        e8e487298228002f3d8a82e0cd5c8ea9c567f57f
fi

if [[ "${restore_baselines}" == 1 ]]; then
    download_model Qwen/Qwen3-VL-8B-Instruct \
        0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
    download_model Qwen/Qwen2.5-VL-7B-Instruct \
        cc594898137f460bfe9f0759e9844b3ce807cfb5
    download_model IDEA-Research/grounding-dino-tiny \
        a2bb814dd30d776dcf7e30523b00659f4f141c71
fi

if [[ "${restore_autogaze}" == 1 ]]; then
    destination=${SCRATCH_ROOT}/src/AutoGaze
    if [[ ! -d "${destination}/.git" ]]; then
        mkdir -p "$(dirname "${destination}")"
        git clone https://github.com/NVlabs/AutoGaze.git "${destination}"
    fi
    git -C "${destination}" fetch --all --tags
    git -C "${destination}" checkout ba48d0f94ac2929d6fe3ee4380dc893aa6eed0ab
fi

echo "Requested assets restored under ${SCRATCH_ROOT}"
