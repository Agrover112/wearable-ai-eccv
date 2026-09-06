#!/usr/bin/env python3
"""Validate the corrected Docker tarball without unpacking its 67 GB base layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tarfile


BASE_LAYER_DIGEST = "3686bf29ed06f77b282f002dcee87ec24603d51c8bf7cf609caba4faa547b453"
BASE_DIFF_ID = "sha256:36fbfe1cc1ea79c0edd9f2e633a19ffd4f80f55211f4e0c2f783639395fbc032"
EXPECTED_FILES = {
    "app/SUBMISSION_BUILD_PROVENANCE.json": "ee3eceac686dba5dd2bc293b3ced6ff63d05eec8fb8a3bebb21541db62fd179c",
    "app/longqa_dual_view_fusion.py": "91678f1ea1f9ebd6c12dfe5161a07a371f06363ffb4bf8c4aea66581ea21ab61",
    "app/model.py": "98745b9b35c6b6f013b141eab0a0c82a51edeeb077c3bd91d5a0ab72d3c2c920",
    "app/run_evaluation.py": "f0b210368be1281d42333f3fa8d186e0632c39978d0026048f84a9c70cb8b84a",
}


def _read_json(archive: tarfile.TarFile, name: str) -> dict[str, object]:
    handle = archive.extractfile(name)
    if handle is None:
        raise RuntimeError(f"missing tar member: {name}")
    return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=pathlib.Path)
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"image is missing: {args.image}")

    with tarfile.open(args.image, "r") as image:
        manifests = _read_json(image, "manifest.json")
        if not isinstance(manifests, list) or len(manifests) != 1:
            raise RuntimeError("expected exactly one image manifest")
        manifest = manifests[0]
        config = _read_json(image, str(manifest["Config"]))
        layers = manifest["Layers"]
        if len(layers) != 2:
            raise RuntimeError(f"expected base plus correction layer, found {len(layers)}")
        if BASE_LAYER_DIGEST not in str(layers[0]):
            raise RuntimeError(f"unexpected base layer: {layers[0]}")

        diff_ids = config.get("rootfs", {}).get("diff_ids", [])
        if len(diff_ids) != 2 or diff_ids[0] != BASE_DIFF_ID:
            raise RuntimeError(f"organizer-tested root filesystem was not preserved: {diff_ids}")

        env = {}
        for item in config.get("config", {}).get("Env", []):
            key, value = item.split("=", 1)
            env[key] = value
        if env.get("WAI_QWEN_TP_SIZE") != "1":
            raise RuntimeError(f"unexpected WAI_QWEN_TP_SIZE: {env.get('WAI_QWEN_TP_SIZE')}")
        if env.get("WAI_SIGLIP_GPU_INDEX") != "1":
            raise RuntimeError(
                f"unexpected WAI_SIGLIP_GPU_INDEX: {env.get('WAI_SIGLIP_GPU_INDEX')}"
            )
        if env.get("VLLM_QWEN_MAX_MODEL_LEN") != "32768":
            raise RuntimeError(
                f"unexpected VLLM_QWEN_MAX_MODEL_LEN: {env.get('VLLM_QWEN_MAX_MODEL_LEN')}"
            )
        if env.get("VLLM_GPU_MEMORY_UTILIZATION") != "0.90":
            raise RuntimeError(
                "unexpected VLLM_GPU_MEMORY_UTILIZATION: "
                f"{env.get('VLLM_GPU_MEMORY_UTILIZATION')}"
            )

        correction = image.extractfile(str(layers[1]))
        if correction is None:
            raise RuntimeError("correction layer is missing")
        seen_files: dict[str, str] = {}
        with tarfile.open(fileobj=correction, mode="r|*") as layer:
            for member in layer:
                name = member.name.removeprefix("./")
                if name.startswith("/") or ".." in pathlib.PurePosixPath(name).parts:
                    raise RuntimeError(f"unsafe correction-layer path: {member.name}")
                if member.isfile():
                    handle = layer.extractfile(member)
                    if handle is None:
                        raise RuntimeError(f"could not read correction file: {name}")
                    seen_files[name] = hashlib.sha256(handle.read()).hexdigest()
                elif not member.isdir():
                    raise RuntimeError(f"unexpected correction-layer entry: {member.name}")

    if seen_files != EXPECTED_FILES:
        raise RuntimeError(
            "correction layer does not match audited source\n"
            f"expected: {EXPECTED_FILES}\n"
            f"found: {seen_files}"
        )
    print("PASS: tarball preserves the old root filesystem and contains only audited corrections")


if __name__ == "__main__":
    main()
