#!/usr/bin/env python3
"""Validate that the compiler repair preserves the audited v2 image layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tarfile


BASE_DIFF_ID = "sha256:36fbfe1cc1ea79c0edd9f2e633a19ffd4f80f55211f4e0c2f783639395fbc032"
SOURCE_DIFF_ID = "sha256:697bad429755d9a219e7d7c6c1193a947c5b7803e4dd9f1b290f5f27bd1ee4c3"
EXPECTED_SOURCE = {
    "app/SUBMISSION_BUILD_PROVENANCE.json": "ee3eceac686dba5dd2bc293b3ced6ff63d05eec8fb8a3bebb21541db62fd179c",
    "app/longqa_dual_view_fusion.py": "91678f1ea1f9ebd6c12dfe5161a07a371f06363ffb4bf8c4aea66581ea21ab61",
    "app/model.py": "98745b9b35c6b6f013b141eab0a0c82a51edeeb077c3bd91d5a0ab72d3c2c920",
    "app/run_evaluation.py": "f0b210368be1281d42333f3fa8d186e0632c39978d0026048f84a9c70cb8b84a",
}


def read_json(archive: tarfile.TarFile, name: str) -> object:
    handle = archive.extractfile(name)
    if handle is None:
        raise RuntimeError(f"missing tar member: {name}")
    return json.load(handle)


def layer_files(archive: tarfile.TarFile, name: str) -> dict[str, str]:
    handle = archive.extractfile(name)
    if handle is None:
        raise RuntimeError(f"missing image layer: {name}")
    files: dict[str, str] = {}
    with tarfile.open(fileobj=handle, mode="r|*") as layer:
        for member in layer:
            clean = member.name.removeprefix("./")
            if member.isfile():
                source = layer.extractfile(member)
                if source is None:
                    raise RuntimeError(f"could not read layer file: {clean}")
                if clean in EXPECTED_SOURCE:
                    files[clean] = hashlib.sha256(source.read()).hexdigest()
                elif clean in {
                    "usr/bin/gcc",
                    "usr/bin/cc",
                    "usr/include/python3.10/Python.h",
                }:
                    files[clean] = "present"
            elif member.issym() and clean in {"usr/bin/gcc", "usr/bin/cc"}:
                files[clean] = f"symlink:{member.linkname}"
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=pathlib.Path)
    args = parser.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"image is missing: {args.image}")

    with tarfile.open(args.image, "r") as image:
        manifests = read_json(image, "manifest.json")
        if not isinstance(manifests, list) or len(manifests) != 1:
            raise RuntimeError("expected exactly one image manifest")
        manifest = manifests[0]
        config = read_json(image, str(manifest["Config"]))
        layers = manifest["Layers"]
        if len(layers) != 3:
            raise RuntimeError(f"expected base, source, and compiler layers; found {len(layers)}")

        diff_ids = config.get("rootfs", {}).get("diff_ids", [])
        if diff_ids[:2] != [BASE_DIFF_ID, SOURCE_DIFF_ID] or len(diff_ids) != 3:
            raise RuntimeError(f"audited v2 layers were not preserved: {diff_ids}")

        source_files = layer_files(image, str(layers[1]))
        if source_files != EXPECTED_SOURCE:
            raise RuntimeError(f"audited source layer changed: {source_files}")

        compiler_files = layer_files(image, str(layers[2]))
        if "usr/bin/gcc" not in compiler_files and "usr/bin/cc" not in compiler_files:
            raise RuntimeError("compiler layer does not provide a compiler entry point")
        if "usr/include/python3.10/Python.h" not in compiler_files:
            raise RuntimeError("compiler layer does not provide Python development headers")

        env = {}
        for item in config.get("config", {}).get("Env", []):
            key, value = item.split("=", 1)
            env[key] = value
        expected_env = {
            "CC": "/usr/bin/gcc",
            "WAI_QWEN_TP_SIZE": "1",
            "WAI_SIGLIP_GPU_INDEX": "1",
            "VLLM_QWEN_MAX_MODEL_LEN": "32768",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.90",
        }
        found_env = {key: env.get(key) for key in expected_env}
        if found_env != expected_env:
            raise RuntimeError(f"unexpected runtime environment: {found_env}")

    print("PASS: compiler repair preserves both audited v2 layers and runtime settings")


if __name__ == "__main__":
    main()
