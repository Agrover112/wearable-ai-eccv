"""Deterministic saved-proof-pack evidence adapter."""

from __future__ import annotations

import json
from typing import Any

from longqa_utils import sample_key

SOURCE_PRIORITY = {
    "pivot": 0,
    "directional_target": 0,
    "pivot_context": 1,
    "directional_target_context": 1,
    "bridge": 2,
    "event_center": 2,
    "anchor": 3,
    "semantic_boundary": 4,
}


def load_proofpack_map(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "r") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["sample_key"])
        if key in indexed:
            raise RuntimeError(f"Duplicate proof-pack sample key: {key}")
        if not isinstance(row["selected"], list):
            raise RuntimeError(f"Proof pack has no selected list: {key}")
        indexed[key] = row
    return indexed


def select_evidence(
    row: dict[str, Any],
    proofpack: dict[str, Any],
    fps: float,
    max_frames: int,
) -> list[dict[str, Any]]:
    if str(proofpack["video_path"]) != str(row["video_path"]):
        raise RuntimeError(f"Proof-pack video mismatch for {sample_key(row)}")
    if str(proofpack["sample_key"]) != sample_key(row):
        raise RuntimeError(f"Proof-pack sample-key mismatch for {sample_key(row)}")
    selected = list(proofpack["selected"])
    if len(selected) > max_frames:
        selected = sorted(
            selected,
            key=lambda item: (
                SOURCE_PRIORITY.get(str(item.get("source", "")), 5),
                -float(item.get("score") or 0.0),
                int(item["frame_index"]),
            ),
        )[:max_frames]
    selected.sort(key=lambda item: int(item["frame_index"]))
    evidence = []
    for frame_id, item in enumerate(selected, start=1):
        frame_index = int(item["frame_index"])
        evidence.append(
            {
                "frame_id": frame_id,
                "frame_index": frame_index,
                "timestamp": round(frame_index / fps, 3),
                "source": str(item.get("source", "proofpack")),
                "source_score": item.get("score"),
            }
        )
    return evidence
