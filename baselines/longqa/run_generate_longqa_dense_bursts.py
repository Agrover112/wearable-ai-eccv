#!/usr/bin/env python3
"""Run Qwen3.5 on global anchors plus proof-pack-centered temporal bursts.

This is deliberately a standalone LongQA experiment.  It does not change a
parent proof pack: it reads that pack's ranked event evidence, expands each
event into nearby frames using the video's measured FPS, and writes a new
frame-metadata JSONL before generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from longqa_utils import (
    PROMPT_VARIANTS,
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    index_row_aligned_metadata,
    load_jsonl,
    sample_key,
)
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices

logger = logging.getLogger(__name__)

DENSE_BURST_SCHEMA = 1
GLOBAL_ANCHOR_COUNT = 16
MAX_DENSE_BURST_FRAMES = 64
DEFAULT_BURST_OFFSETS_SECONDS = (-2.0, -1.0, -0.33, 0.33, 1.0, 2.0)


@dataclass(frozen=True)
class EventCenter:
    frame_index: int
    score: float | None
    source: str
    parent_rank: int


@dataclass(frozen=True)
class DenseBurstFrame:
    frame_index: int
    source: str
    center_frame_index: int | None = None
    offset_seconds: float | None = None
    center_rank: int | None = None
    center_score: float | None = None
    center_source: str | None = None


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _load_jsonl_if_exists(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    return load_jsonl(path)


def _uniform_indices(total_frames: int, count: int) -> list[int]:
    """Return distinct anchors including the beginning and end when possible."""
    count = min(count, total_frames)
    if count <= 0:
        return []
    if count == 1:
        return [total_frames // 2]
    return [round(position * (total_frames - 1) / (count - 1)) for position in range(count)]


def _parent_center_fields(parent_record: dict[str, Any]) -> list[tuple[str, list[int]]]:
    """Extract the explicit center lists emitted by the proof-pack runners."""
    meta = parent_record.get("selection_meta", {})
    fields: list[tuple[str, list[int]]] = []
    for name, source in (
        ("target_centers", "parent_target_center"),
        ("event_centers", "parent_event_center"),
        ("pivot_centers", "parent_pivot_center"),
    ):
        values = meta.get(name, [])
        if isinstance(values, dict):
            for label in sorted(values):
                fields.append((f"{source}:{label}", [int(value) for value in values[label]]))
        else:
            fields.append((source, [int(value) for value in values]))
    option_centers = meta.get("option_centers", {})
    if isinstance(option_centers, dict):
        for label in sorted(option_centers):
            fields.append(
                (f"parent_option_center:{label}", [int(value) for value in option_centers[label]])
            )
    return fields


def _source_priority(source: str) -> int:
    if source.startswith("parent_target_center"):
        return 0
    if source.startswith("parent_event_center"):
        return 1
    if source.startswith("parent_pivot_center"):
        return 2
    if source.startswith("parent_option_center"):
        return 3
    if source == "directional_target":
        return 4
    if source == "event_center":
        return 5
    if source == "pivot":
        return 6
    return 7


def rank_event_centers(parent_record: dict[str, Any], max_centers: int = 8) -> list[EventCenter]:
    """Rank explicit parent centers, falling back to non-context proof evidence.

    Parent proof packs retain relevance scores alongside their selected frames.
    Those scores provide a stable cross-center ranking; source and original
    parent order break score ties deterministically.
    """
    if max_centers <= 0:
        return []

    selected = parent_record.get("selected", [])
    selected_by_index = {
        int(item["frame_index"]): item for item in selected if "frame_index" in item
    }
    candidates: list[EventCenter] = []
    parent_rank = 0
    for source, indices in _parent_center_fields(parent_record):
        for frame_index in indices:
            item = selected_by_index.get(frame_index, {})
            score = item.get("score")
            candidates.append(
                EventCenter(
                    frame_index=frame_index,
                    score=float(score) if score is not None else None,
                    source=source,
                    parent_rank=parent_rank,
                )
            )
            parent_rank += 1

    excluded_sources = {
        "anchor",
        "uniform_global",
        "coverage_fill",
        "semantic_boundary",
        "bridge",
    }
    for item in selected:
        source = str(item.get("source", ""))
        if source in excluded_sources or source.endswith("_context"):
            continue
        score = item.get("score")
        candidates.append(
            EventCenter(
                frame_index=int(item["frame_index"]),
                score=float(score) if score is not None else None,
                source=source,
                parent_rank=parent_rank,
            )
        )
        parent_rank += 1

    ranked = sorted(
        candidates,
        key=lambda center: (
            -(center.score if center.score is not None else float("-inf")),
            _source_priority(center.source),
            center.parent_rank,
            center.frame_index,
        ),
    )
    unique: list[EventCenter] = []
    seen: set[int] = set()
    for center in ranked:
        if center.frame_index in seen:
            continue
        seen.add(center.frame_index)
        unique.append(center)
        if len(unique) == min(max_centers, 8):
            break
    return unique


def _clamp_frame_index(index: int, total_frames: int) -> int:
    return max(0, min(index, total_frames - 1))


def _fill_uniform_coverage(
    selected: dict[int, DenseBurstFrame],
    target_count: int,
    total_frames: int,
) -> None:
    """Fill residual budget at the middle of the largest remaining time gaps."""
    while len(selected) < target_count and len(selected) < total_frames:
        remaining = (index for index in range(total_frames) if index not in selected)
        chosen = max(
            remaining,
            key=lambda index: (
                min(abs(index - present) for present in selected),
                -index,
            ),
        )
        selected[chosen] = DenseBurstFrame(chosen, "coverage_fill")


def build_dense_burst_frame_plan(
    total_frames: int,
    fps: float,
    parent_record: dict[str, Any],
    max_frames: int = MAX_DENSE_BURST_FRAMES,
    event_centers: int = 8,
    burst_offsets_seconds: tuple[float, ...] = DEFAULT_BURST_OFFSETS_SECONDS,
) -> list[DenseBurstFrame]:
    """Construct a <=64-frame plan with immutable global anchors.

    Burst offsets are transformed through the video's native FPS rather than
    an assumed frame rate.  If configurable bursts overflow the budget, their
    parent rank and offset order determine the retained frames; anchors are
    never removed.  Any collisions or boundary clipping are filled by a
    deterministic coverage policy.
    """
    if max_frames > MAX_DENSE_BURST_FRAMES:
        raise ValueError(f"Dense bursts are capped at {MAX_DENSE_BURST_FRAMES} frames")
    if max_frames < min(GLOBAL_ANCHOR_COUNT, total_frames):
        raise ValueError("max_frames cannot be smaller than the retained global anchors")
    if total_frames <= 0 or fps <= 0:
        return []

    selected: dict[int, DenseBurstFrame] = {
        index: DenseBurstFrame(index, "global_anchor")
        for index in _uniform_indices(total_frames, GLOBAL_ANCHOR_COUNT)
    }
    anchors = set(selected)
    burst_candidates: dict[int, DenseBurstFrame] = {}
    for center_rank, center in enumerate(rank_event_centers(parent_record, event_centers)):
        for offset_seconds in burst_offsets_seconds:
            frame_index = _clamp_frame_index(
                round(center.frame_index + fps * offset_seconds), total_frames
            )
            if frame_index in anchors or frame_index in burst_candidates:
                continue
            burst_candidates[frame_index] = DenseBurstFrame(
                frame_index=frame_index,
                source="event_burst",
                center_frame_index=center.frame_index,
                offset_seconds=float(offset_seconds),
                center_rank=center_rank,
                center_score=center.score,
                center_source=center.source,
            )

    keep_bursts = max_frames - len(selected)
    ordered_bursts = sorted(
        burst_candidates.values(),
        key=lambda frame: (
            frame.center_rank,
            burst_offsets_seconds.index(frame.offset_seconds),
            frame.frame_index,
        ),
    )
    for frame in ordered_bursts[:keep_bursts]:
        selected[frame.frame_index] = frame

    _fill_uniform_coverage(selected, min(max_frames, total_frames), total_frames)
    return [selected[index] for index in sorted(selected)]


def dense_burst_fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "schema": DENSE_BURST_SCHEMA,
        "parent_proofpack": os.path.abspath(args.parent_proofpack),
        "global_anchors": GLOBAL_ANCHOR_COUNT,
        "event_centers": min(args.event_centers, 8),
        "burst_offsets_seconds": list(args.burst_offsets_seconds),
        "max_frames": args.max_frames,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def inference_fingerprint(args: argparse.Namespace, frame_fingerprint: str) -> str:
    payload = {
        "frame_fingerprint": frame_fingerprint,
        "model_type": args.model_type,
        "llm_model": args.llm_model,
        "backend": args.backend,
        "prompt_variant": args.prompt_variant,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def _video_metadata(video_path: str) -> tuple[float, int]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        return float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _parent_fingerprint(record: dict[str, Any]) -> str:
    return str(
        record.get("proofpack_fingerprint")
        or record.get("grounding_config_fingerprint")
        or ""
    )


def _frame_metadata_record(
    row_index: int,
    row: dict[str, Any],
    parent_record: dict[str, Any],
    parent_proofpack: str,
    fingerprint: str,
    fps: float,
    total_frames: int,
    selected: list[DenseBurstFrame],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "index": row_index,
        "sample_key": sample_key(row),
        "video_path": row.get("video_path", ""),
        "frame_metadata_schema": DENSE_BURST_SCHEMA,
        "strategy": "global_anchor_dense_bursts",
        "dense_burst_fingerprint": fingerprint,
        "parent_proofpack": os.path.abspath(parent_proofpack),
        "parent_proofpack_fingerprint": _parent_fingerprint(parent_record),
        "video_fps": fps,
        "total_video_frames": total_frames,
        "global_anchors": GLOBAL_ANCHOR_COUNT,
        "event_centers_requested": min(args.event_centers, 8),
        "burst_offsets_seconds": list(args.burst_offsets_seconds),
        "selected_frames": len(selected),
        "selected": [
            {
                "frame_index": frame.frame_index,
                "timestamp": round(frame.frame_index / fps, 3),
                "source": frame.source,
                "center": frame.center_frame_index,
                "center_frame_index": frame.center_frame_index,
                "offset_seconds": frame.offset_seconds,
                "center_rank": frame.center_rank,
                "center_score": frame.center_score,
                "center_source": frame.center_source,
            }
            for frame in selected
        ],
    }


def _valid_cached_frame_records(
    cached: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    parents: dict[str, dict[str, Any]],
    fingerprint: str,
    max_frames: int,
) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for index, record in enumerate(cached[: len(rows)]):
        row = rows[index]
        parent = parents[sample_key(row)]
        if int(record.get("index", -1)) != index:
            break
        if str(record.get("video_path", "")) != str(row.get("video_path", "")):
            break
        if str(record.get("sample_key", "")) != sample_key(row):
            break
        if str(record.get("dense_burst_fingerprint", "")) != fingerprint:
            break
        if str(record.get("parent_proofpack_fingerprint", "")) != _parent_fingerprint(parent):
            break
        if int(record.get("selected_frames", -1)) > max_frames:
            break
        valid.append(record)
    return valid


def _rewrite_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    from model import MODEL_TYPES

    parser = argparse.ArgumentParser(
        description="Generate LongQA predictions from global anchors and dense proof-pack bursts."
    )
    parser.add_argument(
        "--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--parent-proofpack", required=True)
    parser.add_argument("--frame-metadata-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-resume-frames", action="store_true")
    parser.add_argument("--no-resume-predictions", action="store_true")
    parser.add_argument("--event-centers", type=int, default=8)
    parser.add_argument(
        "--burst-offset-seconds",
        type=float,
        nargs="+",
        default=list(DEFAULT_BURST_OFFSETS_SECONDS),
    )
    parser.add_argument("--max-frames", type=int, default=MAX_DENSE_BURST_FRAMES)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="baseline")
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="qwen")
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    import time

    from model import create_model, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.max_frames > MAX_DENSE_BURST_FRAMES:
        raise ValueError(f"--max-frames must be <= {MAX_DENSE_BURST_FRAMES}")
    if args.event_centers < 0:
        raise ValueError("--event-centers must be non-negative")

    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    parent_proofpack = _resolve_path(args.parent_proofpack)
    frame_metadata_output = _resolve_path(args.frame_metadata_output)
    output_path = _resolve_path(args.output)
    eval_output = _resolve_path(args.eval_output) if args.eval_output else None
    fingerprint = dense_burst_fingerprint(args)
    generation_fingerprint = inference_fingerprint(args, fingerprint)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    parents = index_row_aligned_metadata(
        load_jsonl(parent_proofpack), rows, "parent proof pack"
    )

    os.makedirs(os.path.dirname(frame_metadata_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(
        "Dense-burst config: "
        f"anchors={GLOBAL_ANCHOR_COUNT}, event_centers={min(args.event_centers, 8)}, "
        f"offsets={args.burst_offsets_seconds}, max_frames={args.max_frames}, "
        f"fingerprint={fingerprint}"
    )
    started = time.time()

    cached = [] if args.no_resume_frames else _load_jsonl_if_exists(frame_metadata_output)
    records = _valid_cached_frame_records(
        cached, rows, parents, fingerprint, args.max_frames
    )
    if records:
        print(f"Resuming frame selection from {len(records)}/{len(rows)} rows")
    if len(records) != len(cached):
        _rewrite_jsonl(frame_metadata_output, records)
    if len(records) < len(rows):
        with open(frame_metadata_output, "a" if records else "w") as handle:
            for row_index, row in enumerate(rows[len(records) :], start=len(records)):
                video_path = os.path.join(video_folder, str(row["video_path"]))
                fps, total_frames = _video_metadata(video_path)
                selected = build_dense_burst_frame_plan(
                    total_frames,
                    fps,
                    parents[sample_key(row)],
                    max_frames=args.max_frames,
                    event_centers=args.event_centers,
                    burst_offsets_seconds=tuple(args.burst_offsets_seconds),
                )
                record = _frame_metadata_record(
                    row_index,
                    row,
                    parents[sample_key(row)],
                    parent_proofpack,
                    fingerprint,
                    fps,
                    total_frames,
                    selected,
                    args,
                )
                records.append(record)
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(f"  Frame selection progress: {row_index + 1}/{len(rows)}")

    existing = [] if args.no_resume_predictions else _load_jsonl_if_exists(output_path)
    prediction_start = 0
    for index, prediction in enumerate(existing[: len(rows)]):
        if sample_key(prediction) != sample_key(rows[index]):
            break
        if str(prediction.get("dense_burst_inference_fingerprint", "")) != generation_fingerprint:
            break
        if not str(prediction.get("mcq_answer", "")).strip():
            break
        prediction_start += 1
    if prediction_start:
        print(f"Resuming generation from {prediction_start}/{len(rows)} rows")
    if prediction_start != len(existing):
        _rewrite_jsonl(output_path, existing[:prediction_start])

    if prediction_start < len(rows):
        model = create_model(
            args.model_type,
            args.llm_model,
            backend=args.backend,
            tp_size=args.tp,
            concurrency=args.concurrency,
            max_frames=args.max_frames,
        )
        reset_prompt_token_stats()
        with model, open(output_path, "a" if prediction_start else "w") as handle:
            for row_index, (row, record) in enumerate(
                zip(rows[prediction_start:], records[prediction_start:]),
                start=prediction_start,
            ):
                video_path = os.path.join(video_folder, str(row["video_path"]))
                frame_indices = [int(item["frame_index"]) for item in record["selected"]]
                frames = extract_frames_by_indices(video_path, frame_indices)
                prompt = build_longqa_prompt(
                    row["question"], row["mcq_options"], prompt_variant=args.prompt_variant
                )
                response = model.generate(
                    frames, [{"role": "user", "content": prompt}], max_new_tokens=16
                )
                prediction = build_prediction_row(row, response, prompt_variant=args.prompt_variant)
                prediction["dense_burst_fingerprint"] = fingerprint
                prediction["dense_burst_inference_fingerprint"] = generation_fingerprint
                prediction["dense_burst_frame_count"] = len(frames)
                handle.write(json.dumps(prediction) + "\n")
                handle.flush()
                print(f"  Generation progress: {row_index + 1}/{len(rows)}")
        _print_context_summary(summarize_prompt_token_stats())

    print(f"Predictions written to {output_path}")
    print(f"Dense-burst frame metadata written to {frame_metadata_output}")
    print(f"Runtime seconds: {time.time() - started:.0f}")
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
