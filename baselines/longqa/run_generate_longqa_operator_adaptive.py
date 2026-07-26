#!/usr/bin/env python3
"""Run Qwen3.5 on operator-adaptive frame packs derived from a proof pack.

The parent proof pack supplies already-ranked retrieval evidence.  This runner
only turns that evidence into a fixed, auditable 64-frame pack; retrieval and
the shared proof-pack implementation remain unchanged.
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
from run_generate_longqa_proofpack import TemporalProgram, compile_temporal_program

logger = logging.getLogger(__name__)

OPERATOR_ADAPTIVE_SCHEMA = 1
MAX_OPERATOR_ADAPTIVE_FRAMES = 64
GLOBAL_ANCHOR_COUNT = 16
EDGE_ANCHOR_COUNT = 24
MAX_PIVOT_CENTERS = 2
MAX_TARGET_CENTERS = 6
MAX_EDGE_CENTERS = 5
MAX_STATE_CHANGE_CENTERS = 4

AFTER_PIVOT_OFFSETS_SECONDS = (-2.0, -1.0, -0.33, 0.0, 0.33, 1.0)
AFTER_TARGET_OFFSETS_SECONDS = (-0.33, 0.0, 0.33, 1.0, 2.0, 4.0)
BEFORE_PIVOT_OFFSETS_SECONDS = (-1.0, -0.33, 0.0, 0.33, 1.0, 2.0)
BEFORE_TARGET_OFFSETS_SECONDS = (-4.0, -2.0, -1.0, -0.33, 0.0, 0.33)
FIRST_OFFSETS_SECONDS = (-0.33, 0.0, 0.33, 1.0, 2.0, 4.0, 6.0, 10.0)
LAST_OFFSETS_SECONDS = (-10.0, -6.0, -4.0, -2.0, -1.0, -0.33, 0.0, 0.33)
STATE_CHANGE_PRE_OFFSETS_SECONDS = (-8.0, -4.0, -2.0, -1.0, -0.33, 0.0)
STATE_CHANGE_POST_OFFSETS_SECONDS = (0.33, 1.0, 2.0, 4.0, 6.0, 8.0)


@dataclass(frozen=True)
class ParentCenter:
    frame_index: int
    score: float | None
    source: str
    role: str
    parent_rank: int


@dataclass(frozen=True)
class AdaptiveFrame:
    frame_index: int
    source: str
    policy: str
    is_global_anchor: bool = False
    center_frame_index: int | None = None
    center_role: str | None = None
    center_rank: int | None = None
    center_score: float | None = None
    center_source: str | None = None
    offset_seconds: float | None = None


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _load_jsonl_if_exists(path: str) -> list[dict[str, Any]]:
    return load_jsonl(path) if os.path.exists(path) else []


def _uniform_indices(total_frames: int, count: int) -> list[int]:
    """Return distinct, end-inclusive uniform frame indices."""
    count = min(total_frames, count)
    if count <= 0:
        return []
    if count == 1:
        return [total_frames // 2]
    return [round(position * (total_frames - 1) / (count - 1)) for position in range(count)]


def _clamp_frame_index(frame_index: int, total_frames: int) -> int:
    return max(0, min(frame_index, total_frames - 1))


def _metadata_center_values(value: object) -> list[tuple[str, int]]:
    if isinstance(value, dict):
        pairs: list[tuple[str, int]] = []
        for label in sorted(value):
            pairs.extend((str(label), int(frame_index)) for frame_index in value[label])
        return pairs
    return [("", int(frame_index)) for frame_index in value or []]


def _score_for_index(selected_by_index: dict[int, dict[str, Any]], frame_index: int) -> float | None:
    score = selected_by_index.get(frame_index, {}).get("score")
    return float(score) if score is not None else None


def _source_role(source: str) -> str | None:
    lowered = source.lower()
    if "pivot" in lowered:
        return "pivot"
    if "target" in lowered:
        return "target"
    if "event" in lowered or "option" in lowered:
        return "event"
    if any(
        token in lowered
        for token in ("anchor", "uniform", "coverage", "bridge", "semantic_boundary")
    ):
        return None
    return "evidence"


def _parent_centers(parent_record: dict[str, Any]) -> list[ParentCenter]:
    """Collect named parent centers first, then non-global selected evidence."""
    selected = parent_record.get("selected", [])
    selected_by_index = {
        int(item["frame_index"]): item for item in selected if "frame_index" in item
    }
    meta = parent_record.get("selection_meta", {})
    candidates: list[ParentCenter] = []
    parent_rank = 0

    for field, role in (
        ("pivot_centers", "pivot"),
        ("target_centers", "target"),
        ("event_centers", "event"),
        ("option_centers", "event"),
    ):
        for label, frame_index in _metadata_center_values(meta.get(field, [])):
            candidates.append(
                ParentCenter(
                    frame_index=frame_index,
                    score=_score_for_index(selected_by_index, frame_index),
                    source=f"parent_{field}:{label}" if label else f"parent_{field}",
                    role=role,
                    parent_rank=parent_rank,
                )
            )
            parent_rank += 1

    for item in selected:
        source = str(item.get("source", ""))
        role = _source_role(source)
        if role is None:
            continue
        score = item.get("score")
        candidates.append(
            ParentCenter(
                frame_index=int(item["frame_index"]),
                score=float(score) if score is not None else None,
                source=source or "parent_selected",
                role=role,
                parent_rank=parent_rank,
            )
        )
        parent_rank += 1
    return candidates


def _role_priority(role: str) -> int:
    return {"pivot": 0, "target": 1, "event": 2, "evidence": 3}[role]


def rank_parent_centers(
    parent_record: dict[str, Any],
    roles: tuple[str, ...] = ("pivot", "target", "event", "evidence"),
    max_centers: int | None = None,
) -> list[ParentCenter]:
    """Rank parent evidence by score, source role, and saved parent order."""
    role_set = set(roles)
    ordered = sorted(
        (center for center in _parent_centers(parent_record) if center.role in role_set),
        key=lambda center: (
            -(center.score if center.score is not None else float("-inf")),
            _role_priority(center.role),
            center.parent_rank,
            center.frame_index,
        ),
    )
    unique: list[ParentCenter] = []
    seen: set[int] = set()
    for center in ordered:
        if center.frame_index in seen:
            continue
        seen.add(center.frame_index)
        unique.append(center)
        if max_centers is not None and len(unique) >= max_centers:
            break
    return unique


def _policy_for_operator(operator: str) -> str:
    return {
        "GLOBAL": "uniform64",
        "AFTER": "after_directional_bursts",
        "BEFORE": "before_directional_bursts",
        "FIRST": "first_edge_neighborhoods",
        "LAST": "last_edge_neighborhoods",
        "STATE_CHANGE": "state_change_pre_post_bursts",
    }[operator]


def _add_anchor_frames(
    selected: dict[int, AdaptiveFrame],
    total_frames: int,
    count: int,
    policy: str,
    source: str,
) -> None:
    for frame_index in _uniform_indices(total_frames, count):
        selected[frame_index] = AdaptiveFrame(
            frame_index=frame_index,
            source=source,
            policy=policy,
            is_global_anchor=True,
        )


def _add_center_offsets(
    selected: dict[int, AdaptiveFrame],
    total_frames: int,
    fps: float,
    target_count: int,
    policy: str,
    source: str,
    centers: list[ParentCenter],
    offsets_seconds: tuple[float, ...],
) -> None:
    for center_rank, center in enumerate(centers):
        for offset_seconds in offsets_seconds:
            frame_index = _clamp_frame_index(
                round(center.frame_index + fps * offset_seconds), total_frames
            )
            existing = selected.get(frame_index)
            frame = AdaptiveFrame(
                frame_index=frame_index,
                source=source,
                policy=policy,
                is_global_anchor=existing.is_global_anchor if existing else False,
                center_frame_index=center.frame_index,
                center_role=center.role,
                center_rank=center_rank,
                center_score=center.score,
                center_source=center.source,
                offset_seconds=float(offset_seconds),
            )
            if existing is None and len(selected) >= target_count:
                continue
            if existing is None or existing.is_global_anchor:
                selected[frame_index] = frame


def _fill_uniform_coverage(
    selected: dict[int, AdaptiveFrame],
    target_count: int,
    total_frames: int,
    policy: str,
) -> None:
    """Use a deterministic, dense uniform candidate grid for residual coverage."""
    candidate_count = min(total_frames, max(target_count, target_count * 4))
    for frame_index in _uniform_indices(total_frames, candidate_count):
        if len(selected) >= target_count:
            return
        if frame_index not in selected:
            selected[frame_index] = AdaptiveFrame(
                frame_index=frame_index,
                source="coverage_fill",
                policy=policy,
            )
    for frame_index in range(total_frames):
        if len(selected) >= target_count:
            return
        if frame_index not in selected:
            selected[frame_index] = AdaptiveFrame(
                frame_index=frame_index,
                source="coverage_fill",
                policy=policy,
            )


def _directional_centers(parent_record: dict[str, Any]) -> tuple[list[ParentCenter], list[ParentCenter]]:
    pivots = rank_parent_centers(parent_record, ("pivot",), MAX_PIVOT_CENTERS)
    targets = rank_parent_centers(parent_record, ("target", "event"), MAX_TARGET_CENTERS)
    generic = rank_parent_centers(parent_record, ("evidence",), MAX_TARGET_CENTERS)
    if not targets:
        targets = generic
    if not pivots:
        pivots = targets[:MAX_PIVOT_CENTERS]
    if not targets:
        targets = pivots[:MAX_TARGET_CENTERS]
    return pivots, targets


def _edge_centers(parent_record: dict[str, Any], operator: str) -> list[ParentCenter]:
    centers = rank_parent_centers(parent_record, max_centers=None)
    if operator == "FIRST":
        return sorted(
            centers,
            key=lambda center: (center.frame_index, center.parent_rank, center.source),
        )[:MAX_EDGE_CENTERS]
    return sorted(
        centers,
        key=lambda center: (-center.frame_index, center.parent_rank, center.source),
    )[:MAX_EDGE_CENTERS]


def build_operator_adaptive_frame_plan(
    total_frames: int,
    fps: float,
    parent_record: dict[str, Any],
    program: TemporalProgram,
    max_frames: int = MAX_OPERATOR_ADAPTIVE_FRAMES,
) -> list[AdaptiveFrame]:
    """Build a chronological, unique, bounded plan for one temporal operator."""
    if total_frames <= 0 or fps <= 0:
        return []
    if max_frames > MAX_OPERATOR_ADAPTIVE_FRAMES:
        raise ValueError(f"Operator-adaptive packs are capped at {MAX_OPERATOR_ADAPTIVE_FRAMES}")
    if max_frames <= 0:
        return []

    target_count = min(max_frames, total_frames)
    operator = program.operator
    policy = _policy_for_operator(operator)
    selected: dict[int, AdaptiveFrame] = {}

    if operator == "GLOBAL":
        _add_anchor_frames(selected, total_frames, target_count, policy, "uniform_global")
        return [selected[frame_index] for frame_index in sorted(selected)]

    anchor_count = EDGE_ANCHOR_COUNT if operator in {"FIRST", "LAST"} else GLOBAL_ANCHOR_COUNT
    if target_count < min(anchor_count, total_frames):
        raise ValueError(f"{operator} requires at least {anchor_count} frame slots for global anchors")
    anchor_source = "broad_uniform_anchor" if operator in {"FIRST", "LAST"} else "global_anchor"
    _add_anchor_frames(selected, total_frames, anchor_count, policy, anchor_source)

    if operator == "AFTER":
        pivots, targets = _directional_centers(parent_record)
        _add_center_offsets(
            selected,
            total_frames,
            fps,
            target_count,
            policy,
            "after_pivot_burst",
            pivots,
            AFTER_PIVOT_OFFSETS_SECONDS,
        )
        _add_center_offsets(
            selected,
            total_frames,
            fps,
            target_count,
            policy,
            "after_target_burst",
            targets,
            AFTER_TARGET_OFFSETS_SECONDS,
        )
    elif operator == "BEFORE":
        pivots, targets = _directional_centers(parent_record)
        _add_center_offsets(
            selected,
            total_frames,
            fps,
            target_count,
            policy,
            "before_pivot_burst",
            pivots,
            BEFORE_PIVOT_OFFSETS_SECONDS,
        )
        _add_center_offsets(
            selected,
            total_frames,
            fps,
            target_count,
            policy,
            "before_target_burst",
            targets,
            BEFORE_TARGET_OFFSETS_SECONDS,
        )
    elif operator in {"FIRST", "LAST"}:
        centers = _edge_centers(parent_record, operator)
        _add_center_offsets(
            selected,
            total_frames,
            fps,
            target_count,
            policy,
            f"{operator.lower()}_event_neighborhood",
            centers,
            FIRST_OFFSETS_SECONDS if operator == "FIRST" else LAST_OFFSETS_SECONDS,
        )
    elif operator == "STATE_CHANGE":
        centers = rank_parent_centers(parent_record, max_centers=MAX_STATE_CHANGE_CENTERS)
        _add_center_offsets(
            selected,
            total_frames,
            fps,
            target_count,
            policy,
            "state_change_pre_burst",
            centers,
            STATE_CHANGE_PRE_OFFSETS_SECONDS,
        )
        _add_center_offsets(
            selected,
            total_frames,
            fps,
            target_count,
            policy,
            "state_change_post_burst",
            centers,
            STATE_CHANGE_POST_OFFSETS_SECONDS,
        )

    _fill_uniform_coverage(selected, target_count, total_frames, policy)
    return [selected[frame_index] for frame_index in sorted(selected)]


def operator_adaptive_fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "schema": OPERATOR_ADAPTIVE_SCHEMA,
        "parent_proofpack": os.path.abspath(args.parent_proofpack),
        "max_frames": args.max_frames,
        "global_anchor_count": GLOBAL_ANCHOR_COUNT,
        "edge_anchor_count": EDGE_ANCHOR_COUNT,
        "max_pivot_centers": MAX_PIVOT_CENTERS,
        "max_target_centers": MAX_TARGET_CENTERS,
        "max_edge_centers": MAX_EDGE_CENTERS,
        "max_state_change_centers": MAX_STATE_CHANGE_CENTERS,
        "after_pivot_offsets": AFTER_PIVOT_OFFSETS_SECONDS,
        "after_target_offsets": AFTER_TARGET_OFFSETS_SECONDS,
        "before_pivot_offsets": BEFORE_PIVOT_OFFSETS_SECONDS,
        "before_target_offsets": BEFORE_TARGET_OFFSETS_SECONDS,
        "first_offsets": FIRST_OFFSETS_SECONDS,
        "last_offsets": LAST_OFFSETS_SECONDS,
        "state_change_pre_offsets": STATE_CHANGE_PRE_OFFSETS_SECONDS,
        "state_change_post_offsets": STATE_CHANGE_POST_OFFSETS_SECONDS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def inference_fingerprint(args: argparse.Namespace, frame_fingerprint: str) -> str:
    payload = {
        "frame_fingerprint": frame_fingerprint,
        "model_type": args.model_type,
        "llm_model": args.llm_model,
        "backend": args.backend,
        "tp": args.tp,
        "concurrency": args.concurrency,
        "prompt_variant": args.prompt_variant,
        "max_new_tokens": args.max_new_tokens,
        "thinking": False,
        "qwen_min_pixels": os.environ.get("QWEN_MIN_PIXELS", "784"),
        "qwen_max_pixels": os.environ.get("QWEN_MAX_PIXELS", "50176"),
        "vllm_max_model_len": os.environ.get("VLLM_QWEN_MAX_MODEL_LEN", "16384"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def _video_metadata(video_path: str) -> tuple[float, int]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return fps, total_frames


def _parent_fingerprint(record: dict[str, Any]) -> str:
    return str(record.get("proofpack_fingerprint") or record.get("grounding_config_fingerprint") or "")


def _frame_metadata_record(
    row_index: int,
    row: dict[str, Any],
    parent_record: dict[str, Any],
    parent_proofpack: str,
    fingerprint: str,
    fps: float,
    total_frames: int,
    program: TemporalProgram,
    selected: list[AdaptiveFrame],
) -> dict[str, Any]:
    policy = _policy_for_operator(program.operator)
    return {
        "index": row_index,
        "sample_key": sample_key(row),
        "video_path": row.get("video_path", ""),
        "frame_metadata_schema": OPERATOR_ADAPTIVE_SCHEMA,
        "strategy": "operator_adaptive_proofpack_bursts",
        "operator_adaptive_fingerprint": fingerprint,
        "parent_proofpack": os.path.abspath(parent_proofpack),
        "parent_proofpack_fingerprint": _parent_fingerprint(parent_record),
        "video_fps": fps,
        "total_video_frames": total_frames,
        "operator": program.operator,
        "policy": policy,
        "temporal_program": {
            "operator": program.operator,
            "pivot": program.pivot,
            "direction": program.direction,
            "target": program.target,
        },
        "selected_frames": len(selected),
        "selected": [
            {
                "frame_index": frame.frame_index,
                "timestamp": round(frame.frame_index / fps, 3),
                "source": frame.source,
                "policy": frame.policy,
                "is_global_anchor": frame.is_global_anchor,
                "center_frame_index": frame.center_frame_index,
                "center_role": frame.center_role,
                "center_rank": frame.center_rank,
                "center_score": frame.center_score,
                "center_source": frame.center_source,
                "offset_seconds": frame.offset_seconds,
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
        if str(record.get("operator_adaptive_fingerprint", "")) != fingerprint:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from model import MODEL_TYPES

    parser = argparse.ArgumentParser(
        description="Generate LongQA predictions from operator-adaptive proof-pack bursts."
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
    parser.add_argument("--max-frames", type=int, default=MAX_OPERATOR_ADAPTIVE_FRAMES)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="baseline")
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="qwen")
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args(argv)


def main() -> None:
    import time

    from model import create_model, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.max_frames > MAX_OPERATOR_ADAPTIVE_FRAMES:
        raise ValueError(f"--max-frames must be <= {MAX_OPERATOR_ADAPTIVE_FRAMES}")

    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    parent_proofpack = _resolve_path(args.parent_proofpack)
    frame_metadata_output = _resolve_path(args.frame_metadata_output)
    output_path = _resolve_path(args.output)
    eval_output = _resolve_path(args.eval_output) if args.eval_output else None
    frame_fingerprint = operator_adaptive_fingerprint(args)
    generation_fingerprint = inference_fingerprint(args, frame_fingerprint)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    parents = index_row_aligned_metadata(
        load_jsonl(parent_proofpack), rows, "parent proof pack"
    )

    os.makedirs(os.path.dirname(frame_metadata_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(
        "Operator-adaptive config: "
        f"max_frames={args.max_frames}, frame_fingerprint={frame_fingerprint}, "
        f"inference_fingerprint={generation_fingerprint}"
    )
    started = time.time()

    cached = [] if args.no_resume_frames else _load_jsonl_if_exists(frame_metadata_output)
    records = _valid_cached_frame_records(
        cached, rows, parents, frame_fingerprint, args.max_frames
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
                program = compile_temporal_program(row["question"])
                selected = build_operator_adaptive_frame_plan(
                    total_frames,
                    fps,
                    parents[sample_key(row)],
                    program,
                    max_frames=args.max_frames,
                )
                record = _frame_metadata_record(
                    row_index,
                    row,
                    parents[sample_key(row)],
                    parent_proofpack,
                    frame_fingerprint,
                    fps,
                    total_frames,
                    program,
                    selected,
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
        if str(prediction.get("operator_adaptive_inference_fingerprint", "")) != generation_fingerprint:
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
                    frames,
                    [{"role": "user", "content": prompt}],
                    max_new_tokens=args.max_new_tokens,
                )
                prediction = build_prediction_row(row, response, prompt_variant=args.prompt_variant)
                prediction["operator_adaptive_frame_fingerprint"] = frame_fingerprint
                prediction["operator_adaptive_inference_fingerprint"] = generation_fingerprint
                prediction["operator_adaptive_frame_count"] = len(frames)
                handle.write(json.dumps(prediction) + "\n")
                handle.flush()
                print(f"  Generation progress: {row_index + 1}/{len(rows)}")
        _print_context_summary(summarize_prompt_token_stats())

    print(f"Predictions written to {output_path}")
    print(f"Operator-adaptive frame metadata written to {frame_metadata_output}")
    print(f"Runtime seconds: {time.time() - started:.0f}")
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
