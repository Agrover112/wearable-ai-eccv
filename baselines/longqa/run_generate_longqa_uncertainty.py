#!/usr/bin/env python3
"""Run Qwen-native uncertainty-guided EgoLongQA experiments.

The scorer uses the lower bound of next-token Shannon entropy from vLLM's
top-K log-probabilities. The unobserved tail is represented as one aggregate
outcome, so this is explicitly an approximation of full-vocabulary UG entropy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from longqa_utils import (
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    index_row_aligned_metadata,
    sample_key,
)
from run_generate_longqa_grounded import (
    CandidateFrame,
    _run_eval,
    extract_frames_by_indices,
    load_jsonl,
)
from run_generate_longqa_object_hints import (
    _top_detections,
    choose_base_frames,
    choose_detail_frames,
    crop_detection,
)
from run_generate_longqa_proofpack import (
    baseline_uniform_indices,
    compile_temporal_program,
    resize_frame_to_max_pixels,
    select_temporal_pivot_pack,
)
from run_generate_longqa_tcot import (
    build_final_pack,
    select_with_qwen,
    selection_fingerprint as tcot_selection_fingerprint,
)


SCHEMA_VERSION = 1
PROMPT_VERSION = "ug-qopts-v1"
MODES = (
    "segmented",
    "temporal_pivot",
    "short_window",
    "object_crops",
    "disagreement_router",
    "dynamic_tcot",
)


def _resolve_path(path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _video_metadata(video_path: str) -> tuple[float, int]:
    import cv2

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            return 0.0, 0
        return (
            float(capture.get(cv2.CAP_PROP_FPS)),
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload) + "\n")
    os.replace(temporary, path)


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _answer(row: dict[str, Any]) -> str:
    return str(row.get("mcq_answer_parsed") or row.get("mcq_answer", "")).strip().upper()


def build_entropy_prompt(row: dict[str, Any]) -> str:
    return (
        "Use only the supplied chronological visual evidence. Answer the "
        "multiple-choice question. Return only A, B, C, or D.\n\n"
        f"Question: {row['question']}\n\nOptions:\n{row['mcq_options']}"
    )


def build_presence_prompt(event: str) -> str:
    return (
        "Decide whether the supplied visual evidence visibly contains the "
        "specified event. Do not infer it from general plausibility.\n\n"
        f"Event: {event}\n\nA. Yes\nB. No\n\nReturn only A or B."
    )


def _compact_score(score: dict[str, Any]) -> dict[str, Any]:
    compact = dict(score)
    distribution = list(compact.get("distribution", []))
    compact["letter_probabilities"] = {
        letter: max(
            [
                math.exp(float(item["logprob"]))
                for item in distribution
                if str(item.get("token", "")).strip().upper() == letter
            ],
            default=0.0,
        )
        for letter in "ABCD"
    }
    compact["distribution"] = distribution[:20]
    compact["distribution_entries_cached"] = min(20, len(distribution))
    compact["distribution_truncated_for_cache"] = len(distribution) > 20
    return compact


def _letter_probability(score: dict[str, Any], letter: str) -> float:
    wanted = letter.strip().upper()
    cached = score.get("letter_probabilities", {})
    if wanted in cached:
        return float(cached[wanted])
    values = [
        math.exp(float(item["logprob"]))
        for item in score.get("distribution", [])
        if str(item.get("token", "")).strip().upper() == wanted
    ]
    return max(values, default=0.0)


def _collection_fingerprint(
    args: argparse.Namespace,
    prompt: str,
    descriptors: list[list[int]],
) -> str:
    payload = {
        "schema": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": args.llm_model,
        "top_logprobs": args.top_logprobs,
        "scoring_max_pixels": args.scoring_max_pixels,
        "prompt": prompt,
        "descriptors": descriptors,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def score_collections(
    model: Any,
    row: dict[str, Any],
    descriptors: list[list[int]],
    collections: list[list[object]],
    prompt: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], bool]:
    if len(descriptors) != len(collections):
        raise ValueError("score descriptors and visual collections must have equal lengths")
    fingerprint = _collection_fingerprint(args, prompt, descriptors)
    digest = hashlib.sha1(sample_key(row).encode()).hexdigest()
    cache_path = Path(args.score_cache_dir) / fingerprint / f"{digest}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if (
            cached.get("fingerprint") == fingerprint
            and cached.get("sample_key") == sample_key(row)
            and cached.get("descriptors") == descriptors
        ):
            return list(cached["scores"]), True

    messages = [[{"role": "user", "content": prompt}] for _ in collections]
    scores = model.score_token_uncertainty_batch(
        collections,
        messages,
        top_logprobs=args.top_logprobs,
    )
    compact = [_compact_score(score) for score in scores]
    _write_atomic(
        cache_path,
        {
            "schema": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "sample_key": sample_key(row),
            "descriptors": descriptors,
            "scores": compact,
        },
    )
    return compact, False


def _frame_collections(
    video_path: str,
    indices: list[int],
    max_pixels: int,
) -> list[list[object]]:
    frames = extract_frames_by_indices(video_path, indices)
    if len(frames) != len(indices):
        raise RuntimeError(
            f"Extracted {len(frames)}/{len(indices)} requested frames from {video_path}"
        )
    return [[resize_frame_to_max_pixels(frame, max_pixels)] for frame in frames]


def _split_positions(length: int, segments: int) -> list[list[int]]:
    segments = min(max(1, segments), max(1, length))
    boundaries = [round(index * length / segments) for index in range(segments + 1)]
    return [
        list(range(boundaries[index], boundaries[index + 1]))
        for index in range(segments)
    ]


def select_segmented_positions(
    scores: list[dict[str, Any]],
    segments: int,
    per_segment: int,
) -> list[int]:
    selected: list[int] = []
    for positions in _split_positions(len(scores), segments):
        ranked = sorted(
            positions,
            key=lambda index: (
                float(scores[index]["entropy_lower_bound"]),
                index,
            ),
        )
        selected.extend(ranked[:per_segment])
    return sorted(set(selected))


def fill_with_uniform(
    priority_indices: list[int],
    total_frames: int,
    uniform_quota: int,
    max_frames: int,
) -> list[int]:
    chosen: list[int] = []
    seen: set[int] = set()
    for frame_index in priority_indices:
        if frame_index not in seen:
            seen.add(frame_index)
            chosen.append(frame_index)
        if len(chosen) >= max_frames:
            return sorted(chosen)
    pools = [
        baseline_uniform_indices(total_frames, uniform_quota),
        baseline_uniform_indices(total_frames, max_frames * 4),
        baseline_uniform_indices(total_frames, max_frames),
    ]
    for pool in pools:
        for frame_index in pool:
            if frame_index not in seen:
                seen.add(frame_index)
                chosen.append(frame_index)
            if len(chosen) >= max_frames:
                return sorted(chosen)
    return sorted(chosen)


def _score_single_frames(
    model: Any,
    row: dict[str, Any],
    video_path: str,
    candidate_indices: list[int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], bool]:
    return score_collections(
        model,
        row,
        [[index] for index in candidate_indices],
        _frame_collections(video_path, candidate_indices, args.scoring_max_pixels),
        build_entropy_prompt(row),
        args,
    )


def select_segmented_pack(
    model: Any,
    row: dict[str, Any],
    video_path: str,
    total_frames: int,
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, Any]]:
    candidate_indices = baseline_uniform_indices(total_frames, args.candidate_frames)
    scores, cache_hit = _score_single_frames(
        model, row, video_path, candidate_indices, args
    )
    positions = select_segmented_positions(
        scores, args.segments, args.per_segment
    )
    selected = [candidate_indices[position] for position in positions]
    final_indices = fill_with_uniform(
        selected,
        total_frames,
        args.uniform_quota,
        args.max_frames,
    )
    return final_indices, {
        "candidate_indices": candidate_indices,
        "selected_positions": positions,
        "selected_indices": selected,
        "scores": scores,
        "score_cache_hit": cache_hit,
        "uniform_quota": args.uniform_quota,
    }


def select_uncertainty_pivot_pack(
    model: Any,
    row: dict[str, Any],
    video_path: str,
    fps: float,
    total_frames: int,
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, Any]]:
    import numpy as np

    candidate_indices = baseline_uniform_indices(total_frames, args.candidate_frames)
    target_scores_raw, target_cache_hit = _score_single_frames(
        model, row, video_path, candidate_indices, args
    )
    target_scores = [
        -float(score["entropy_lower_bound"]) for score in target_scores_raw
    ]
    program = compile_temporal_program(row["question"])
    pivot_event = program.pivot or str(row["question"])
    pivot_prompt = build_presence_prompt(pivot_event)
    pivot_scores_raw, pivot_cache_hit = score_collections(
        model,
        row,
        [[index] for index in candidate_indices],
        _frame_collections(video_path, candidate_indices, args.scoring_max_pixels),
        pivot_prompt,
        args,
    )
    pivot_scores = [
        _letter_probability(score, "A") - _letter_probability(score, "B")
        for score in pivot_scores_raw
    ]
    candidates = [
        CandidateFrame(
            index=frame_index,
            timestamp=frame_index / max(fps, 1e-6),
            image=None,
        )
        for frame_index in candidate_indices
    ]
    selected, selector_meta = select_temporal_pivot_pack(
        candidates,
        pivot_scores,
        target_scores,
        np.zeros((len(candidates), 1), dtype=np.float32),
        program,
        args.pivot_centers,
        args.target_centers,
        args.eventlet_radius,
        args.anchor_k,
        args.bridge_k,
        args.max_frames,
        args.temporal_nms_seconds,
        "uniform_coverage",
    )
    return [item.candidate.index for item in selected], {
        "candidate_indices": candidate_indices,
        "target_scores": target_scores_raw,
        "pivot_scores": pivot_scores_raw,
        "pivot_brc": pivot_scores,
        "target_cache_hit": target_cache_hit,
        "pivot_cache_hit": pivot_cache_hit,
        **selector_meta,
    }


def select_short_window_pack(
    model: Any,
    row: dict[str, Any],
    video_path: str,
    total_frames: int,
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, Any]]:
    candidate_indices = baseline_uniform_indices(total_frames, args.candidate_frames)
    half = args.window_size // 2
    starts = list(
        range(0, max(1, len(candidate_indices) - args.window_size + 1), args.window_stride)
    )
    final_start = max(0, len(candidate_indices) - args.window_size)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    position_windows = [
        list(range(start, min(len(candidate_indices), start + args.window_size)))
        for start in starts
    ]
    descriptors = [
        [candidate_indices[position] for position in positions]
        for positions in position_windows
    ]
    unique_indices = sorted({index for descriptor in descriptors for index in descriptor})
    images = extract_frames_by_indices(video_path, unique_indices)
    if len(images) != len(unique_indices):
        raise RuntimeError("Could not extract every short-window candidate frame")
    image_by_index = dict(zip(unique_indices, images))
    collections = [
        [
            resize_frame_to_max_pixels(image_by_index[index], args.scoring_max_pixels)
            for index in descriptor
        ]
        for descriptor in descriptors
    ]
    scores, cache_hit = score_collections(
        model,
        row,
        descriptors,
        collections,
        build_entropy_prompt(row),
        args,
    )
    selected_windows = select_segmented_positions(scores, args.segments, 1)
    priority: list[int] = []
    for window_position in selected_windows:
        positions = position_windows[window_position]
        center = positions[min(half, len(positions) - 1)]
        for position in range(max(0, center - 1), min(len(candidate_indices), center + 2)):
            priority.append(candidate_indices[position])
    final_indices = fill_with_uniform(
        priority, total_frames, args.uniform_quota, args.max_frames
    )
    return final_indices, {
        "candidate_indices": candidate_indices,
        "window_descriptors": descriptors,
        "window_scores": scores,
        "selected_window_positions": selected_windows,
        "priority_indices": sorted(set(priority)),
        "score_cache_hit": cache_hit,
    }


def _index_proofpack(
    proofpack_path: str,
    reference_path: str,
) -> dict[str, dict[str, Any]]:
    reference = load_jsonl(reference_path)
    records = load_jsonl(proofpack_path)
    return index_row_aligned_metadata(records, reference, "uncertainty proof pack")


def _index_jsonl(path: str) -> dict[str, dict[str, Any]]:
    return {sample_key(row): row for row in load_jsonl(path)}


def build_crop_context(
    video_path: str,
    selected: list[dict[str, Any]],
    detection_record: dict[str, Any],
    detail_frames: list[dict[str, Any]],
    max_frames: int,
) -> tuple[list[object], dict[str, Any]]:
    fps, _total_frames = _video_metadata(video_path)
    detail_indices = {int(item["frame_index"]) for item in detail_frames}
    base_meta = choose_base_frames(
        selected,
        max(1, max_frames - len(detail_frames)),
        detail_indices,
    )
    indices = sorted(
        {int(item["frame_index"]) for item in base_meta} | detail_indices
    )
    extracted = extract_frames_by_indices(video_path, indices)
    image_by_index = dict(zip(indices, extracted))
    detection_map = {
        int(item["frame_index"]): item.get("detections", [])
        for item in detection_record["frames"]
    }
    evidence: list[tuple[float, int, object]] = []
    for item in base_meta:
        frame_index = int(item["frame_index"])
        if frame_index in image_by_index:
            evidence.append(
                (
                    float(item.get("timestamp", frame_index / max(fps, 1e-6))),
                    0,
                    image_by_index[frame_index],
                )
            )
    crop_indices: list[int] = []
    for item in detail_frames:
        frame_index = int(item["frame_index"])
        detections = _top_detections(detection_map.get(frame_index, []), 2)
        if frame_index not in image_by_index or not detections:
            continue
        timestamp = float(item["timestamp"])
        evidence.append(
            (
                timestamp,
                1,
                crop_detection(image_by_index[frame_index], detections[0], timestamp),
            )
        )
        crop_indices.append(frame_index)
    evidence.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in evidence[:max_frames]], {
        "base_indices": [int(item["frame_index"]) for item in base_meta],
        "crop_indices": crop_indices,
    }


def select_uncertainty_crops(
    model: Any,
    row: dict[str, Any],
    video_path: str,
    selected: list[dict[str, Any]],
    detection_record: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[object], dict[str, Any]]:
    selected_indices = {int(item["frame_index"]) for item in selected}
    candidates = choose_detail_frames(
        [
            item
            for item in detection_record["frames"]
            if int(item["frame_index"]) in selected_indices
        ],
        args.crop_candidates,
    )
    candidate_indices = [int(item["frame_index"]) for item in candidates]
    source_frames = extract_frames_by_indices(video_path, candidate_indices)
    if len(source_frames) != len(candidate_indices):
        raise RuntimeError("Could not extract every uncertainty crop candidate")
    detection_map = {
        int(item["frame_index"]): item.get("detections", [])
        for item in detection_record["frames"]
    }
    descriptors: list[list[int]] = []
    collections: list[list[object]] = []
    valid_candidates: list[dict[str, Any]] = []
    for item, frame in zip(candidates, source_frames):
        frame_index = int(item["frame_index"])
        detections = _top_detections(detection_map.get(frame_index, []), 2)
        if not detections:
            continue
        timestamp = float(item["timestamp"])
        crop = crop_detection(frame, detections[0], timestamp)
        descriptors.append([frame_index, -frame_index - 1])
        collections.append(
            [
                resize_frame_to_max_pixels(frame, args.scoring_max_pixels),
                resize_frame_to_max_pixels(crop, args.scoring_max_pixels),
            ]
        )
        valid_candidates.append(item)
    scores, cache_hit = score_collections(
        model,
        row,
        descriptors,
        collections,
        build_entropy_prompt(row),
        args,
    )
    ranked = sorted(
        range(len(scores)),
        key=lambda index: (float(scores[index]["entropy_lower_bound"]), index),
    )
    chosen_positions = ranked[: args.detail_count]
    detail_frames = [valid_candidates[index] for index in chosen_positions]
    detail_frames.sort(key=lambda item: float(item["timestamp"]))
    final_frames, context_meta = build_crop_context(
        video_path,
        selected,
        detection_record,
        detail_frames,
        args.max_frames,
    )
    return final_frames, {
        "crop_candidate_indices": [
            int(item["frame_index"]) for item in valid_candidates
        ],
        "crop_scores": scores,
        "chosen_crop_positions": chosen_positions,
        "score_cache_hit": cache_hit,
        **context_meta,
    }


def select_uncertainty_dynamic_tcot(
    model: Any,
    row: dict[str, Any],
    video_path: str,
    total_frames: int,
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, Any]]:
    candidate_indices = baseline_uniform_indices(total_frames, args.candidate_frames)
    scores, cache_hit = _score_single_frames(
        model, row, video_path, candidate_indices, args
    )
    positions = select_segmented_positions(
        scores,
        args.segments,
        args.uncertainty_per_segment,
    )
    shortlist = [candidate_indices[position] for position in positions]
    tcot_args = SimpleNamespace(
        llm_model=args.llm_model,
        candidate_frames=len(shortlist),
        segments=args.tcot_segments,
        max_selected_per_segment=args.max_selected_per_segment,
        selector_max_pixels=args.selector_max_pixels,
        selection_cache_dir=args.tcot_cache_dir,
    )
    tcot_fingerprint = tcot_selection_fingerprint(tcot_args)
    selection = select_with_qwen(
        model,
        row,
        video_path,
        shortlist,
        tcot_args,
        tcot_fingerprint,
    )
    final_indices, pack_meta = build_final_pack(
        shortlist,
        selection["selected_indices"],
        total_frames,
        args.neighborhood_radius,
        args.selected_quota,
        args.uniform_quota,
        args.max_frames,
    )
    return final_indices, {
        "candidate_indices": candidate_indices,
        "uncertainty_scores": scores,
        "uncertainty_positions": positions,
        "shortlist_indices": shortlist,
        "score_cache_hit": cache_hit,
        "tcot_selection": selection,
        "tcot_selection_fingerprint": tcot_fingerprint,
        **pack_meta,
    }


def _run_fingerprint(args: argparse.Namespace) -> str:
    excluded = {
        "output",
        "eval_output",
        "selection_output",
        "no_resume",
        "no_eval",
        "max_samples",
    }
    payload = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _resume_prefix(
    rows: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    fingerprint: str,
) -> int:
    start = 0
    for index, prediction in enumerate(existing[: len(rows)]):
        if (
            sample_key(prediction) != sample_key(rows[index])
            or prediction.get("uncertainty_fingerprint") != fingerprint
            or _answer(prediction) not in tuple("ABCD")
        ):
            break
        start += 1
    return start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl",
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--selection-output", required=True)
    parser.add_argument("--score-cache-dir", required=True)
    parser.add_argument("--tcot-cache-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-eval", action="store_true")

    parser.add_argument("--candidate-frames", type=int, default=64)
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--per-segment", type=int, default=2)
    parser.add_argument("--uncertainty-per-segment", type=int, default=4)
    parser.add_argument("--uniform-quota", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--scoring-max-pixels", type=int, default=50176)
    parser.add_argument("--top-logprobs", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--window-stride", type=int, default=3)

    parser.add_argument("--pivot-centers", type=int, default=2)
    parser.add_argument("--target-centers", type=int, default=8)
    parser.add_argument("--eventlet-radius", type=int, default=1)
    parser.add_argument("--anchor-k", type=int, default=24)
    parser.add_argument("--bridge-k", type=int, default=8)
    parser.add_argument("--temporal-nms-seconds", type=float, default=10.0)

    parser.add_argument("--proofpack", default=None)
    parser.add_argument("--proofpack-reference", default=None)
    parser.add_argument("--detections-input", default=None)
    parser.add_argument("--crop-candidates", type=int, default=24)
    parser.add_argument("--detail-count", type=int, default=8)

    parser.add_argument("--primary-predictions", default=None)
    parser.add_argument("--secondary-predictions", default=None)
    parser.add_argument("--tertiary-predictions", default=None)

    parser.add_argument("--tcot-segments", type=int, default=4)
    parser.add_argument("--max-selected-per-segment", type=int, default=6)
    parser.add_argument("--selector-max-pixels", type=int, default=50176)
    parser.add_argument("--neighborhood-radius", type=int, default=1)
    parser.add_argument("--selected-quota", type=int, default=48)

    parser.add_argument("--llm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if args.window_size <= 0 or args.window_size % 2 == 0:
        parser.error("window-size must be a positive odd integer")
    if args.top_logprobs <= 0:
        parser.error("top-logprobs must be positive")
    if args.mode in {"object_crops", "disagreement_router"}:
        required = {
            "--proofpack": args.proofpack,
            "--proofpack-reference": args.proofpack_reference,
            "--detections-input": args.detections_input,
        }
        missing = [flag for flag, value in required.items() if not value]
        if missing:
            parser.error(f"{', '.join(missing)} required for {args.mode}")
    if args.mode == "disagreement_router":
        required = {
            "--primary-predictions": args.primary_predictions,
            "--secondary-predictions": args.secondary_predictions,
            "--tertiary-predictions": args.tertiary_predictions,
        }
        missing = [flag for flag, value in required.items() if not value]
        if missing:
            parser.error(f"{', '.join(missing)} required for disagreement_router")
    if args.mode == "dynamic_tcot" and not args.tcot_cache_dir:
        parser.error("--tcot-cache-dir is required for dynamic_tcot")
    return args


def main() -> None:
    import time

    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    eval_output = _resolve_path(args.eval_output)
    selection_output = _resolve_path(args.selection_output)
    args.score_cache_dir = _resolve_path(args.score_cache_dir)
    args.tcot_cache_dir = _resolve_path(args.tcot_cache_dir)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    fingerprint = _run_fingerprint(args)

    proofpacks: dict[str, dict[str, Any]] = {}
    detections: dict[str, dict[str, Any]] = {}
    if args.proofpack:
        proofpacks = _index_proofpack(
            _resolve_path(args.proofpack),
            _resolve_path(args.proofpack_reference),
        )
    if args.detections_input:
        detections = _index_jsonl(_resolve_path(args.detections_input))

    prediction_sets: dict[str, dict[str, dict[str, Any]]] = {}
    if args.mode == "disagreement_router":
        prediction_sets = {
            "pivot": _index_jsonl(_resolve_path(args.primary_predictions)),
            "uniform": _index_jsonl(_resolve_path(args.secondary_predictions)),
            "crop": _index_jsonl(_resolve_path(args.tertiary_predictions)),
        }

    existing = (
        []
        if args.no_resume or not os.path.exists(output_path)
        else load_jsonl(output_path)
    )
    start = _resume_prefix(rows, existing, fingerprint)
    selection_existing = (
        []
        if args.no_resume or not os.path.exists(selection_output)
        else load_jsonl(selection_output)
    )
    selection_start = 0
    for index, selection in enumerate(selection_existing[: len(rows)]):
        if (
            selection.get("sample_key") != sample_key(rows[index])
            or selection.get("uncertainty_fingerprint") != fingerprint
        ):
            break
        selection_start += 1
    start = min(start, selection_start)
    if len(existing) != start:
        _write_jsonl(output_path, existing[:start])
    if len(selection_existing) != start:
        _write_jsonl(selection_output, selection_existing[:start])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(selection_output) or ".", exist_ok=True)
    os.makedirs(args.score_cache_dir, exist_ok=True)
    if args.tcot_cache_dir:
        os.makedirs(args.tcot_cache_dir, exist_ok=True)

    print(
        f"Uncertainty config: mode={args.mode}, rows={len(rows)}, "
        f"candidates={args.candidate_frames}, top_logprobs={args.top_logprobs}, "
        f"scoring_pixels={args.scoring_max_pixels}, fingerprint={fingerprint}"
    )
    model = VLLMModel(
        args.llm_model,
        tp_size=1,
        concurrency=args.concurrency,
        max_frames=args.max_frames,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    begun = time.time()
    with (
        model,
        open(output_path, "a" if start else "w") as prediction_handle,
        open(selection_output, "a" if start else "w") as selection_handle,
    ):
        for row_index, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            video_path = os.path.join(video_folder, str(row["video_path"]))
            fps, total_frames = _video_metadata(video_path)
            if fps <= 0 or total_frames <= 0:
                raise RuntimeError(f"Could not read video metadata: {video_path}")

            if args.mode == "segmented":
                final_indices, metadata = select_segmented_pack(
                    model, row, video_path, total_frames, args
                )
                final_frames = extract_frames_by_indices(video_path, final_indices)
            elif args.mode == "temporal_pivot":
                final_indices, metadata = select_uncertainty_pivot_pack(
                    model, row, video_path, fps, total_frames, args
                )
                final_frames = extract_frames_by_indices(video_path, final_indices)
            elif args.mode == "short_window":
                final_indices, metadata = select_short_window_pack(
                    model, row, video_path, total_frames, args
                )
                final_frames = extract_frames_by_indices(video_path, final_indices)
            elif args.mode == "object_crops":
                final_frames, metadata = select_uncertainty_crops(
                    model,
                    row,
                    video_path,
                    proofpacks[key]["selected"],
                    detections[key],
                    args,
                )
                final_indices = list(metadata["base_indices"])
            elif args.mode == "dynamic_tcot":
                final_indices, metadata = select_uncertainty_dynamic_tcot(
                    model, row, video_path, total_frames, args
                )
                final_frames = extract_frames_by_indices(video_path, final_indices)
            else:
                answers = {
                    name: _answer(predictions[key])
                    for name, predictions in prediction_sets.items()
                }
                if len(set(answers.values())) == 1:
                    chosen_view = "pivot"
                    view_scores: dict[str, dict[str, Any]] = {}
                    metadata = {
                        "answers": answers,
                        "view_scores": view_scores,
                        "chosen_view": chosen_view,
                        "agreement": True,
                    }
                else:
                    pivot_indices = sorted(
                        int(item["frame_index"])
                        for item in proofpacks[key]["selected"][: args.max_frames]
                    )
                    pivot_frames = extract_frames_by_indices(video_path, pivot_indices)
                    uniform_indices = baseline_uniform_indices(total_frames, args.max_frames)
                    uniform_frames = extract_frames_by_indices(video_path, uniform_indices)
                    detail_frames = choose_detail_frames(
                        [
                            item
                            for item in detections[key]["frames"]
                            if int(item["frame_index"]) in set(pivot_indices)
                        ],
                        args.detail_count,
                    )
                    crop_frames, crop_meta = build_crop_context(
                        video_path,
                        proofpacks[key]["selected"],
                        detections[key],
                        detail_frames,
                        args.max_frames,
                    )
                    contexts = {
                        "pivot": pivot_frames,
                        "uniform": uniform_frames,
                        "crop": crop_frames,
                    }
                    scored = model.score_token_uncertainty_batch(
                        list(contexts.values()),
                        [
                            [{"role": "user", "content": build_entropy_prompt(row)}]
                            for _ in contexts
                        ],
                        top_logprobs=args.top_logprobs,
                    )
                    view_scores = {
                        name: _compact_score(score)
                        for name, score in zip(contexts, scored)
                    }
                    chosen_view = min(
                        contexts,
                        key=lambda name: (
                            float(view_scores[name]["entropy_lower_bound"]),
                            {"pivot": 0, "uniform": 1, "crop": 2}[name],
                        ),
                    )
                    metadata = {
                        "answers": answers,
                        "view_scores": view_scores,
                        "chosen_view": chosen_view,
                        "pivot_indices": pivot_indices,
                        "uniform_indices": uniform_indices,
                        "crop_meta": crop_meta,
                    }
                response = answers[chosen_view]
                prediction = build_prediction_row(
                    row, response, prompt_variant="uncertainty_disagreement_router"
                )
                final_indices = []
                final_frames = []

            if args.mode != "disagreement_router":
                response = model.generate(
                    final_frames,
                    [
                        {
                            "role": "user",
                            "content": build_longqa_prompt(
                                row["question"], row["mcq_options"]
                            ),
                        }
                    ],
                    max_new_tokens=16,
                )
                prediction = build_prediction_row(
                    row, response, prompt_variant=f"uncertainty_{args.mode}"
                )

            selection_record = {
                "index": row_index,
                "sample_key": key,
                "video_path": row["video_path"],
                "uncertainty_schema": SCHEMA_VERSION,
                "uncertainty_fingerprint": fingerprint,
                "uncertainty_mode": args.mode,
                "scoring_method": "topk_entropy_lower_bound",
                "top_logprobs": args.top_logprobs,
                "final_indices": final_indices,
                "final_frames": (
                    len(final_frames)
                    if args.mode != "disagreement_router"
                    else 0
                ),
                "metadata": metadata,
            }
            selection_handle.write(json.dumps(selection_record) + "\n")
            selection_handle.flush()
            prediction.update(
                {
                    "uncertainty_schema": SCHEMA_VERSION,
                    "uncertainty_fingerprint": fingerprint,
                    "uncertainty_mode": args.mode,
                    "uncertainty_scoring_method": "topk_entropy_lower_bound",
                    "uncertainty_top_logprobs": args.top_logprobs,
                    "uncertainty_final_frames": (
                        len(final_frames)
                        if args.mode != "disagreement_router"
                        else 0
                    ),
                }
            )
            if args.mode == "disagreement_router":
                prediction["uncertainty_chosen_view"] = chosen_view
                prediction["uncertainty_candidate_answers"] = answers
                prediction["uncertainty_view_scores"] = view_scores
            prediction_handle.write(json.dumps(prediction) + "\n")
            prediction_handle.flush()
            print(f"  Uncertainty progress: {row_index + 1}/{len(rows)}")

    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
