#!/usr/bin/env python3
"""Run coarse-to-fine, occurrence-aware temporal grounding for EgoLongQA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from longqa_utils import apply_subset, build_longqa_prompt, build_prediction_row, sample_key
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices, load_jsonl
from run_generate_longqa_proofpack import compile_temporal_program
from run_generate_longqa_uncertainty import (
    _letter_probability,
    _video_metadata,
    build_entropy_prompt,
    build_presence_prompt,
    score_collections,
)
from run_generate_longqa_proofpack import resize_frame_to_max_pixels


SCHEMA_VERSION = 1


def _resolve(path: str | None) -> str | None:
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def endpoint_uniform_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0 or count <= 0:
        return []
    if count == 1 or total_frames == 1:
        return [0]
    count = min(count, total_frames)
    return sorted(
        {
            int(round(position * (total_frames - 1) / (count - 1)))
            for position in range(count)
        }
    )


def split_positions(length: int, segments: int) -> list[list[int]]:
    segments = min(max(1, segments), max(1, length))
    boundaries = [round(index * length / segments) for index in range(segments + 1)]
    return [
        list(range(boundaries[index], boundaries[index + 1]))
        for index in range(segments)
    ]


def representative_positions(positions: list[int], count: int) -> list[int]:
    if len(positions) <= count:
        return positions
    if count <= 1:
        return [positions[len(positions) // 2]]
    return [
        positions[int(round(index * (len(positions) - 1) / (count - 1)))]
        for index in range(count)
    ]


def choose_distinct_centers(
    positions: list[int],
    scores: list[float],
    count: int,
    minimum_gap: int,
) -> list[int]:
    ranked = sorted(positions, key=lambda position: (-scores[position], position))
    selected: list[int] = []
    for position in ranked:
        if all(abs(position - other) >= minimum_gap for other in selected):
            selected.append(position)
        if len(selected) >= count:
            break
    for position in ranked:
        if position not in selected:
            selected.append(position)
        if len(selected) >= count:
            break
    return sorted(selected)


def relation_allowed(target: int, pivots: list[int], direction: str) -> bool:
    if not pivots:
        return True
    if direction == "forward":
        return any(target > pivot for pivot in pivots)
    if direction == "backward":
        return any(target < pivot for pivot in pivots)
    return True


def fill_pack(
    candidate_indices: list[int],
    priority_positions: list[int],
    total_frames: int,
    local_frames: int,
    global_frames: int,
    max_frames: int,
) -> tuple[list[int], list[int]]:
    local = []
    for position in priority_positions:
        frame_index = candidate_indices[position]
        if frame_index not in local:
            local.append(frame_index)
        if len(local) >= local_frames:
            break
    global_indices = endpoint_uniform_indices(total_frames, global_frames)
    final = list(local)
    for frame_index in global_indices:
        if frame_index not in final:
            final.append(frame_index)
        if len(final) >= max_frames:
            break
    if len(final) < max_frames:
        for frame_index in endpoint_uniform_indices(total_frames, max_frames * 2):
            if frame_index not in final:
                final.append(frame_index)
            if len(final) >= max_frames:
                break
    return sorted(final), sorted(set(global_indices) & set(final))


def image_collections(
    video_path: str,
    descriptors: list[list[int]],
    max_pixels: int,
) -> list[list[object]]:
    unique = sorted({frame_index for descriptor in descriptors for frame_index in descriptor})
    images = extract_frames_by_indices(video_path, unique)
    if len(images) != len(unique):
        raise RuntimeError("Could not extract every hierarchical candidate frame")
    by_index = dict(zip(unique, images))
    return [
        [resize_frame_to_max_pixels(by_index[index], max_pixels) for index in descriptor]
        for descriptor in descriptors
    ]


def score_descriptors(
    model: Any,
    row: dict[str, Any],
    video_path: str,
    descriptors: list[list[int]],
    prompt: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], bool]:
    return score_collections(
        model,
        row,
        descriptors,
        image_collections(video_path, descriptors, args.scoring_max_pixels),
        prompt,
        args,
    )


def select_hierarchical_pack(
    model: Any,
    row: dict[str, Any],
    video_path: str,
    fps: float,
    total_frames: int,
    args: argparse.Namespace,
) -> tuple[list[int], dict[str, Any]]:
    candidate_indices = endpoint_uniform_indices(total_frames, args.candidate_frames)
    windows = split_positions(len(candidate_indices), args.coarse_windows)
    coarse_positions = [
        representative_positions(window, args.coarse_frames_per_window)
        for window in windows
    ]
    coarse_descriptors = [
        [candidate_indices[position] for position in positions]
        for positions in coarse_positions
    ]
    program = compile_temporal_program(row["question"])
    pivot_event = program.pivot or str(row["question"])
    coarse_target, target_cache = score_descriptors(
        model,
        row,
        video_path,
        coarse_descriptors,
        build_entropy_prompt(row),
        args,
    )
    coarse_pivot, pivot_cache = score_descriptors(
        model,
        row,
        video_path,
        coarse_descriptors,
        build_presence_prompt(pivot_event),
        args,
    )
    target_window_scores = [-float(score["entropy_lower_bound"]) for score in coarse_target]
    pivot_window_scores = [
        _letter_probability(score, "A") - _letter_probability(score, "B")
        for score in coarse_pivot
    ]
    pivot_windows = choose_distinct_centers(
        list(range(len(windows))),
        pivot_window_scores,
        args.coarse_pivot_windows,
        1,
    )
    candidate_target_windows = [
        window
        for window in range(len(windows))
        if relation_allowed(window, pivot_windows, program.direction)
    ] or list(range(len(windows)))
    target_windows = choose_distinct_centers(
        candidate_target_windows,
        target_window_scores,
        args.coarse_target_windows,
        1,
    )
    selected_windows = sorted(set(pivot_windows + target_windows))
    fine_positions = sorted(
        {position for window in selected_windows for position in windows[window]}
    )
    fine_descriptors = [[candidate_indices[position]] for position in fine_positions]
    fine_target_raw, fine_target_cache = score_descriptors(
        model,
        row,
        video_path,
        fine_descriptors,
        build_entropy_prompt(row),
        args,
    )
    fine_pivot_raw, fine_pivot_cache = score_descriptors(
        model,
        row,
        video_path,
        fine_descriptors,
        build_presence_prompt(pivot_event),
        args,
    )
    target_scores = [-math.inf] * len(candidate_indices)
    pivot_scores = [-math.inf] * len(candidate_indices)
    for position, target_score, pivot_score in zip(
        fine_positions, fine_target_raw, fine_pivot_raw
    ):
        target_scores[position] = -float(target_score["entropy_lower_bound"])
        pivot_scores[position] = (
            _letter_probability(pivot_score, "A")
            - _letter_probability(pivot_score, "B")
        )
    minimum_gap = max(
        1,
        int(round(args.temporal_nms_seconds * fps * len(candidate_indices) / total_frames)),
    )
    pivot_centers = choose_distinct_centers(
        fine_positions,
        pivot_scores,
        args.pivot_centers,
        minimum_gap,
    )
    eligible_targets = [
        position
        for position in fine_positions
        if relation_allowed(position, pivot_centers, program.direction)
    ] or fine_positions
    target_centers = choose_distinct_centers(
        eligible_targets,
        target_scores,
        args.target_centers,
        minimum_gap,
    )
    priority: list[int] = []
    for center in sorted(set(pivot_centers + target_centers)):
        for position in range(
            max(0, center - args.neighborhood_radius),
            min(len(candidate_indices), center + args.neighborhood_radius + 1),
        ):
            priority.append(position)
    for pivot in pivot_centers:
        compatible = [
            target for target in target_centers if relation_allowed(target, [pivot], program.direction)
        ]
        if compatible:
            target = min(compatible, key=lambda value: abs(value - pivot))
            priority.append(int(round((pivot + target) / 2)))
    for position in fine_positions:
        if position not in priority:
            priority.append(position)
    final_indices, global_indices = fill_pack(
        candidate_indices,
        priority,
        total_frames,
        args.local_frames,
        args.global_frames,
        args.max_frames,
    )
    return final_indices, {
        "candidate_indices": candidate_indices,
        "coarse_windows": [
            [candidate_indices[position] for position in window] for window in windows
        ],
        "coarse_pivot_windows": pivot_windows,
        "coarse_target_windows": target_windows,
        "selected_windows": selected_windows,
        "pivot_centers": [candidate_indices[position] for position in pivot_centers],
        "target_centers": [candidate_indices[position] for position in target_centers],
        "global_indices": global_indices,
        "temporal_program": {
            "operator": program.operator,
            "pivot": program.pivot,
            "direction": program.direction,
            "target": program.target,
        },
        "cache_hits": {
            "coarse_target": target_cache,
            "coarse_pivot": pivot_cache,
            "fine_target": fine_target_cache,
            "fine_pivot": fine_pivot_cache,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl")
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--score-cache-dir", required=True)
    parser.add_argument("--candidate-frames", type=int, default=128)
    parser.add_argument("--coarse-windows", type=int, default=16)
    parser.add_argument("--coarse-frames-per-window", type=int, default=4)
    parser.add_argument("--coarse-pivot-windows", type=int, default=3)
    parser.add_argument("--coarse-target-windows", type=int, default=4)
    parser.add_argument("--pivot-centers", type=int, default=3)
    parser.add_argument("--target-centers", type=int, default=8)
    parser.add_argument("--neighborhood-radius", type=int, default=1)
    parser.add_argument("--temporal-nms-seconds", type=float, default=8.0)
    parser.add_argument("--local-frames", type=int, default=48)
    parser.add_argument("--global-frames", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--scoring-max-pixels", type=int, default=50176)
    parser.add_argument("--top-logprobs", type=int, default=100)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.local_frames + args.global_frames > args.max_frames:
        parser.error("--local-frames + --global-frames cannot exceed --max-frames")
    return args


def fingerprint(args: argparse.Namespace) -> str:
    ignored = {"output", "selection_output", "eval_output", "no_resume", "max_samples"}
    payload = {key: value for key, value in vars(args).items() if key not in ignored}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def write_rows(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    import time
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve(args.input)
    video_folder = _resolve(args.video_folder)
    output_path = _resolve(args.output)
    selection_path = _resolve(args.selection_output)
    eval_path = _resolve(args.eval_output)
    args.score_cache_dir = _resolve(args.score_cache_dir)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    run_fingerprint = fingerprint(args)
    existing_predictions = [] if args.no_resume or not os.path.exists(output_path) else load_jsonl(output_path)
    existing_selections = [] if args.no_resume or not os.path.exists(selection_path) else load_jsonl(selection_path)
    start = 0
    for index, row in enumerate(rows):
        if index >= len(existing_predictions) or index >= len(existing_selections):
            break
        if (
            sample_key(existing_predictions[index]) != sample_key(row)
            or existing_predictions[index].get("hierarchical_fingerprint") != run_fingerprint
            or existing_selections[index].get("hierarchical_fingerprint") != run_fingerprint
        ):
            break
        start += 1
    write_rows(output_path, existing_predictions[:start])
    write_rows(selection_path, existing_selections[:start])
    Path(args.score_cache_dir).mkdir(parents=True, exist_ok=True)
    model = VLLMModel(
        args.llm_model,
        tp_size=1,
        concurrency=args.concurrency,
        max_frames=args.max_frames,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    begun = time.time()
    with model, open(output_path, "a") as prediction_handle, open(selection_path, "a") as selection_handle:
        for index, row in enumerate(rows[start:], start=start):
            video_path = os.path.join(video_folder, str(row["video_path"]))
            fps, total_frames = _video_metadata(video_path)
            if fps <= 0 or total_frames <= 0:
                raise RuntimeError(f"Could not read video metadata: {video_path}")
            final_indices, metadata = select_hierarchical_pack(
                model, row, video_path, fps, total_frames, args
            )
            frames = extract_frames_by_indices(video_path, final_indices)
            response = model.generate(
                frames,
                [{"role": "user", "content": build_longqa_prompt(row["question"], row["mcq_options"])}],
                max_new_tokens=16,
            )
            prediction = build_prediction_row(row, response, prompt_variant="hierarchical_temporal_pivot")
            prediction.update(
                {
                    "hierarchical_schema": SCHEMA_VERSION,
                    "hierarchical_fingerprint": run_fingerprint,
                    "hierarchical_final_frames": len(final_indices),
                }
            )
            selection = {
                "index": index,
                "sample_key": sample_key(row),
                "video_path": row["video_path"],
                "hierarchical_schema": SCHEMA_VERSION,
                "hierarchical_fingerprint": run_fingerprint,
                "final_indices": final_indices,
                "metadata": metadata,
            }
            prediction_handle.write(json.dumps(prediction) + "\n")
            prediction_handle.flush()
            selection_handle.write(json.dumps(selection) + "\n")
            selection_handle.flush()
            print(f"  Hierarchical progress: {index + 1}/{len(rows)}")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    _run_eval(input_path, output_path, eval_path)


if __name__ == "__main__":
    main()
