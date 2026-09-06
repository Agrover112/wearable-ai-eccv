#!/usr/bin/env python3
"""Build EgoLongQA proof packs from Qwen-verified event bursts.

This selector deliberately avoids using SigLIP similarity to establish temporal
anchors. It first scores coarse windows spanning the complete video with the
Qwen multimodal reranker, then refines the strongest event windows to short
one-second neighborhoods. Half of the final frame budget remains a uniform,
endpoint-inclusive global view so an incorrect event proposal cannot remove
the rest of the recording.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from longqa_utils import apply_subset, load_jsonl, parse_mcq_options, sample_key
from run_generate_longqa_grounded import (
    CandidateFrame,
    _uniform_positions,
    extract_frames_by_indices,
    load_jsonl_if_exists,
)
from run_generate_longqa_proofpack import TemporalProgram, compile_temporal_program_v2
from run_generate_longqa_qwen_rerank import QwenVLWindowReranker, _video_metadata


logger = logging.getLogger(__name__)
SCHEMA = 1
STRATEGY = "qwen_verified_event_bursts"
QUERY_MODES = ("events_only", "events_options")


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def split_positions(length: int, parts: int) -> list[list[int]]:
    """Split a candidate sequence into non-empty chronological windows."""
    parts = min(max(1, parts), max(1, length))
    boundaries = [round(index * length / parts) for index in range(parts + 1)]
    return [
        list(range(boundaries[index], boundaries[index + 1]))
        for index in range(parts)
        if boundaries[index] < boundaries[index + 1]
    ]


def representative_positions(positions: list[int], count: int) -> list[int]:
    if len(positions) <= count:
        return positions
    if count <= 1:
        return [positions[len(positions) // 2]]
    return [
        positions[round(index * (len(positions) - 1) / (count - 1))]
        for index in range(count)
    ]


def rank_windows(
    scores: list[float],
    count: int,
    minimum_gap: int = 1,
    eligible: set[int] | None = None,
) -> list[int]:
    ranked = sorted(
        (index for index in range(len(scores)) if eligible is None or index in eligible),
        key=lambda index: (-scores[index], index),
    )
    selected: list[int] = []
    for index in ranked:
        if all(abs(index - other) >= minimum_gap for other in selected):
            selected.append(index)
            if len(selected) >= count:
                return selected
    for index in ranked:
        if index not in selected:
            selected.append(index)
            if len(selected) >= count:
                break
    return selected


def build_event_queries(
    row: dict[str, Any], program: TemporalProgram, query_mode: str
) -> dict[str, str]:
    question = str(row["question"])
    queries: dict[str, str] = {}
    if program.pivot:
        queries["anchor"] = (
            "Locate this temporal reference event, not a visually similar event from "
            f"another time: {program.pivot}"
        )
    target = program.target or question
    queries["target"] = (
        "Locate the visible event or state needed to answer this question. "
        f"Question: {question}\nTarget evidence: {target}"
    )
    for index, event in enumerate(program.event_slots[:2]):
        queries[f"slot_{index}"] = (
            "Locate this specific event occurrence in the video. Do not substitute a "
            f"similar occurrence: {event}"
        )
    if query_mode == "events_options":
        for letter, answer in sorted(parse_mcq_options(row["mcq_options"]).items()):
            queries[f"option_{letter}"] = (
                f"Question: {question}\nCandidate answer: {answer}\n"
                "Find direct visual evidence for this complete candidate answer."
            )
    return queries


def relation_eligible(
    program: TemporalProgram, anchor_window: int, window_count: int
) -> set[int]:
    if program.operator == "AFTER":
        return set(range(anchor_window + 1, window_count))
    if program.operator == "BEFORE":
        return set(range(0, anchor_window))
    return set(range(window_count))


def adjusted_option_scores(
    scores: dict[str, list[float]], contrast_weight: float
) -> dict[str, list[float]]:
    labels = sorted(label for label in scores if label.startswith("option_"))
    adjusted: dict[str, list[float]] = {}
    for label in labels:
        values = []
        for index, raw in enumerate(scores[label]):
            competitors = [scores[other][index] for other in labels if other != label]
            competing = max(competitors) if competitors else raw
            values.append(raw + contrast_weight * (raw - competing))
        adjusted[label] = values
    return adjusted


def choose_coarse_proposals(
    scores: dict[str, list[float]],
    program: TemporalProgram,
    max_proposals: int,
    option_contrast_weight: float,
    qualification_delta: float,
) -> list[dict[str, Any]]:
    """Choose event windows without allowing one uncertain anchor to erase recall."""
    if "target" not in scores:
        raise ValueError("target scores are required")
    window_count = len(scores["target"])
    proposals: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    def add(role: str, window: int, reason: str, value: float | None = None) -> None:
        key = (role, window)
        if key in seen or len(proposals) >= max_proposals:
            return
        seen.add(key)
        proposals.append(
            {
                "role": role,
                "window": window,
                "reason": reason,
                "score": float(scores[role][window] if value is None else value),
            }
        )

    anchors: list[int] = []
    if "anchor" in scores:
        anchors = rank_windows(scores["anchor"], 2, minimum_gap=2)
        for window in anchors:
            add("anchor", window, "verified_anchor")

    target_ranked = rank_windows(scores["target"], 2, minimum_gap=2)
    if target_ranked:
        add("target", target_ranked[0], "best_unrestricted_target")

    # Add relation-compatible evidence, but retain the unrestricted target above.
    if anchors and program.operator in {"AFTER", "BEFORE"}:
        eligible = relation_eligible(program, anchors[0], window_count)
        compatible = rank_windows(scores["target"], 1, eligible=eligible)
        if compatible:
            add("target", compatible[0], "best_relation_compatible_target")

    for role in sorted(label for label in scores if label.startswith("slot_")):
        top = rank_windows(scores[role], 1)
        if top:
            add(role, top[0], "event_slot")

    target_values = scores["target"]
    threshold = max(target_values) - qualification_delta
    qualifying = [
        index for index, value in enumerate(target_values) if value >= threshold
    ]
    if qualifying and program.operator in {"FIRST", "MULTI_TIME", "STATE_CHANGE"}:
        add("target", min(qualifying), "earliest_qualified_target")
    if qualifying and program.operator in {"LAST", "MULTI_TIME", "STATE_CHANGE"}:
        add("target", max(qualifying), "latest_qualified_target")

    adjusted = adjusted_option_scores(scores, option_contrast_weight)
    option_choices = []
    for role, values in adjusted.items():
        top = rank_windows(values, 1)
        if top:
            option_choices.append((values[top[0]], role, top[0]))
    for value, role, window in sorted(option_choices, key=lambda item: (-item[0], item[1])):
        add(role, window, "option_contrast", value)

    # Fill spare slots from independent role/window scores, preserving role identity.
    remaining = sorted(
        (
            value,
            role,
            window,
        )
        for role, values in scores.items()
        for window, value in enumerate(values)
        if (role, window) not in seen
    )
    for value, role, window in reversed(remaining):
        add(role, window, "score_fill", value)
        if len(proposals) >= max_proposals:
            break
    return proposals


def temporal_burst_indices(
    center: int,
    fps: float,
    total_frames: int,
    count: int,
    span_seconds: float,
) -> list[int]:
    if count <= 1:
        return [min(max(center, 0), total_frames - 1)]
    offsets = [
        -span_seconds / 2 + index * span_seconds / (count - 1)
        for index in range(count)
    ]
    offsets[min(range(len(offsets)), key=lambda index: (abs(offsets[index]), -index))] = 0.0
    return sorted(
        {
            min(total_frames - 1, max(0, int(round(center + offset * fps))))
            for offset in offsets
        }
    )


def build_final_indices(
    centers: list[int],
    fps: float,
    total_frames: int,
    burst_frames: int,
    burst_span_seconds: float,
    global_frames: int,
    final_frames: int,
) -> tuple[list[int], list[int], list[int]]:
    burst_indices: list[int] = []
    for center in centers:
        for index in temporal_burst_indices(
            center, fps, total_frames, burst_frames, burst_span_seconds
        ):
            if index not in burst_indices:
                burst_indices.append(index)
    if len(burst_indices) > final_frames - global_frames:
        burst_indices = burst_indices[: final_frames - global_frames]
    global_indices = _uniform_positions(total_frames, global_frames)
    selected = set(burst_indices)
    selected.update(global_indices)
    for index in _uniform_positions(total_frames, final_frames * 2):
        if len(selected) >= final_frames:
            break
        selected.add(index)
    if len(selected) < final_frames:
        for index in range(total_frames):
            if len(selected) >= final_frames:
                break
            selected.add(index)
    if len(selected) != final_frames:
        raise RuntimeError(f"selected {len(selected)}/{final_frames} final frames")
    return sorted(selected), sorted(set(burst_indices)), sorted(set(global_indices))


def _fingerprint(args: argparse.Namespace) -> str:
    ignored = {"output", "max_samples"}
    payload = {
        "schema": SCHEMA,
        **{key: value for key, value in vars(args).items() if key not in ignored},
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--query-mode", choices=QUERY_MODES, default="events_options")
    parser.add_argument("--candidate-frames", type=int, default=256)
    parser.add_argument("--coarse-windows", type=int, default=32)
    parser.add_argument("--coarse-frames-per-window", type=int, default=4)
    parser.add_argument("--max-event-centers", type=int, default=6)
    parser.add_argument("--fine-verification-frames", type=int, default=3)
    parser.add_argument("--fine-verification-span-seconds", type=float, default=2.0)
    parser.add_argument("--burst-frames", type=int, default=5)
    parser.add_argument("--burst-span-seconds", type=float, default=4.0)
    parser.add_argument("--global-frames", type=int, default=32)
    parser.add_argument("--final-frames", type=int, default=64)
    parser.add_argument("--option-contrast-weight", type=float, default=0.5)
    parser.add_argument("--qualification-delta", type=float, default=0.10)
    parser.add_argument("--reranker-model", default="Qwen/Qwen3-VL-Reranker-2B")
    parser.add_argument(
        "--reranker-revision",
        default="93eac850736c677b682c67fc0302b03e552a7b16",
    )
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--reranker-max-pixels", type=int, default=100352)
    args = parser.parse_args()
    if args.burst_frames * args.max_event_centers > (
        args.final_frames - args.global_frames
    ):
        parser.error("event bursts exceed the reserved non-global frame budget")
    return args


def main() -> None:
    import torch

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    fingerprint = _fingerprint(args)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cached = load_jsonl_if_exists(output_path)
    records: list[dict[str, Any]] = []
    for index, record in enumerate(cached[: len(rows)]):
        if (
            int(record.get("index", -1)) != index
            or record.get("sample_key") != sample_key(rows[index])
            or record.get("proofpack_fingerprint") != fingerprint
        ):
            break
        records.append(record)
    if len(records) != len(cached):
        with open(output_path, "w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    if len(records) == len(rows):
        print(f"Verified-burst proof pack already complete: {output_path}")
        return

    reranker = QwenVLWindowReranker(
        args.reranker_model,
        args.reranker_revision,
        args.reranker_batch_size,
        args.reranker_max_pixels,
    )
    mode = "a" if records else "w"
    with open(output_path, mode) as handle:
        for row_index, row in enumerate(rows[len(records) :], start=len(records)):
            video_path = os.path.join(video_folder, str(row["video_path"]))
            total_frames, fps = _video_metadata(video_path)
            candidate_indices = _uniform_positions(total_frames, args.candidate_frames)
            candidates = [
                CandidateFrame(index, index / fps, None) for index in candidate_indices
            ]
            coarse_positions = split_positions(
                len(candidates), args.coarse_windows
            )
            coarse_descriptors = [
                representative_positions(window, args.coarse_frames_per_window)
                for window in coarse_positions
            ]
            coarse_indices = sorted(
                {
                    candidates[position].index
                    for descriptor in coarse_descriptors
                    for position in descriptor
                }
            )
            coarse_images = extract_frames_by_indices(video_path, coarse_indices)
            if len(coarse_images) != len(coarse_indices):
                raise RuntimeError("Could not decode all coarse verification frames")
            coarse_by_index = dict(zip(coarse_indices, coarse_images))
            coarse_windows = [
                [coarse_by_index[candidates[position].index] for position in descriptor]
                for descriptor in coarse_descriptors
            ]
            coarse_timestamps = [
                [candidates[position].timestamp for position in descriptor]
                for descriptor in coarse_descriptors
            ]
            program = compile_temporal_program_v2(str(row["question"]))
            queries = build_event_queries(row, program, args.query_mode)
            coarse_scores = {
                role: reranker.score(query, coarse_windows, coarse_timestamps)
                for role, query in queries.items()
            }
            proposals = choose_coarse_proposals(
                coarse_scores,
                program,
                args.max_event_centers,
                args.option_contrast_weight,
                args.qualification_delta,
            )

            fine_specs: list[tuple[dict[str, Any], list[int]]] = []
            needed_fine_indices: set[int] = set()
            for proposal in proposals:
                positions = coarse_positions[int(proposal["window"])]
                fine_specs.append((proposal, positions))
                for position in positions:
                    needed_fine_indices.update(
                        temporal_burst_indices(
                            candidates[position].index,
                            fps,
                            total_frames,
                            args.fine_verification_frames,
                            args.fine_verification_span_seconds,
                        )
                    )
            fine_indices = sorted(needed_fine_indices)
            fine_images = extract_frames_by_indices(video_path, fine_indices)
            if len(fine_images) != len(fine_indices):
                raise RuntimeError("Could not decode all fine verification frames")
            fine_by_index = dict(zip(fine_indices, fine_images))
            refined: list[dict[str, Any]] = []
            for proposal, positions in fine_specs:
                windows = []
                timestamps = []
                for position in positions:
                    indices = temporal_burst_indices(
                        candidates[position].index,
                        fps,
                        total_frames,
                        args.fine_verification_frames,
                        args.fine_verification_span_seconds,
                    )
                    windows.append([fine_by_index[index] for index in indices])
                    timestamps.append([index / fps for index in indices])
                values = reranker.score(queries[proposal["role"]], windows, timestamps)
                best_local = max(range(len(values)), key=lambda index: values[index])
                center_position = positions[best_local]
                refined.append(
                    {
                        **proposal,
                        "coarse_score": proposal["score"],
                        "fine_score": float(values[best_local]),
                        "candidate_position": center_position,
                        "frame_index": candidates[center_position].index,
                        "timestamp": candidates[center_position].timestamp,
                    }
                )

            unique_centers: list[int] = []
            for item in refined:
                center = int(item["frame_index"])
                if center not in unique_centers:
                    unique_centers.append(center)
            final_indices, burst_indices, global_indices = build_final_indices(
                unique_centers,
                fps,
                total_frames,
                args.burst_frames,
                args.burst_span_seconds,
                args.global_frames,
                args.final_frames,
            )
            burst_set = set(burst_indices)
            global_set = set(global_indices)
            selected = []
            for frame_index in final_indices:
                if frame_index in burst_set:
                    source = "qwen_verified_event_burst"
                elif frame_index in global_set:
                    source = "endpoint_global"
                else:
                    source = "coverage_fill"
                selected.append(
                    {
                        "frame_index": frame_index,
                        "timestamp": round(frame_index / fps, 3),
                        "score": None,
                        "source": source,
                    }
                )
            record = {
                "index": row_index,
                "sample_key": sample_key(row),
                "video_path": row.get("video_path", ""),
                "strategy": STRATEGY,
                "candidate_frames": len(candidates),
                "selected_frames": len(selected),
                "proofpack_fingerprint": fingerprint,
                "reranker_model": args.reranker_model,
                "reranker_revision": args.reranker_revision,
                "retrieval_query_mode": args.query_mode,
                "final_indices": final_indices,
                "selected": selected,
                "selection_meta": {
                    "temporal_program": {
                        "operator": program.operator,
                        "pivot": program.pivot,
                        "direction": program.direction,
                        "target": program.target,
                        "event_slots": program.event_slots,
                    },
                    "coarse_windows": [
                        [candidates[position].index for position in window]
                        for window in coarse_positions
                    ],
                    "coarse_scores": {
                        role: [round(value, 6) for value in values]
                        for role, values in coarse_scores.items()
                    },
                    "proposals": proposals,
                    "refined_events": refined,
                    "verified_centers": unique_centers,
                    "burst_frame_indices": burst_indices,
                    "global_frame_indices": global_indices,
                    "route": f"qwen_verified_bursts_{args.query_mode}",
                },
            }
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"Verified-burst progress: {row_index + 1}/{len(rows)}")

    del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Verified-burst proof pack complete: {len(records)} rows -> {output_path}")


if __name__ == "__main__":
    main()
