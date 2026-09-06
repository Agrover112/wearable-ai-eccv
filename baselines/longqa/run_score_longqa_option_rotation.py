#!/usr/bin/env python3
"""Score all four semantic answers across cyclic option placements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from typing import Any

from longqa_utils import apply_subset, sample_key
from run_generate_longqa_grounded import extract_frames_by_indices, load_jsonl
from run_generate_longqa_multicandidate_judge import build_judge_prompt
from run_generate_longqa_uncertainty import _index_jsonl, _video_metadata
from run_generate_longqa_verifier import rotate_mcq_options


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(os.path.dirname(__file__), path)


def _map_scores(
    scores: dict[str, float], displayed_to_original: dict[str, str]
) -> dict[str, float]:
    return {
        original: float(scores[displayed])
        for displayed, original in displayed_to_original.items()
    }


def _probabilities(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    weights = {letter: math.exp(scores[letter] - maximum) for letter in "ABCD"}
    total = sum(weights.values())
    return {letter: weights[letter] / total for letter in "ABCD"}


def _fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "evidence": sorted(os.path.abspath(path) for path in args.evidence_predictions),
        "evidence_key": args.evidence_key,
        "model": args.llm_model,
        "rotations": args.option_rotations,
        "subset": os.path.abspath(args.subset_file) if args.subset_file else None,
        "max_samples": args.max_samples,
        "timestamps": args.include_frame_timestamps,
        "prompt": "candidate_blind_rotation_v1",
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--evidence-predictions", action="append", required=True)
    parser.add_argument(
        "--evidence-key",
        default="multicandidate_judge_frames",
        help=(
            "Field containing frame indices. Supports integer lists, lists of "
            "objects with frame_index, and dot-separated nested fields."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--option-rotations", type=int, choices=(2, 4), default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--include-frame-timestamps", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _nested_value(record: dict[str, Any], key: str) -> object:
    value: object = record
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _frame_indices(record: dict[str, Any], key: str) -> list[int]:
    values = _nested_value(record, key)
    if not isinstance(values, list):
        return []
    indices = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("frame_index")
        if value is not None:
            indices.append(int(value))
    return indices


def main() -> None:
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    rows = apply_subset(load_jsonl(_resolve(args.input)), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    evidence: dict[str, dict[str, Any]] = {}
    for path in args.evidence_predictions:
        for key, record in _index_jsonl(_resolve(path)).items():
            if key in evidence:
                raise ValueError(f"Duplicate evidence key across inputs: {key}")
            evidence[key] = record
    missing = [sample_key(row) for row in rows if sample_key(row) not in evidence]
    if missing:
        raise RuntimeError(f"Evidence predictions are missing {len(missing)} rows")

    output = _resolve(args.output)
    fingerprint = _fingerprint(args)
    existing = [] if args.no_resume or not os.path.exists(output) else load_jsonl(output)
    start = 0
    for index, record in enumerate(existing[: len(rows)]):
        if (
            record.get("sample_key") != sample_key(rows[index])
            or record.get("rotation_fingerprint") != fingerprint
        ):
            break
        start += 1
    if len(existing) != start:
        with open(output, "w") as handle:
            for record in existing[:start]:
                handle.write(json.dumps(record) + "\n")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(
        f"27B option rotation: rows={len(rows)}, resume={start}, "
        f"calls={len(rows) * args.option_rotations}, fingerprint={fingerprint}"
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
    row_seconds: list[float] = []
    with model, open(output, "a" if start else "w") as handle:
        for index, row in enumerate(rows[start:], start=start):
            row_begun = time.perf_counter()
            key = sample_key(row)
            source = evidence[key]
            indices = _frame_indices(source, args.evidence_key)
            if len(indices) != args.max_frames or len(set(indices)) != args.max_frames:
                raise RuntimeError(
                    f"Expected {args.max_frames} unique recorded evidence frames for {key}; "
                    f"got {len(indices)}"
                )
            video_path = os.path.join(_resolve(args.video_folder), str(row["video_path"]))
            fps, total_frames = _video_metadata(video_path)
            if total_frames <= 0:
                raise RuntimeError(f"Could not read video metadata: {video_path}")
            frames = extract_frames_by_indices(video_path, indices)
            timestamps = (
                [frame_index / max(fps, 1e-6) for frame_index in indices]
                if args.include_frame_timestamps
                else None
            )

            rotations: list[dict[str, Any]] = []
            mapped_scores: list[dict[str, float]] = []
            semantic_votes: list[str] = []
            for offset in range(args.option_rotations):
                permuted, displayed_to_original = rotate_mcq_options(row, offset)
                prompt = build_judge_prompt(
                    permuted,
                    {},
                    show_candidate_suggestions=False,
                    frame_timestamps=timestamps,
                )
                displayed_scores = model.score_choice_letters(
                    frames, [{"role": "user", "content": prompt}]
                )
                semantic_scores = _map_scores(displayed_scores, displayed_to_original)
                semantic_probabilities = _probabilities(semantic_scores)
                semantic_answer = max("ABCD", key=semantic_probabilities.get)
                semantic_votes.append(semantic_answer)
                mapped_scores.append(semantic_scores)
                rotations.append(
                    {
                        "offset": offset,
                        "displayed_to_original": displayed_to_original,
                        "mapped_logprobs": semantic_scores,
                        "mapped_probabilities": semantic_probabilities,
                        "semantic_answer": semantic_answer,
                    }
                )

            averaged_scores = {
                letter: sum(scores[letter] for scores in mapped_scores) / len(mapped_scores)
                for letter in "ABCD"
            }
            averaged_probabilities = _probabilities(averaged_scores)
            selected = max("ABCD", key=averaged_probabilities.get)
            seconds = time.perf_counter() - row_begun
            row_seconds.append(seconds)
            record = {
                "sample_key": key,
                "video_path": row.get("video_path"),
                "question": row.get("question"),
                "rotation_fingerprint": fingerprint,
                "model": args.llm_model,
                "frame_indices": indices,
                "frame_timestamps": timestamps or [],
                "option_rotations": args.option_rotations,
                "rotations": rotations,
                "semantic_votes": semantic_votes,
                "semantic_vote_counts": dict(Counter(semantic_votes)),
                "averaged_logprobs": averaged_scores,
                "averaged_probabilities": averaged_probabilities,
                "rotation_average_answer": selected,
                "rotation_seconds": round(seconds, 3),
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(
                f"  Rotation progress: {index + 1}/{len(rows)} "
                f"answer={selected} seconds={seconds:.1f}"
            )

    print(f"Runtime seconds: {time.time() - begun:.0f}")
    if row_seconds:
        ordered = sorted(row_seconds)
        p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
        print(
            "Per-question rotation seconds: "
            f"mean={sum(row_seconds) / len(row_seconds):.2f}, "
            f"p95={p95:.2f}, max={max(row_seconds):.2f}, "
            f"over300={sum(value > 300 for value in row_seconds)}"
        )
    _print_context_summary(summarize_prompt_token_stats())


if __name__ == "__main__":
    main()
