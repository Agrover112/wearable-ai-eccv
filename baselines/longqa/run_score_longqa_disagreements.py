#!/usr/bin/env python3
"""Score pivot, uniform, mixed, and blind views on ensemble disagreements."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from typing import Any

from longqa_utils import apply_subset, index_row_aligned_metadata, sample_key
from run_generate_longqa_grounded import extract_frames_by_indices, load_jsonl
from run_generate_longqa_likelihood import build_scoring_prompt
from run_generate_longqa_proofpack import (
    baseline_uniform_indices,
    compile_temporal_program,
)
from run_generate_longqa_uncertainty import _index_jsonl, _video_metadata
from run_generate_longqa_verifier import (
    build_verifier_frame_indices,
    rotate_mcq_options,
)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _answer(row: dict[str, Any]) -> str:
    return str(
        row.get("mcq_answer_parsed") or row.get("mcq_answer", "")
    ).strip().upper()


def _normalized_choice_stats(scores: dict[str, float]) -> dict[str, Any]:
    maximum = max(float(value) for value in scores.values())
    weights = {
        letter: math.exp(float(value) - maximum) for letter, value in scores.items()
    }
    denominator = sum(weights.values())
    probabilities = {
        letter: weight / max(denominator, 1e-12)
        for letter, weight in weights.items()
    }
    ranked = sorted(probabilities, key=probabilities.get, reverse=True)
    entropy = -sum(
        probability * math.log(max(probability, 1e-12))
        for probability in probabilities.values()
    )
    return {
        "logprobs": {letter: float(scores[letter]) for letter in "ABCD"},
        "probabilities": probabilities,
        "predicted": ranked[0],
        "margin": probabilities[ranked[0]] - probabilities[ranked[1]],
        "entropy": entropy,
    }


def _fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "primary": os.path.abspath(args.primary_predictions),
        "secondary": os.path.abspath(args.secondary_predictions),
        "tertiary": os.path.abspath(args.tertiary_predictions),
        "proofpack": os.path.abspath(args.proofpack),
        "model": args.llm_model,
        "max_frames": args.max_frames,
        "proofpack_quota": args.proofpack_quota,
        "option_rotations": args.option_rotations,
        "score_views": sorted(args.score_views),
        "subset": os.path.abspath(args.subset_file) if args.subset_file else None,
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl",
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--primary-predictions", required=True)
    parser.add_argument("--secondary-predictions", required=True)
    parser.add_argument("--tertiary-predictions", required=True)
    parser.add_argument("--proofpack", required=True)
    parser.add_argument("--proofpack-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--proofpack-quota", type=int, default=32)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--option-rotations",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="Cyclic option placements to score before mapping back and averaging.",
    )
    parser.add_argument(
        "--score-views",
        nargs="+",
        choices=("pivot", "uniform", "mixed", "blind"),
        default=["pivot", "uniform", "mixed", "blind"],
        help="Contexts to score. Restricting this list avoids unused model calls.",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _map_displayed_scores(
    scores: dict[str, float],
    displayed_to_original: dict[str, str],
) -> dict[str, float]:
    return {
        original: float(scores[displayed])
        for displayed, original in displayed_to_original.items()
    }


def _average_scores(
    score_rows: list[dict[str, float]],
) -> dict[str, float]:
    return {
        letter: sum(row[letter] for row in score_rows) / len(score_rows)
        for letter in "ABCD"
    }


def main() -> None:
    import time

    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    prediction_sets = {
        "q35_pivot": _index_jsonl(_resolve_path(args.primary_predictions)),
        "q35_uniform": _index_jsonl(_resolve_path(args.secondary_predictions)),
        "q3_verifier": _index_jsonl(_resolve_path(args.tertiary_predictions)),
    }
    proofpack_reference = load_jsonl(_resolve_path(args.proofpack_reference))
    proofpacks = index_row_aligned_metadata(
        load_jsonl(_resolve_path(args.proofpack)),
        proofpack_reference,
        "router proof pack",
    )
    disagreement_rows = []
    for row in rows:
        key = sample_key(row)
        answers = [_answer(predictions[key]) for predictions in prediction_sets.values()]
        if len(set(answers)) > 1:
            disagreement_rows.append(row)
    required = {sample_key(row) for row in disagreement_rows}
    for label, indexed in {**prediction_sets, "proofpack": proofpacks}.items():
        missing = required - set(indexed)
        if missing:
            raise RuntimeError(f"{label} is missing {len(missing)} disagreement rows")

    fingerprint = _fingerprint(args)
    existing = (
        []
        if args.no_resume or not os.path.exists(output_path)
        else load_jsonl(output_path)
    )
    start = 0
    for index, record in enumerate(existing[: len(disagreement_rows)]):
        if (
            record.get("sample_key") != sample_key(disagreement_rows[index])
            or record.get("score_fingerprint") != fingerprint
        ):
            break
        start += 1
    if len(existing) != start:
        with open(output_path, "w") as handle:
            for record in existing[:start]:
                handle.write(json.dumps(record) + "\n")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(
        f"Disagreement scoring: rows={len(disagreement_rows)}, resume={start}, "
        f"fingerprint={fingerprint}"
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
    requested_views = set(args.score_views)
    with model, open(output_path, "a" if start else "w") as handle:
        for index, row in enumerate(disagreement_rows[start:], start=start):
            key = sample_key(row)
            video_path = os.path.join(video_folder, str(row["video_path"]))
            _fps, total_frames = _video_metadata(video_path)
            if total_frames <= 0:
                raise RuntimeError(f"Could not read video metadata: {video_path}")
            frame_indices: dict[str, list[int]] = {}
            mixed_meta: dict[str, Any] | None = None
            if "pivot" in requested_views:
                frame_indices["pivot"] = sorted(
                    int(item["frame_index"])
                    for item in proofpacks[key]["selected"][: args.max_frames]
                )
            if "uniform" in requested_views:
                frame_indices["uniform"] = baseline_uniform_indices(
                    total_frames, args.max_frames
                )
            if "mixed" in requested_views:
                mixed_indices, mixed_meta = build_verifier_frame_indices(
                    proofpacks[key]["selected"],
                    total_frames,
                    max_frames=args.max_frames,
                    proofpack_quota=args.proofpack_quota,
                )
                frame_indices["mixed"] = mixed_indices
            contexts = {
                name: extract_frames_by_indices(video_path, indices)
                for name, indices in frame_indices.items()
            }
            rotation_records: list[dict[str, Any]] = []
            mapped_context_scores: dict[str, list[dict[str, float]]] = {
                name: [] for name in contexts
            }
            mapped_blind_scores: list[dict[str, float]] = []
            for offset in range(args.option_rotations):
                permuted_row, displayed_to_original = rotate_mcq_options(
                    row, offset
                )
                visual_prompt = [
                    {
                        "role": "user",
                        "content": build_scoring_prompt(permuted_row, True),
                    }
                ]
                rotation_contexts = {}
                for name, frames in contexts.items():
                    mapped = _map_displayed_scores(
                        model.score_choice_letters(frames, visual_prompt),
                        displayed_to_original,
                    )
                    mapped_context_scores[name].append(mapped)
                    rotation_contexts[name] = _normalized_choice_stats(mapped)
                mapped_blind = None
                if "blind" in requested_views:
                    blind_prompt = [
                        {
                            "role": "user",
                            "content": build_scoring_prompt(
                                permuted_row, False
                            ),
                        }
                    ]
                    mapped_blind = _map_displayed_scores(
                        model.score_choice_letters([], blind_prompt),
                        displayed_to_original,
                    )
                    mapped_blind_scores.append(mapped_blind)
                rotation_record = {
                    "offset": offset,
                    "displayed_to_original": displayed_to_original,
                    "view_scores": rotation_contexts,
                }
                if mapped_blind is not None:
                    rotation_record["blind_scores"] = (
                        _normalized_choice_stats(mapped_blind)
                    )
                rotation_records.append(rotation_record)
            scored = {
                name: _normalized_choice_stats(_average_scores(score_rows))
                for name, score_rows in mapped_context_scores.items()
            }
            blind = (
                _normalized_choice_stats(_average_scores(mapped_blind_scores))
                if mapped_blind_scores
                else None
            )
            answers = {
                name: _answer(predictions[key])
                for name, predictions in prediction_sets.items()
            }
            vote_counts = Counter(answers.values())
            record = {
                "sample_key": key,
                "video_path": row.get("video_path"),
                "question": row.get("question"),
                "category": row.get("category"),
                "temporal_operator": compile_temporal_program(
                    str(row.get("question", ""))
                ).operator,
                "candidate_answers": answers,
                "vote_counts": dict(vote_counts),
                "agreement_pattern": (
                    "all_different"
                    if len(vote_counts) == 3
                    else "two_to_one"
                ),
                "view_scores": scored,
                "blind_scores": blind,
                "frame_counts": {
                    name: len(indices)
                    for name, indices in frame_indices.items()
                },
                "mixed_frame_meta": mixed_meta,
                "option_rotations": args.option_rotations,
                "score_views": sorted(requested_views),
                "score_fingerprint": fingerprint,
            }
            if args.option_rotations > 1:
                record["rotation_scores"] = rotation_records
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"  Score progress: {index + 1}/{len(disagreement_rows)}")

    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())


if __name__ == "__main__":
    main()
