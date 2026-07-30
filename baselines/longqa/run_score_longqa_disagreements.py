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
from run_generate_longqa_verifier import build_verifier_frame_indices


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
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


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
    with model, open(output_path, "a" if start else "w") as handle:
        for index, row in enumerate(disagreement_rows[start:], start=start):
            key = sample_key(row)
            video_path = os.path.join(video_folder, str(row["video_path"]))
            _fps, total_frames = _video_metadata(video_path)
            if total_frames <= 0:
                raise RuntimeError(f"Could not read video metadata: {video_path}")
            pivot_indices = sorted(
                int(item["frame_index"])
                for item in proofpacks[key]["selected"][: args.max_frames]
            )
            uniform_indices = baseline_uniform_indices(total_frames, args.max_frames)
            mixed_indices, mixed_meta = build_verifier_frame_indices(
                proofpacks[key]["selected"],
                total_frames,
                max_frames=args.max_frames,
                proofpack_quota=args.proofpack_quota,
            )
            contexts = {
                "pivot": extract_frames_by_indices(video_path, pivot_indices),
                "uniform": extract_frames_by_indices(video_path, uniform_indices),
                "mixed": extract_frames_by_indices(video_path, mixed_indices),
            }
            prompt = [{"role": "user", "content": build_scoring_prompt(row, True)}]
            scored = {
                name: _normalized_choice_stats(
                    model.score_choice_letters(frames, prompt)
                )
                for name, frames in contexts.items()
            }
            blind = _normalized_choice_stats(
                model.score_choice_letters(
                    [],
                    [{"role": "user", "content": build_scoring_prompt(row, False)}],
                )
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
                    "pivot": len(pivot_indices),
                    "uniform": len(uniform_indices),
                    "mixed": len(mixed_indices),
                },
                "mixed_frame_meta": mixed_meta,
                "score_fingerprint": fingerprint,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"  Score progress: {index + 1}/{len(disagreement_rows)}")

    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())


if __name__ == "__main__":
    main()
