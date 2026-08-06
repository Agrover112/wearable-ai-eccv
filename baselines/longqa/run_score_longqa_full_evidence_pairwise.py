#!/usr/bin/env python3
"""Score a baseline answer against one challenger using exact recorded evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from longqa_utils import apply_subset, parse_mcq_options
from run_generate_longqa_grounded import extract_frames_by_indices, load_jsonl


VALID_ANSWERS = {"A", "B", "C", "D"}


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _answer(row: dict[str, Any]) -> str:
    answer = str(
        row.get("mcq_answer_parsed") or row.get("mcq_answer", "")
    ).strip().upper()
    if answer not in VALID_ANSWERS:
        raise ValueError(
            f"Invalid answer {answer!r} for {row.get('video_path', '<unknown>')}"
        )
    return answer


def _index_video_paths(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_path = str(row.get("video_path", "")).strip()
        if not video_path:
            raise ValueError(f"{label} contains a row without video_path")
        if video_path in indexed:
            raise ValueError(f"{label} contains duplicate video_path {video_path!r}")
        indexed[video_path] = row
    return indexed


def build_pair_prompt(
    row: dict[str, Any], first_answer: str, second_answer: str
) -> str:
    options = parse_mcq_options(str(row["mcq_options"]))
    if first_answer not in options or second_answer not in options:
        raise ValueError("Pairwise candidates must be valid options")
    return (
        "Use the complete chronological frame sequence to compare exactly two "
        "candidate answers. Check object identity, visible support, visible "
        "contradiction, repeated occurrences, and the temporal order requested "
        "by the question. Frame frequency is not evidence: repeated nearby views "
        "of one event must not count as separate supporting events. Choose C when "
        "the supplied frames do not clearly distinguish the candidates.\n\n"
        f"Question: {row['question']}\n\n"
        f"A. {options[first_answer]}\n"
        f"B. {options[second_answer]}\n"
        "C. The visual evidence is insufficient or ambiguous.\n\n"
        "Return only A, B, or C."
    )


def normalize_logprobs(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    weights = {key: math.exp(value - maximum) for key, value in scores.items()}
    denominator = sum(weights.values())
    return {key: value / denominator for key, value in weights.items()}


def map_order_probabilities(
    displayed: dict[str, float], first: str, second: str
) -> dict[str, float]:
    return {
        first: float(displayed["A"]),
        second: float(displayed["B"]),
        "INSUFFICIENT": float(displayed["C"]),
    }


def _fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "baseline": os.path.abspath(args.baseline_predictions),
        "challenger": os.path.abspath(args.challenger_predictions),
        "evidence": os.path.abspath(args.evidence_predictions),
        "model": args.llm_model,
        "frames_key": args.frames_key,
        "subset": os.path.abspath(args.subset_file) if args.subset_file else None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode()).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--challenger-predictions", required=True)
    parser.add_argument("--evidence-predictions", required=True)
    parser.add_argument("--frames-key", default="multicandidate_judge_frames")
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    baseline = _index_video_paths(
        load_jsonl(_resolve_path(args.baseline_predictions)), "baseline predictions"
    )
    challenger = _index_video_paths(
        load_jsonl(_resolve_path(args.challenger_predictions)), "challenger predictions"
    )
    evidence = _index_video_paths(
        load_jsonl(_resolve_path(args.evidence_predictions)), "evidence predictions"
    )
    required = {str(row["video_path"]) for row in rows}
    for label, indexed in (
        ("baseline", baseline),
        ("challenger", challenger),
        ("evidence", evidence),
    ):
        missing = required - set(indexed)
        if missing:
            raise RuntimeError(f"{label} is missing {len(missing)} selected videos")

    fingerprint = _fingerprint(args)
    existing = (
        []
        if args.no_resume or not os.path.exists(output_path)
        else load_jsonl(output_path)
    )
    start = 0
    for index, record in enumerate(existing[: len(rows)]):
        if (
            record.get("video_path") != rows[index].get("video_path")
            or record.get("pairwise_fingerprint") != fingerprint
        ):
            break
        start += 1
    if len(existing) != start:
        with open(output_path, "w", encoding="utf-8") as handle:
            for record in existing[:start]:
                handle.write(json.dumps(record) + "\n")

    target_count = sum(
        _answer(baseline[str(row["video_path"])])
        != _answer(challenger[str(row["video_path"])])
        for row in rows
    )
    print(
        f"Full-evidence pairwise scoring: rows={len(rows)}, targets={target_count}, "
        f"resume={start}, model={args.llm_model}, fingerprint={fingerprint}"
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model = VLLMModel(
        args.llm_model,
        tp_size=args.tp,
        concurrency=args.concurrency,
        max_frames=64,
    )
    reset_prompt_token_stats()
    begun = time.time()
    calls = 0
    with model, open(output_path, "a" if start else "w", encoding="utf-8") as handle:
        for index, row in enumerate(rows[start:], start=start):
            row_started = time.perf_counter()
            video_id = str(row["video_path"])
            baseline_answer = _answer(baseline[video_id])
            challenger_answer = _answer(challenger[video_id])
            record: dict[str, Any] = {
                "video_path": video_id,
                "baseline_answer": baseline_answer,
                "challenger_answer": challenger_answer,
                "pairwise_applied": baseline_answer != challenger_answer,
                "pairwise_fingerprint": fingerprint,
            }
            if baseline_answer != challenger_answer:
                frame_indices = [
                    int(value) for value in evidence[video_id].get(args.frames_key, [])
                ]
                if len(frame_indices) != 64 or len(set(frame_indices)) != 64:
                    raise RuntimeError(
                        f"{video_id}: expected 64 unique recorded evidence frames, "
                        f"got {len(frame_indices)} values and {len(set(frame_indices))} unique"
                    )
                frames = extract_frames_by_indices(
                    os.path.join(video_folder, video_id), frame_indices
                )
                forward_logprobs = model.score_choice_letters(
                    frames,
                    [{"role": "user", "content": build_pair_prompt(
                        row, baseline_answer, challenger_answer
                    )}],
                    letters=("A", "B", "C"),
                )
                reverse_logprobs = model.score_choice_letters(
                    frames,
                    [{"role": "user", "content": build_pair_prompt(
                        row, challenger_answer, baseline_answer
                    )}],
                    letters=("A", "B", "C"),
                )
                forward = map_order_probabilities(
                    normalize_logprobs(forward_logprobs),
                    baseline_answer,
                    challenger_answer,
                )
                reverse = map_order_probabilities(
                    normalize_logprobs(reverse_logprobs),
                    challenger_answer,
                    baseline_answer,
                )
                record.update(
                    {
                        "frame_indices": frame_indices,
                        "forward_probabilities": forward,
                        "reverse_probabilities": reverse,
                        "forward_logprobs": forward_logprobs,
                        "reverse_logprobs": reverse_logprobs,
                    }
                )
                calls += 2
            record["pairwise_seconds"] = round(time.perf_counter() - row_started, 3)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"  Pairwise progress: {index + 1}/{len(rows)} calls={calls}")

    print(f"Pairwise model calls this invocation: {calls}")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())


if __name__ == "__main__":
    main()
