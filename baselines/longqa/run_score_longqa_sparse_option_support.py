#!/usr/bin/env python3
"""Score each LongQA option against a small option-specific evidence pack."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import time
from typing import Any

from longqa_utils import (
    apply_subset,
    build_prediction_row,
    normalize_answer,
    parse_mcq_options,
    sample_key,
)
from run_generate_longqa_grounded import extract_frames_by_indices, load_jsonl
from run_generate_longqa_proofpack import baseline_uniform_indices
from run_generate_longqa_uncertainty import _index_jsonl, _video_metadata


def resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def answer(row: dict[str, Any]) -> str:
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))


def logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def option_evidence_indices(
    proofpack: dict[str, Any],
    option_letter: str,
    total_frames: int,
    max_frames: int,
    global_quota: int,
) -> tuple[list[int], dict[str, int]]:
    metadata = proofpack.get("selection_meta") or {}
    quotas = metadata.get("target_quota_centers") or {}
    candidate_frames = max(2, int(proofpack.get("candidate_frames") or 128))
    step = (total_frames - 1) / (candidate_frames - 1)
    groups = [
        ("option", [int(value) for value in quotas.get(f"option_{option_letter}", [])[:1]]),
        ("pivot", [int(value) for value in metadata.get("pivot_centers", [])[:2]]),
        ("question_target", [int(value) for value in quotas.get("target", [])[:2]]),
    ]
    chosen: list[int] = []
    seen: set[int] = set()
    contributions: Counter[str] = Counter()

    def add(value: int, source: str) -> None:
        value = max(0, min(total_frames - 1, int(value)))
        if value in seen or len(chosen) >= max_frames:
            return
        seen.add(value)
        chosen.append(value)
        contributions[source] += 1

    for source, centers in groups:
        for center in centers:
            add(center, source)
    for source, centers in groups:
        for center in centers:
            add(round(center - step), f"{source}_context")
            add(round(center + step), f"{source}_context")
    for value in baseline_uniform_indices(total_frames, global_quota):
        add(value, "global")
    for value in baseline_uniform_indices(total_frames, max_frames):
        add(value, "global_fill")
    return sorted(chosen), dict(contributions)


def support_prompt(
    row: dict[str, Any], candidate: str, timestamps: list[float]
) -> str:
    timestamp_index = ", ".join(
        f"image {index}={timestamp:.1f}s"
        for index, timestamp in enumerate(timestamps, 1)
    )
    return f"""Evaluate one candidate answer to a question about a long first-person video.

The images are chronological but sampled, so missing evidence is not contradiction. Repeated nearby images describe one event and must not count as multiple supporting observations. For temporal questions, verify the referenced event, target event, and their order. Every part of the candidate answer must be visibly supported.

Question: {row['question']}
Candidate answer: {candidate}

Image timestamps from the start of the original video:
{timestamp_index}

Choose exactly one verdict:
A. SUPPORTED - the visible evidence supports every decisive part of the candidate.
B. CONTRADICTED - visible evidence is incompatible with a decisive part.
C. INSUFFICIENT - the supplied images do not establish the candidate either way.

Answer with only A, B, or C."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl")
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--candidate-predictions", action="append", required=True)
    parser.add_argument("--candidate-labels", nargs="+", required=True)
    parser.add_argument("--primary-predictions", required=True)
    parser.add_argument("--option-proofpack", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--backend", choices=("vllm",), default="vllm")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--global-quota", type=int, default=5)
    parser.add_argument("--score-all", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()
    if len(args.candidate_predictions) != len(args.candidate_labels):
        parser.error("candidate prediction and label counts must match")
    if not 0 <= args.global_quota <= args.max_frames:
        parser.error("--global-quota must be within the frame budget")
    return args


def run_fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "candidates": [os.path.abspath(path) for path in args.candidate_predictions],
        "labels": args.candidate_labels,
        "primary": os.path.abspath(args.primary_predictions),
        "proofpack": os.path.abspath(args.option_proofpack),
        "model": args.llm_model,
        "max_frames": args.max_frames,
        "global_quota": args.global_quota,
        "score_all": args.score_all,
        "subset": os.path.abspath(args.subset_file) if args.subset_file else None,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def main() -> None:
    from model import create_model, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa_grounded import _run_eval

    args = parse_args()
    input_path = resolve(args.input)
    output_path = resolve(args.output)
    video_folder = resolve(args.video_folder)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    candidates = {
        label: _index_jsonl(resolve(path))
        for label, path in zip(args.candidate_labels, args.candidate_predictions)
    }
    primary = _index_jsonl(resolve(args.primary_predictions))
    proofpacks = _index_jsonl(resolve(args.option_proofpack))
    required = {sample_key(row) for row in rows}
    for label, indexed in {**candidates, "primary": primary, "proofpack": proofpacks}.items():
        missing = required - set(indexed)
        if missing:
            raise RuntimeError(f"{label} is missing {len(missing)} rows")

    fingerprint = run_fingerprint(args)
    existing = [] if args.no_resume or not os.path.exists(output_path) else load_jsonl(output_path)
    start = 0
    for index, record in enumerate(existing[:len(rows)]):
        if sample_key(record) != sample_key(rows[index]) or record.get("sparse_support_fingerprint") != fingerprint:
            break
        start += 1
    if len(existing) != start:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as handle:
            for record in existing[:start]:
                handle.write(json.dumps(record) + "\n")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model = create_model(
        "qwen", args.llm_model, backend=args.backend, tp_size=args.tp,
        concurrency=1, max_frames=args.max_frames,
    )
    reset_prompt_token_stats()
    calls = 0
    begun = time.time()
    with model, open(output_path, "a" if start else "w") as handle:
        for index, row in enumerate(rows[start:], start=start):
            row_begun = time.perf_counter()
            key = sample_key(row)
            branch_answers = {
                label: answer(indexed[key]) for label, indexed in candidates.items()
            }
            primary_answer = answer(primary[key])
            if primary_answer not in {"A", "B", "C", "D"}:
                primary_answer = Counter(branch_answers.values()).most_common(1)[0][0]
            proposed = set(branch_answers.values()) | {primary_answer}
            should_score = args.score_all or len(proposed) > 1
            selected = primary_answer
            scored: dict[str, dict[str, Any]] = {}
            if should_score:
                video_path = os.path.join(video_folder, str(row["video_path"]))
                fps, total_frames = _video_metadata(video_path)
                options = parse_mcq_options(row["mcq_options"])
                for letter, text in options.items():
                    indices, sources = option_evidence_indices(
                        proofpacks[key], letter, total_frames,
                        args.max_frames, args.global_quota,
                    )
                    frames = extract_frames_by_indices(video_path, indices)
                    timestamps = [value / max(fps, 1e-6) for value in indices]
                    error = None
                    verdict_scores = None
                    support_margin = None
                    try:
                        verdict_scores = model.score_choice_letters(
                            frames,
                            [{"role": "user", "content": support_prompt(row, text, timestamps)}],
                            letters=("A", "B", "C"),
                        )
                        support_margin = verdict_scores["A"] - logsumexp(
                            [verdict_scores["B"], verdict_scores["C"]]
                        )
                    except Exception as exception:
                        error = f"{type(exception).__name__}: {exception}"
                    scored[letter] = {
                        "answer_text": text,
                        "support_margin": support_margin,
                        "verdict_logprobs": verdict_scores,
                        "frames": indices,
                        "frame_timestamps": [round(value, 3) for value in timestamps],
                        "frame_sources": sources,
                        "error": error,
                    }
                    calls += 1
                valid = {
                    letter: record["support_margin"] for letter, record in scored.items()
                    if record["support_margin"] is not None
                }
                if valid:
                    maximum = max(valid.values())
                    tied = [letter for letter, value in valid.items() if math.isclose(value, maximum)]
                    selected = primary_answer if primary_answer in tied else tied[0]

            prediction = build_prediction_row(row, selected, prompt_variant="sparse_option_support")
            prediction.update({
                "sparse_support_applied": should_score,
                "sparse_support_primary": primary_answer,
                "sparse_support_candidate_answers": branch_answers,
                "sparse_support_scores": scored,
                "sparse_support_selected": selected,
                "sparse_support_seconds": round(time.perf_counter() - row_begun, 3),
                "sparse_support_fingerprint": fingerprint,
            })
            handle.write(json.dumps(prediction) + "\n")
            handle.flush()
            print(f"  Sparse support progress: {index + 1}/{len(rows)} calls={calls}")

    print(f"Runtime seconds: {time.time() - begun:.0f}")
    print(f"Context summary: {json.dumps(summarize_prompt_token_stats(), sort_keys=True)}")
    if not args.no_eval:
        _run_eval(input_path, output_path, resolve(args.eval_output) if args.eval_output else None)


if __name__ == "__main__":
    main()
