#!/usr/bin/env python3
"""Resolve LongQA candidate disagreements with order-balanced pairwise scoring."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from typing import Any

from longqa_utils import (
    apply_subset,
    build_prediction_row,
    index_row_aligned_metadata,
    parse_mcq_options,
    sample_key,
)
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices, load_jsonl


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _answer(row: dict[str, Any]) -> str:
    return str(row.get("mcq_answer_parsed") or row.get("mcq_answer", "")).strip().upper()


def pairwise_prompt(question: str, first_text: str, second_text: str) -> str:
    return (
        "Use the chronological video frames to decide which of these two candidate "
        "answers is better supported. Check object identity and temporal order.\n\n"
        f"Question: {question}\n\n"
        f"Candidate 1: {first_text}\n"
        f"Candidate 2: {second_text}\n\n"
        "Return ONLY 1 or 2.\nAnswer:"
    )


def aggregate_pairwise_margins(
    candidates: list[str], pair_margins: dict[str, float]
) -> tuple[str, dict[str, float]]:
    totals = {candidate: 0.0 for candidate in candidates}
    for pair_key, margin in pair_margins.items():
        first, second = pair_key.split("_vs_", 1)
        totals[first] += float(margin)
        totals[second] -= float(margin)
    return max(candidates, key=lambda item: (totals[item], -candidates.index(item))), totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl")
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--candidate-predictions", action="append", required=True)
    parser.add_argument("--fallback-predictions", required=True)
    parser.add_argument("--proofpack", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    import time

    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    eval_output = _resolve_path(args.eval_output) if args.eval_output else None
    all_rows = load_jsonl(input_path)
    included_rows = apply_subset(all_rows, args.subset_file)
    rows = included_rows[: args.max_samples] if args.max_samples is not None else included_rows

    def aligned(path: str, label: str) -> dict[str, dict[str, Any]]:
        records = load_jsonl(_resolve_path(path))
        if len(records) == len(all_rows):
            reference = all_rows
        elif len(records) == len(included_rows):
            reference = included_rows
        else:
            raise RuntimeError(
                f"{label} has {len(records)} rows; expected {len(all_rows)} or "
                f"{len(included_rows)}"
            )
        return index_row_aligned_metadata(records, reference, label)

    candidate_runs = [
        aligned(path, f"candidate predictions {index + 1}")
        for index, path in enumerate(args.candidate_predictions)
    ]
    fallback = aligned(args.fallback_predictions, "fallback predictions")
    proofpack = aligned(args.proofpack, "proof pack")
    required = {sample_key(row) for row in rows}
    for label, mapping in [
        *( (f"candidate {index + 1}", run) for index, run in enumerate(candidate_runs) ),
        ("fallback", fallback),
        ("proof pack", proofpack),
    ]:
        missing = required - set(mapping)
        if missing:
            raise RuntimeError(f"{label} is missing {len(missing)} required rows")

    payload = {
        "candidate_predictions": [os.path.abspath(_resolve_path(path)) for path in args.candidate_predictions],
        "fallback_predictions": os.path.abspath(_resolve_path(args.fallback_predictions)),
        "proofpack": os.path.abspath(_resolve_path(args.proofpack)),
        "model": args.llm_model,
        "max_frames": args.max_frames,
        "subset": os.path.abspath(_resolve_path(args.subset_file)) if args.subset_file else None,
    }
    fingerprint = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    existing = []
    if not args.no_resume and os.path.exists(output_path):
        existing = load_jsonl(output_path)
    start = 0
    for index, pred in enumerate(existing[: len(rows)]):
        if sample_key(pred) != sample_key(rows[index]):
            break
        if pred.get("pairwise_tournament_fingerprint") != fingerprint:
            break
        start += 1

    disagreement_count = sum(
        len({_answer(run[sample_key(row)]) for run in candidate_runs}) > 1 for row in rows
    )
    print(
        f"Pairwise tournament: rows={len(rows)}, disagreements={disagreement_count}, "
        f"resume={start}, fingerprint={fingerprint}"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model = VLLMModel(
        args.llm_model,
        tp_size=args.tp,
        concurrency=args.concurrency,
        max_frames=args.max_frames,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    begun = time.time()
    with model, open(output_path, "a" if start else "w") as handle:
        for index, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            proposed = [_answer(run[key]) for run in candidate_runs]
            candidates = list(dict.fromkeys(answer for answer in proposed if answer in "ABCD"))
            fallback_answer = _answer(fallback[key])
            pair_records: list[dict[str, Any]] = []
            pair_margins: dict[str, float] = {}
            failed_pairs = 0
            if len(candidates) <= 1:
                answer = candidates[0] if candidates else fallback_answer
                totals = {answer: 0.0}
                frame_indices: list[int] = []
            else:
                options = parse_mcq_options(row["mcq_options"])
                frame_indices = sorted(
                    int(item["frame_index"])
                    for item in proofpack[key]["selected"][: args.max_frames]
                )
                frames = extract_frames_by_indices(
                    os.path.join(video_folder, str(row["video_path"])), frame_indices
                )
                for first, second in itertools.combinations(candidates, 2):
                    try:
                        forward = model.score_choice_letters(
                            frames,
                            [{"role": "user", "content": pairwise_prompt(
                                str(row["question"]), options[first], options[second]
                            )}],
                            letters=("1", "2"),
                        )
                        reverse = model.score_choice_letters(
                            frames,
                            [{"role": "user", "content": pairwise_prompt(
                                str(row["question"]), options[second], options[first]
                            )}],
                            letters=("1", "2"),
                        )
                        margin = 0.5 * (
                            (forward["1"] - forward["2"])
                            + (reverse["2"] - reverse["1"])
                        )
                        pair_margins[f"{first}_vs_{second}"] = float(margin)
                        pair_records.append(
                            {
                                "first": first,
                                "second": second,
                                "forward_logprobs": forward,
                                "reverse_logprobs": reverse,
                                "order_balanced_margin": float(margin),
                            }
                        )
                    except RuntimeError as error:
                        failed_pairs += 1
                        pair_records.append(
                            {"first": first, "second": second, "error": str(error)}
                        )
                if pair_margins and failed_pairs == 0:
                    answer, totals = aggregate_pairwise_margins(candidates, pair_margins)
                else:
                    answer = fallback_answer
                    totals = {candidate: 0.0 for candidate in candidates}

            pred = build_prediction_row(row, answer, prompt_variant="pairwise_text_tournament")
            pred.update(
                {
                    "candidate_answers": proposed,
                    "distinct_candidate_answers": candidates,
                    "fallback_answer": fallback_answer,
                    "pairwise_tournament_applied": len(candidates) > 1,
                    "pairwise_tournament_failed_pairs": failed_pairs,
                    "pairwise_tournament_pairs": pair_records,
                    "pairwise_tournament_totals": totals,
                    "pairwise_tournament_frames": len(frame_indices),
                    "pairwise_tournament_fingerprint": fingerprint,
                }
            )
            handle.write(json.dumps(pred) + "\n")
            handle.flush()
            print(f"  Tournament progress: {index + 1}/{len(rows)}")

    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
