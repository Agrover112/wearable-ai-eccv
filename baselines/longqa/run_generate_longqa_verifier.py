#!/usr/bin/env python3
"""Verify disagreements between two completed LongQA candidate runs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from typing import Any

from longqa_utils import (
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    index_row_aligned_metadata,
    normalize_answer,
    parse_mcq_options,
    sample_key,
)
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices, load_jsonl
from run_generate_longqa_likelihood import build_scoring_prompt
from run_generate_longqa_proofpack import baseline_uniform_indices, compile_temporal_program


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _answer(row: dict[str, Any]) -> str:
    return str(row.get("mcq_answer_parsed") or row.get("mcq_answer", "")).strip().upper()


def _load_by_key(path: str) -> dict[str, dict[str, Any]]:
    return {sample_key(row): row for row in load_jsonl(path)}


def _video_metadata(video_path: str) -> tuple[float, int]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return 0.0, 0
        return float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def build_verifier_frame_indices(
    proofpack_selected: list[dict[str, Any]],
    total_frames: int,
    max_frames: int = 64,
    proofpack_quota: int = 32,
) -> tuple[list[int], dict[str, Any]]:
    """Combine high-priority proof evidence with exact uniform coverage."""
    priority = {
        "pivot": 0,
        "directional_target": 0,
        "bridge": 1,
        "pivot_context": 1,
        "directional_target_context": 1,
        "event_center": 2,
        "event_center_context": 2,
        "anchor": 3,
        "coverage_fill": 4,
        "semantic_boundary": 5,
    }
    ranked_proof = sorted(
        proofpack_selected,
        key=lambda item: (
            priority.get(str(item.get("source", "")), 6),
            -(float(item["score"]) if item.get("score") is not None else -1e9),
            float(item.get("timestamp", 0.0)),
        ),
    )
    proof_primary = [int(item["frame_index"]) for item in ranked_proof[:proofpack_quota]]
    proof_remaining = [int(item["frame_index"]) for item in ranked_proof[proofpack_quota:]]
    uniform_all = baseline_uniform_indices(total_frames, max_frames)
    uniform_primary = uniform_all[::2]
    uniform_remaining = [idx for idx in uniform_all if idx not in set(uniform_primary)]

    chosen: list[int] = []
    seen: set[int] = set()
    for pool in (proof_primary, uniform_primary, uniform_remaining, proof_remaining):
        for index in pool:
            if index not in seen:
                seen.add(index)
                chosen.append(index)
                if len(chosen) >= max_frames:
                    break
        if len(chosen) >= max_frames:
            break
    chosen.sort()
    return chosen, {
        "proofpack_primary": len(set(proof_primary) & set(chosen)),
        "uniform_primary": len(set(uniform_primary) & set(chosen)),
        "final_frames": len(chosen),
    }


def build_verifier_prompt(
    row: dict[str, Any], first: str, second: str, variant: str = "baseline"
) -> str:
    if variant == "support_contradiction":
        instruction = (
            "Two independent visual evidence passes disagreed and proposed options "
            f"{first} and {second}. Internally assess every option using three checks: "
            "visible supporting evidence, visible contradictory evidence, and whether "
            "the required temporal order is satisfied. Prefer an option only when its "
            "support survives the contradiction and order checks. Do not assume either "
            "proposed option is correct. Return only the final option letter."
        )
    else:
        instruction = (
            "Two independent visual evidence passes disagreed and proposed options "
            f"{first} and {second}. Re-evaluate the complete question using the supplied "
            "chronological evidence. Compare all four options, verify temporal order, "
            "and reject visually unsupported alternatives. Do not assume either proposed "
            "option is correct. Return only the final option letter."
        )
    return instruction + "\n\n" + build_longqa_prompt(
        row["question"], row["mcq_options"], prompt_variant="baseline"
    )


def build_pairwise_verifier_prompt(
    row: dict[str, Any], first: str, second: str
) -> str:
    options = parse_mcq_options(row["mcq_options"])
    if first not in options or second not in options:
        raise ValueError("pairwise verifier candidates must be valid MCQ letters")
    return (
        "Use the chronological video evidence to decide which of exactly two "
        "candidate answers is better supported. Check visible support, contradiction, "
        "and required temporal order. You must select one of the two candidates; do "
        "not propose another answer.\n\n"
        f"Question: {row['question']}\n\n"
        f"Candidate 1: {options[first]}\n"
        f"Candidate 2: {options[second]}\n\n"
        "Return ONLY 1 or 2."
    )


def build_candidate_pair_prompt(
    row: dict[str, Any], first: str, second: str
) -> str:
    """Build a source-neutral prompt containing only the two proposed answers."""
    options = parse_mcq_options(row["mcq_options"])
    candidates = sorted({first, second})
    if len(candidates) != 2 or any(letter not in options for letter in candidates):
        raise ValueError("candidate-pair verifier requires two valid MCQ letters")
    candidate_text = "\n".join(
        f"{letter}. {options[letter]}" for letter in candidates
    )
    return (
        "Two systems gave different answers to the question below. Use the "
        "chronological video frames to choose between only the two listed answers. "
        "Check what is visibly supported, what is contradicted, and whether events "
        "occur in the order asked by the question. Do not choose an unlisted answer.\n\n"
        f"Question: {row['question']}\n\n"
        f"Candidate answers:\n{candidate_text}\n\n"
        f"Return ONLY {candidates[0]} or {candidates[1]}."
    )


def parse_candidate_pair_answer(
    response: object, first: str, second: str
) -> str | None:
    answer = normalize_answer(response)
    return answer if answer in {first, second} else None


def parse_pairwise_choice(response: object) -> int | None:
    text = str(response).strip()
    if text in {"1", "2"}:
        return int(text)
    import re

    matches = re.findall(r"\b([12])\b", text)
    return int(matches[-1]) if matches else None


def rotate_mcq_options(row: dict[str, Any], offset: int) -> tuple[dict[str, Any], dict[str, str]]:
    """Cyclically rotate option semantics and return displayed->original mapping."""
    options = parse_mcq_options(row["mcq_options"])
    if set(options) != set("ABCD"):
        raise ValueError("option permutation requires exactly A/B/C/D options")
    original_order = list("ABCD")
    rotated = original_order[offset % 4 :] + original_order[: offset % 4]
    displayed_to_original = dict(zip(original_order, rotated))
    permuted = dict(row)
    permuted["mcq_options"] = " ".join(
        f"{displayed}. {options[original]}"
        for displayed, original in displayed_to_original.items()
    )
    return permuted, displayed_to_original


def choose_candidate_from_votes(
    votes: list[str], first: str, second: str
) -> tuple[str, dict[str, int]]:
    counts = Counter(vote for vote in votes if vote and vote in "ABCD")
    first_count = counts[first]
    second_count = counts[second]
    answer = second if second_count > first_count else first
    return answer, {letter: counts[letter] for letter in "ABCD"}


def choose_candidate_from_scores(
    first: str,
    second: str,
    primary_scores: dict[str, float],
    secondary_scores: dict[str, float],
    blind_scores: dict[str, float],
    blind_weight: float,
) -> tuple[str, dict[str, float]]:
    scores = {
        letter: float(
            0.5 * (primary_scores[letter] + secondary_scores[letter])
            - blind_weight * blind_scores[letter]
        )
        for letter in "ABCD"
    }
    answer = second if scores[second] > scores[first] else first
    return answer, scores


def _fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "primary": os.path.abspath(args.primary_predictions),
        "secondary": os.path.abspath(args.secondary_predictions),
        "proofpack": os.path.abspath(args.primary_proofpack),
        "model": args.llm_model,
        "max_frames": args.max_frames,
        "proofpack_quota": args.proofpack_quota,
        "verify_operators": sorted(args.verify_operators or []),
        "verifier_prompt": args.verifier_prompt,
        "blind_weight": args.blind_weight,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode()).hexdigest()[:12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify LongQA candidate disagreements.")
    parser.add_argument(
        "--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--primary-predictions", required=True)
    parser.add_argument("--secondary-predictions", required=True)
    parser.add_argument("--primary-proofpack", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--proofpack-quota", type=int, default=32)
    parser.add_argument(
        "--verify-operators",
        nargs="+",
        choices=["AFTER", "BEFORE", "FIRST", "LAST", "STATE_CHANGE", "GLOBAL"],
        default=None,
        help="Only call the verifier for these operators; copy the primary otherwise.",
    )
    parser.add_argument(
        "--verifier-prompt",
        choices=[
            "baseline",
            "support_contradiction",
            "candidate_pair",
            "pairwise_order_swap",
            "candidate_likelihood",
            "option_permutation",
        ],
        default="baseline",
    )
    parser.add_argument(
        "--blind-weight",
        type=float,
        default=-0.1,
        help="Blind-language score weight used by candidate_likelihood.",
    )
    parser.add_argument("--llm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--backend", default="vllm", choices=["hf", "vllm"])
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-resume-predictions", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    import time

    from model import create_model, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    eval_output = _resolve_path(args.eval_output) if args.eval_output else None
    primary_path = _resolve_path(args.primary_predictions)
    secondary_path = _resolve_path(args.secondary_predictions)
    proofpack_path = _resolve_path(args.primary_proofpack)
    fingerprint = _fingerprint(args)

    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    primary_rows = load_jsonl(primary_path)
    primary = {sample_key(row): row for row in primary_rows}
    secondary = _load_by_key(secondary_path)
    proofpack_rows = load_jsonl(proofpack_path)
    proofpack = index_row_aligned_metadata(proofpack_rows, primary_rows, "primary proof pack")
    missing = [
        sample_key(row)
        for row in rows
        if sample_key(row) not in primary or sample_key(row) not in secondary
    ]
    if missing:
        raise RuntimeError(f"Candidate predictions missing {len(missing)} required rows")

    disagreements = [
        row for row in rows if _answer(primary[sample_key(row)]) != _answer(secondary[sample_key(row)])
    ]
    missing_proof = [
        sample_key(row)
        for row in disagreements
        if sample_key(row) not in proofpack
    ]
    if missing_proof:
        raise RuntimeError(f"Primary proof pack missing {len(missing_proof)} disagreement rows")
    print(
        f"Verifier config: rows={len(rows)}, disagreements={len(disagreements)}, "
        f"max_frames={args.max_frames}, proofpack_quota={args.proofpack_quota}, "
        f"fingerprint={fingerprint}"
    )

    existing = []
    if not args.no_resume_predictions and os.path.exists(output_path):
        existing = load_jsonl(output_path)
    start = 0
    for idx, pred in enumerate(existing[: len(rows)]):
        if sample_key(pred) != sample_key(rows[idx]):
            break
        if str(pred.get("verifier_fingerprint", "")) != fingerprint:
            break
        start += 1
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mode = "a" if start else "w"
    model = create_model(
        "qwen",
        args.llm_model,
        backend=args.backend,
        tp_size=args.tp,
        concurrency=args.concurrency,
        max_frames=args.max_frames,
    )
    reset_prompt_token_stats()
    begun = time.time()
    verified = 0
    gated = 0
    with model, open(output_path, mode) as handle:
        for idx, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            first = _answer(primary[key])
            second = _answer(secondary[key])
            operator = compile_temporal_program(row["question"]).operator
            operator_allowed = not args.verify_operators or operator in args.verify_operators
            if first == second or not operator_allowed:
                pred = build_prediction_row(row, first, prompt_variant="verifier_agreement")
                pred["verifier_applied"] = False
                pred["verifier_frames"] = 0
                pred["verifier_gated"] = first != second and not operator_allowed
                gated += int(pred["verifier_gated"])
            else:
                video_path = os.path.join(video_folder, str(row["video_path"]))
                _fps, total_frames = _video_metadata(video_path)
                if args.verifier_prompt == "candidate_likelihood":
                    frame_indices: list[int] = []
                    frame_meta: dict[str, Any] = {}
                    frames: list[Any] = []
                else:
                    frame_indices, frame_meta = build_verifier_frame_indices(
                        proofpack[key]["selected"],
                        total_frames,
                        max_frames=args.max_frames,
                        proofpack_quota=args.proofpack_quota,
                    )
                    frames = extract_frames_by_indices(video_path, frame_indices)
                if args.verifier_prompt == "candidate_likelihood":
                    primary_indices = sorted(
                        int(item["frame_index"])
                        for item in proofpack[key]["selected"][: args.max_frames]
                    )
                    secondary_indices = baseline_uniform_indices(total_frames, args.max_frames)
                    primary_frames = extract_frames_by_indices(video_path, primary_indices)
                    secondary_frames = extract_frames_by_indices(video_path, secondary_indices)
                    prompt = [{"role": "user", "content": build_scoring_prompt(row, True)}]
                    primary_scores = model.score_choice_letters(primary_frames, prompt)
                    secondary_scores = model.score_choice_letters(secondary_frames, prompt)
                    blind_scores = model.score_choice_letters(
                        [],
                        [{"role": "user", "content": build_scoring_prompt(row, False)}],
                    )
                    answer, fused_scores = choose_candidate_from_scores(
                        first,
                        second,
                        primary_scores,
                        secondary_scores,
                        blind_scores,
                        args.blind_weight,
                    )
                    pred = build_prediction_row(
                        row, answer, prompt_variant="candidate_likelihood_verifier"
                    )
                    pred["candidate_primary_option_logprobs"] = primary_scores
                    pred["candidate_secondary_option_logprobs"] = secondary_scores
                    pred["candidate_blind_option_logprobs"] = blind_scores
                    pred["candidate_fused_option_scores"] = fused_scores
                    pred["candidate_blind_weight"] = args.blind_weight
                    pred["candidate_selected_from"] = [first, second]
                    pred["verifier_frame_meta"] = {
                        "primary_frames": len(primary_frames),
                        "secondary_frames": len(secondary_frames),
                    }
                elif args.verifier_prompt == "option_permutation":
                    mapped_votes: list[str] = []
                    raw_responses: list[str] = []
                    mappings: list[dict[str, str]] = []
                    for offset in range(4):
                        permuted_row, displayed_to_original = rotate_mcq_options(row, offset)
                        response = model.generate(
                            frames,
                            [
                                {
                                    "role": "user",
                                    "content": build_longqa_prompt(
                                        permuted_row["question"],
                                        permuted_row["mcq_options"],
                                        prompt_variant="baseline",
                                    ),
                                }
                            ],
                            max_new_tokens=8,
                        )
                        displayed_answer = normalize_answer(response)
                        mapped_votes.append(displayed_to_original.get(displayed_answer, ""))
                        raw_responses.append(str(response))
                        mappings.append(displayed_to_original)
                    answer, vote_counts = choose_candidate_from_votes(
                        mapped_votes, first, second
                    )
                    pred = build_prediction_row(
                        row, answer, prompt_variant="option_permutation_verifier"
                    )
                    pred["permutation_raw_responses"] = raw_responses
                    pred["permutation_mapped_votes"] = mapped_votes
                    pred["permutation_vote_counts"] = vote_counts
                    pred["permutation_mappings"] = mappings
                    pred["permutation_candidate_restricted"] = True
                    pred["permutation_fallback_primary"] = (
                        vote_counts[first] == vote_counts[second]
                    )
                    pred["verifier_frame_meta"] = frame_meta
                elif args.verifier_prompt == "pairwise_order_swap":
                    forward_raw = model.generate(
                        frames,
                        [{"role": "user", "content": build_pairwise_verifier_prompt(row, first, second)}],
                        max_new_tokens=4,
                    )
                    reverse_raw = model.generate(
                        frames,
                        [{"role": "user", "content": build_pairwise_verifier_prompt(row, second, first)}],
                        max_new_tokens=4,
                    )
                    forward_choice = parse_pairwise_choice(forward_raw)
                    reverse_choice = parse_pairwise_choice(reverse_raw)
                    forward_answer = (
                        [first, second][forward_choice - 1] if forward_choice else None
                    )
                    reverse_answer = (
                        [second, first][reverse_choice - 1] if reverse_choice else None
                    )
                    consensus = (
                        forward_answer
                        if forward_answer is not None and forward_answer == reverse_answer
                        else first
                    )
                    pred = build_prediction_row(
                        row, consensus, prompt_variant="pairwise_order_swap_verifier"
                    )
                    pred["pairwise_forward_raw"] = str(forward_raw)
                    pred["pairwise_reverse_raw"] = str(reverse_raw)
                    pred["pairwise_forward_answer"] = forward_answer
                    pred["pairwise_reverse_answer"] = reverse_answer
                    pred["pairwise_consensus"] = (
                        forward_answer is not None and forward_answer == reverse_answer
                    )
                    pred["pairwise_fallback_primary"] = not pred["pairwise_consensus"]
                elif args.verifier_prompt == "candidate_pair":
                    response = model.generate(
                        frames,
                        [
                            {
                                "role": "user",
                                "content": build_candidate_pair_prompt(
                                    row, first, second
                                ),
                            }
                        ],
                        max_new_tokens=8,
                    )
                    parsed = parse_candidate_pair_answer(response, first, second)
                    answer = parsed or first
                    pred = build_prediction_row(
                        row, answer, prompt_variant="candidate_pair_verifier"
                    )
                    pred["candidate_pair_raw_response"] = str(response)
                    pred["candidate_pair_parsed"] = parsed
                    pred["candidate_pair_fallback_primary"] = parsed is None
                    pred["candidate_pair_source_neutral"] = True
                    pred["verifier_frame_meta"] = frame_meta
                else:
                    response = model.generate(
                        frames,
                        [
                            {
                                "role": "user",
                                "content": build_verifier_prompt(
                                    row, first, second, args.verifier_prompt
                                ),
                            }
                        ],
                        max_new_tokens=16,
                    )
                    pred = build_prediction_row(
                        row, response, prompt_variant="disagreement_verifier"
                    )
                pred["verifier_applied"] = True
                pred["verifier_frames"] = (
                    args.max_frames * 2
                    if args.verifier_prompt == "candidate_likelihood"
                    else len(frames)
                )
                pred.setdefault("verifier_frame_meta", frame_meta)
                pred["verifier_gated"] = False
                verified += 1
            pred["verifier_operator"] = operator
            pred["candidate_answers"] = [first, second]
            pred["verifier_fingerprint"] = fingerprint
            handle.write(json.dumps(pred) + "\n")
            handle.flush()
            print(f"  Verifier progress: {idx + 1}/{len(rows)}")

    print(f"Verifier calls this invocation: {verified}")
    print(f"Gated disagreements copied from primary: {gated}")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
