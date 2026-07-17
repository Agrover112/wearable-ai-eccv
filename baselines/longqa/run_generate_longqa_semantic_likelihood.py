#!/usr/bin/env python3
"""Resolve LongQA run disagreements by scoring complete candidate answer text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

from longqa_utils import (
    apply_subset,
    build_prediction_row,
    index_row_aligned_metadata,
    normalize_answer,
    parse_mcq_options,
    sample_key,
)
from run_generate_longqa_event_ledger import _video_metadata
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices, load_jsonl
from run_generate_longqa_proofpack import baseline_uniform_indices


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def build_semantic_scoring_prompt(row: dict[str, Any]) -> str:
    return (
        "Use the supplied chronological video frames to answer the question. "
        "The answer will be scored as a complete text continuation.\n\n"
        f"Question: {row['question']}"
    )


def winning_candidate(scores: dict[str, dict[str, object]]) -> str:
    return max(scores, key=lambda label: float(scores[label]["mean_logprob"]))


def semantic_decision(
    primary_answer: str,
    primary_winner: str,
    uniform_winner: str,
) -> tuple[str, str]:
    if primary_winner == uniform_winner:
        return primary_winner, "context_consensus"
    return primary_answer, "context_disagreement_primary_fallback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl")
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--primary-predictions", required=True)
    parser.add_argument("--secondary-predictions", required=True)
    parser.add_argument("--primary-proofpack", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scores-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--llm-model", default="Qwen/Qwen3-VL-8B-Instruct")
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
    scores_output = _resolve_path(args.scores_output)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    primary_rows = load_jsonl(_resolve_path(args.primary_predictions))
    secondary_rows = load_jsonl(_resolve_path(args.secondary_predictions))
    primary = index_row_aligned_metadata(primary_rows, rows, "primary predictions")
    secondary = index_row_aligned_metadata(secondary_rows, rows, "secondary predictions")
    packs = index_row_aligned_metadata(
        load_jsonl(_resolve_path(args.primary_proofpack)), rows, "primary proof pack"
    )
    fingerprint_payload = {
        "primary": os.path.abspath(_resolve_path(args.primary_predictions)),
        "secondary": os.path.abspath(_resolve_path(args.secondary_predictions)),
        "proofpack": os.path.abspath(_resolve_path(args.primary_proofpack)),
        "model": args.llm_model,
        "max_frames": args.max_frames,
        "scoring": "complete_answer_mean_logprob_separate_contexts_v1",
    }
    fingerprint = hashlib.sha1(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()[:12]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(scores_output) or ".", exist_ok=True)
    existing = [] if args.no_resume or not os.path.exists(output_path) else load_jsonl(output_path)
    score_existing = (
        [] if args.no_resume or not os.path.exists(scores_output) else load_jsonl(scores_output)
    )
    start = 0
    for index, prediction in enumerate(existing[: len(rows)]):
        if sample_key(prediction) != sample_key(rows[index]):
            break
        if prediction.get("semantic_likelihood_fingerprint") != fingerprint:
            break
        start += 1
    scored_prefix = sum(
        normalize_answer(primary[sample_key(row)].get("mcq_answer_parsed"))
        != normalize_answer(secondary[sample_key(row)].get("mcq_answer_parsed"))
        for row in rows[:start]
    )
    if len(score_existing) < scored_prefix:
        raise RuntimeError("Semantic score cache is shorter than the prediction resume prefix")
    if len(existing) != start:
        with open(output_path, "w") as handle:
            for item in existing[:start]:
                handle.write(json.dumps(item) + "\n")
    if len(score_existing) != scored_prefix:
        with open(scores_output, "w") as handle:
            for item in score_existing[:scored_prefix]:
                handle.write(json.dumps(item) + "\n")
    disagreements = sum(
        normalize_answer(primary[sample_key(row)].get("mcq_answer_parsed"))
        != normalize_answer(secondary[sample_key(row)].get("mcq_answer_parsed"))
        for row in rows
    )
    print(
        f"Semantic likelihood: rows={len(rows)}, disagreements={disagreements}, "
        f"resume={start}, fingerprint={fingerprint}"
    )
    model = VLLMModel(
        args.llm_model,
        tp_size=1,
        concurrency=1,
        max_frames=args.max_frames,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    begun = time.time()
    with model, open(output_path, "a" if start else "w") as prediction_handle, open(
        scores_output, "a" if scored_prefix else "w"
    ) as score_handle:
        for row_index, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            primary_answer = normalize_answer(primary[key].get("mcq_answer_parsed"))
            secondary_answer = normalize_answer(secondary[key].get("mcq_answer_parsed"))
            if primary_answer not in "ABCD" or secondary_answer not in "ABCD":
                raise RuntimeError(f"Invalid candidate answer for {key}")
            if primary_answer == secondary_answer:
                prediction = build_prediction_row(
                    row, primary_answer, prompt_variant="semantic_likelihood_agreement_copy"
                )
                prediction.update(
                    {
                        "semantic_likelihood_fingerprint": fingerprint,
                        "semantic_likelihood_decision": "base_agreement",
                    }
                )
            else:
                options = parse_mcq_options(row.get("mcq_options", ""))
                candidate_texts = {
                    primary_answer: options[primary_answer],
                    secondary_answer: options[secondary_answer],
                }
                video_path = os.path.join(video_folder, str(row["video_path"]))
                _, total_frames = _video_metadata(video_path)
                pivot_indices = [
                    int(item["frame_index"])
                    for item in packs[key]["selected"][: args.max_frames]
                ]
                uniform_indices = baseline_uniform_indices(total_frames, args.max_frames)
                prompt = [{"role": "user", "content": build_semantic_scoring_prompt(row)}]
                pivot_scores = model.score_candidate_texts(
                    extract_frames_by_indices(video_path, pivot_indices), prompt, candidate_texts
                )
                uniform_scores = model.score_candidate_texts(
                    extract_frames_by_indices(video_path, uniform_indices), prompt, candidate_texts
                )
                pivot_winner = winning_candidate(pivot_scores)
                uniform_winner = winning_candidate(uniform_scores)
                answer, rule = semantic_decision(primary_answer, pivot_winner, uniform_winner)
                prediction = build_prediction_row(
                    row, answer, prompt_variant="semantic_candidate_likelihood"
                )
                prediction.update(
                    {
                        "semantic_likelihood_fingerprint": fingerprint,
                        "semantic_likelihood_decision": rule,
                        "primary_candidate": primary_answer,
                        "secondary_candidate": secondary_answer,
                        "pivot_context_winner": pivot_winner,
                        "uniform_context_winner": uniform_winner,
                    }
                )
                score_row = {
                    "sample_key": key,
                    "video_path": row.get("video_path", ""),
                    "primary_candidate": primary_answer,
                    "secondary_candidate": secondary_answer,
                    "candidate_texts": candidate_texts,
                    "pivot_indices": pivot_indices,
                    "uniform_indices": uniform_indices,
                    "pivot_context_scores": pivot_scores,
                    "uniform_context_scores": uniform_scores,
                    "pivot_context_winner": pivot_winner,
                    "uniform_context_winner": uniform_winner,
                    "decision": answer,
                    "decision_rule": rule,
                    "semantic_likelihood_fingerprint": fingerprint,
                }
                score_handle.write(json.dumps(score_row) + "\n")
                score_handle.flush()
            prediction_handle.write(json.dumps(prediction) + "\n")
            prediction_handle.flush()
            print(f"  Semantic likelihood progress: {row_index + 1}/{len(rows)}")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(input_path, output_path, _resolve_path(args.eval_output))


if __name__ == "__main__":
    main()
