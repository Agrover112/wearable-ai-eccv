#!/usr/bin/env python3
"""Score answer-option rankings under endpoint and option-quota evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from longqa_utils import apply_subset, build_longqa_prompt, load_jsonl, normalize_answer, sample_key
from run_generate_longqa_grounded import extract_frames_by_indices
from run_generate_longqa_uncertainty import _video_metadata


LETTERS = "ABCD"


def _answer(row: dict[str, Any]) -> str:
    answer = normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))
    if answer not in LETTERS:
        raise ValueError(f"Invalid answer {answer!r} for {sample_key(row)}")
    return answer


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("sample_key") or sample_key(row))
        if key in indexed:
            raise ValueError(f"Duplicate {label} key: {key}")
        indexed[key] = row
    return indexed


def _probabilities(logprobs: dict[str, float]) -> dict[str, float]:
    maximum = max(logprobs.values())
    weights = {letter: math.exp(logprobs[letter] - maximum) for letter in LETTERS}
    total = sum(weights.values())
    return {letter: weights[letter] / total for letter in LETTERS}


def _scored_view(logprobs: dict[str, float]) -> dict[str, Any]:
    probabilities = _probabilities(logprobs)
    ordered = sorted(LETTERS, key=lambda letter: (-probabilities[letter], letter))
    return {
        "logprobs": {letter: float(logprobs[letter]) for letter in LETTERS},
        "probabilities": probabilities,
        "ranking": ordered,
        "ranks": {letter: rank for rank, letter in enumerate(ordered, start=1)},
        "top_answer": ordered[0],
        "top_margin": probabilities[ordered[0]] - probabilities[ordered[1]],
    }


def _fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "baseline": os.path.abspath(args.baseline_predictions),
        "challenger": os.path.abspath(args.challenger_predictions),
        "proofpacks": sorted(os.path.abspath(path) for path in args.option_proofpack),
        "trigger_predictions": sorted(
            os.path.abspath(path) for path in args.trigger_predictions
        ),
        "trigger_min_votes": args.trigger_min_votes,
        "subset": os.path.abspath(args.subset_file) if args.subset_file else None,
        "model": args.llm_model,
        "sampling": "endpoint_inclusive_64",
        "prompt": "baseline_choice_letter_v1",
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--challenger-predictions", required=True)
    parser.add_argument("--option-proofpack", action="append", required=True)
    parser.add_argument(
        "--trigger-predictions",
        action="append",
        default=[],
        help=(
            "Optional independent prediction branch. When the baseline and challenger "
            "agree, score the row if at least --trigger-min-votes branches agree on "
            "the same alternative answer."
        ),
    )
    parser.add_argument("--trigger-min-votes", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _trigger_details(
    key: str,
    baseline_answer: str,
    challenger_answer: str,
    trigger_sources: list[dict[str, dict[str, Any]]],
    minimum_votes: int,
) -> dict[str, Any]:
    votes = [_answer(source[key]) for source in trigger_sources]
    result: dict[str, Any] = {
        "answers": votes,
        "alternative": None,
        "alternative_votes": 0,
        "applied": False,
    }
    if baseline_answer != challenger_answer or not votes:
        return result
    alternatives = Counter(answer for answer in votes if answer != baseline_answer)
    if not alternatives:
        return result
    alternative, count = sorted(alternatives.items(), key=lambda item: (-item[1], item[0]))[0]
    result.update(
        {
            "alternative": alternative,
            "alternative_votes": count,
            "applied": count >= minimum_votes,
        }
    )
    return result


def main() -> None:
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats, uniform_full_video_indices
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    rows = apply_subset(load_jsonl(args.input), args.subset_file)
    baseline = _index(load_jsonl(args.baseline_predictions), "baseline")
    challenger = _index(load_jsonl(args.challenger_predictions), "challenger")
    proofpacks: dict[str, dict[str, Any]] = {}
    for path in args.option_proofpack:
        for key, record in _index(load_jsonl(path), f"proofpack {path}").items():
            if key in proofpacks:
                raise ValueError(f"Proofpack key occurs in multiple files: {key}")
            proofpacks[key] = record
    trigger_sources = [
        _index(load_jsonl(path), f"trigger predictions {path}")
        for path in args.trigger_predictions
    ]
    if trigger_sources and not 1 <= args.trigger_min_votes <= len(trigger_sources):
        raise ValueError(
            f"--trigger-min-votes must be between 1 and {len(trigger_sources)}"
        )
    for row in rows:
        key = sample_key(row)
        if key not in baseline or key not in challenger or key not in proofpacks:
            raise RuntimeError(f"Missing input artifact for {key}")
        if any(key not in source for source in trigger_sources):
            raise RuntimeError(f"Missing trigger prediction for {key}")

    fingerprint = _fingerprint(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = [] if args.no_resume or not output.exists() else load_jsonl(str(output))
    start = 0
    for index, record in enumerate(existing[: len(rows)]):
        if (
            record.get("sample_key") != sample_key(rows[index])
            or record.get("rank_fusion_fingerprint") != fingerprint
        ):
            break
        start += 1
    if len(existing) != start:
        with output.open("w") as handle:
            for record in existing[:start]:
                handle.write(json.dumps(record) + "\n")

    targets = 0
    consensus_challenges = 0
    for row in rows:
        key = sample_key(row)
        baseline_answer = _answer(baseline[key])
        challenger_answer = _answer(challenger[key])
        trigger = _trigger_details(
            key,
            baseline_answer,
            challenger_answer,
            trigger_sources,
            args.trigger_min_votes,
        )
        targets += int(baseline_answer != challenger_answer or trigger["applied"])
        consensus_challenges += int(trigger["applied"])
    print(
        f"Evidence rank scoring: rows={len(rows)} targets={targets} "
        f"consensus_challenges={consensus_challenges} resume={start} "
        f"calls={targets * 2} fingerprint={fingerprint}"
    )
    model = VLLMModel(
        args.llm_model,
        tp_size=1,
        concurrency=args.concurrency,
        max_frames=64,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    begun = time.time()
    calls = 0
    with model, output.open("a" if start else "w") as handle:
        for index, row in enumerate(rows[start:], start=start):
            row_begun = time.perf_counter()
            key = sample_key(row)
            baseline_answer = _answer(baseline[key])
            challenger_answer = _answer(challenger[key])
            trigger = _trigger_details(
                key,
                baseline_answer,
                challenger_answer,
                trigger_sources,
                args.trigger_min_votes,
            )
            should_score = baseline_answer != challenger_answer or trigger["applied"]
            record: dict[str, Any] = {
                "sample_key": key,
                "video_path": row.get("video_path"),
                "question": row.get("question"),
                "baseline_answer": baseline_answer,
                "challenger_answer": challenger_answer,
                "rank_fusion_applied": should_score,
                "direct_disagreement": baseline_answer != challenger_answer,
                "consensus_challenge": trigger,
                "rank_fusion_fingerprint": fingerprint,
            }
            if should_score:
                video_path = os.path.join(args.video_folder, str(row["video_path"]))
                _, total_frames = _video_metadata(video_path)
                endpoint_indices = uniform_full_video_indices(
                    total_frames, 64, 64, "endpoint_inclusive"
                )
                option_indices = [
                    int(item["frame_index"])
                    for item in proofpacks[key].get("selected", [])
                ]
                if len(endpoint_indices) != 64 or len(set(endpoint_indices)) != 64:
                    raise RuntimeError(f"Invalid endpoint frame pack for {key}")
                if len(option_indices) != 64 or len(set(option_indices)) != 64:
                    raise RuntimeError(f"Invalid option-quota frame pack for {key}")
                prompt = build_longqa_prompt(row["question"], row["mcq_options"], "baseline")
                endpoint_frames = extract_frames_by_indices(video_path, endpoint_indices)
                option_frames = extract_frames_by_indices(video_path, option_indices)
                endpoint_scores = model.score_choice_letters(
                    endpoint_frames, [{"role": "user", "content": prompt}]
                )
                option_scores = model.score_choice_letters(
                    option_frames, [{"role": "user", "content": prompt}]
                )
                record.update(
                    {
                        "endpoint_frame_indices": endpoint_indices,
                        "option_frame_indices": option_indices,
                        "endpoint_view": _scored_view(endpoint_scores),
                        "option_view": _scored_view(option_scores),
                    }
                )
                calls += 2
            record["rank_fusion_seconds"] = round(time.perf_counter() - row_begun, 3)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"  Rank progress: {index + 1}/{len(rows)} calls={calls}")

    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())


if __name__ == "__main__":
    main()
