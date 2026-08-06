#!/usr/bin/env python3
"""Use a larger VLM to reconsider disagreements among three LongQA candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import time
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
from run_generate_longqa import (
    _complete_missing_final_answers,
    _print_context_summary,
    has_final_answer_marker,
    has_unambiguous_final_answer,
)
from run_generate_longqa_grounded import extract_frames_by_indices, load_jsonl
from run_generate_longqa_proofpack import baseline_uniform_indices
from run_generate_longqa_uncertainty import _index_jsonl, _video_metadata


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _answer(row: dict[str, Any]) -> str:
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))


def _uncertainty_priority(selection: dict[str, Any]) -> list[int]:
    metadata = selection.get("metadata") or {}
    priority = [
        *metadata.get("target_centers", []),
        *metadata.get("pivot_centers", []),
    ]
    candidates = metadata.get("candidate_indices", [])
    scores = metadata.get("target_scores", [])
    ranked = sorted(
        zip(candidates, scores),
        key=lambda pair: float(pair[1].get("entropy_lower_bound", 0.0)),
        reverse=True,
    )
    priority.extend(frame_index for frame_index, _score in ranked)
    priority.extend(selection.get("final_indices", []))
    return [int(value) for value in priority]


def build_combined_indices(
    selected: list[dict[str, Any]],
    uncertainty: dict[str, Any],
    total_frames: int,
    max_frames: int,
    pivot_quota: int,
    uniform_quota: int,
    uncertainty_quota: int,
) -> tuple[list[int], dict[str, int]]:
    pivot = [int(item["frame_index"]) for item in selected]
    uniform = baseline_uniform_indices(total_frames, max_frames)
    uncertain = _uncertainty_priority(uncertainty)
    pools = (
        ("pivot", pivot[:pivot_quota]),
        ("uniform", baseline_uniform_indices(total_frames, uniform_quota)),
        ("uncertainty", uncertain[:uncertainty_quota]),
        ("uniform_fill", uniform),
        ("pivot_fill", pivot),
        ("uncertainty_fill", uncertain),
    )
    chosen: list[int] = []
    seen: set[int] = set()
    contribution: Counter[str] = Counter()
    for source, indices in pools:
        for frame_index in indices:
            if frame_index < 0 or frame_index >= total_frames or frame_index in seen:
                continue
            seen.add(frame_index)
            chosen.append(frame_index)
            contribution[source] += 1
            if len(chosen) >= max_frames:
                break
        if len(chosen) >= max_frames:
            break
    return sorted(chosen), dict(contribution)


def build_balanced_option_sparse_indices(
    proofpack: dict[str, Any],
    total_frames: int,
    max_frames: int,
    global_quota: int,
    neighbor_radius: int,
) -> tuple[list[int], dict[str, int]]:
    """Build a sparse pack without rewarding an option through repetition."""
    metadata = proofpack.get("selection_meta") or {}
    quotas = metadata.get("target_quota_centers") or {}
    candidate_frames = max(2, int(proofpack.get("candidate_frames") or 128))
    candidate_step = (total_frames - 1) / (candidate_frames - 1)
    center_groups: list[tuple[str, list[int]]] = [
        ("pivot", [int(value) for value in metadata.get("pivot_centers", [])[:2]]),
    ]
    for letter in "ABCD":
        center_groups.append(
            (
                f"option_{letter}",
                [int(value) for value in quotas.get(f"option_{letter}", [])[:1]],
            )
        )
    center_groups.append(
        ("question_target", [int(value) for value in quotas.get("target", [])[:4]])
    )

    chosen: list[int] = []
    seen: set[int] = set()
    contribution: Counter[str] = Counter()

    def add(frame_index: int, source: str) -> None:
        frame_index = max(0, min(total_frames - 1, int(frame_index)))
        if frame_index in seen or len(chosen) >= max_frames:
            return
        seen.add(frame_index)
        chosen.append(frame_index)
        contribution[source] += 1

    # Add every center once before adding context, ensuring one option cannot
    # dominate merely because retrieval returned more nearby frames for it.
    for source, centers in center_groups:
        for center in centers:
            add(center, source)
    for distance in range(1, neighbor_radius + 1):
        for source, centers in center_groups:
            for center in centers:
                add(round(center - distance * candidate_step), f"{source}_context")
                add(round(center + distance * candidate_step), f"{source}_context")

    for frame_index in baseline_uniform_indices(total_frames, global_quota):
        add(frame_index, "global")
    # If retrieval found few centers, complete the fixed budget with global
    # coverage instead of low-ranked semantic frames.
    for frame_index in baseline_uniform_indices(total_frames, max_frames):
        add(frame_index, "global_fill")
    return sorted(chosen), dict(contribution)


def build_judge_prompt(
    row: dict[str, Any],
    answers: dict[str, str],
    show_candidate_suggestions: bool = True,
    require_final_answer_marker: bool = False,
    frame_timestamps: list[float] | None = None,
    balanced_sparse_evidence: bool = False,
) -> str:
    options = parse_mcq_options(str(row["mcq_options"]))
    if show_candidate_suggestions:
        proposed = Counter(answers.values())
        suggestions = "; ".join(
            f"{count} run{'s' if count != 1 else ''} selected {letter}: {options[letter]}"
            for letter, count in sorted(proposed.items())
        )
        instruction = (
            "Several independent video passes disagreed. Their answers are suggestions, "
            "not restrictions: " + suggestions + ". Re-answer the question from the "
            "chronological frames. Identify the referenced events, distinguish repeated "
            "occurrences, and verify any before/after/first/last relationship. Consider all "
            "four options, including options no earlier run selected. Return only the final "
            "option letter."
        )
    else:
        instruction = (
            "Answer the question independently from the chronological video frames. "
            "Identify the referenced events, distinguish repeated occurrences, and "
            "verify any before/after/first/last relationship. Consider all four options "
            "and return only the final option letter."
        )
    if require_final_answer_marker:
        instruction = instruction.replace(
            "return only the final option letter",
            "reason carefully and end with exactly `Final Answer: X`, where X is A, B, C, or D",
        ).replace(
            "Return only the final option letter",
            "Reason carefully and end with exactly `Final Answer: X`, where X is A, B, C, or D",
        )
    if frame_timestamps:
        timestamp_index = ", ".join(
            f"image {index}={timestamp:.1f}s"
            for index, timestamp in enumerate(frame_timestamps, 1)
        )
        instruction += (
            "\nThe timestamp of each image, measured from the beginning of the "
            f"video, is: {timestamp_index}. Use these timestamps to distinguish "
            "repeated events and verify temporal order."
        )
    if balanced_sparse_evidence:
        instruction += (
            "\nThe evidence pack gives each answer option an equal retrieval "
            "opportunity and includes limited global context. Repeated or nearby "
            "images are context for one event; their frequency is not evidence "
            "that an answer is correct. Base the answer on visible facts and "
            "temporal order."
        )
    return instruction + "\n\n" + build_longqa_prompt(
        str(row["question"]), str(row["mcq_options"]), prompt_variant="baseline"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--candidate-predictions", action="append", required=True)
    parser.add_argument("--candidate-labels", nargs="+", required=True)
    parser.add_argument("--fallback-predictions", required=True)
    parser.add_argument("--proofpack", required=True)
    parser.add_argument("--proofpack-reference", required=True)
    parser.add_argument("--uncertainty-selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="vllm")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--pivot-quota", type=int, default=24)
    parser.add_argument("--uniform-quota", type=int, default=24)
    parser.add_argument("--uncertainty-quota", type=int, default=16)
    parser.add_argument(
        "--evidence-strategy",
        choices=("mixed", "balanced_option_sparse"),
        default="mixed",
    )
    parser.add_argument("--sparse-global-quota", type=int, default=8)
    parser.add_argument("--sparse-neighbor-radius", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--thinking-token-budget",
        type=int,
        default=None,
        help="Optional bounded reasoning budget passed to the vLLM request.",
    )
    parser.add_argument(
        "--require-final-answer-marker",
        action="store_true",
        help="Retry answer-only generation if reasoning omits `Final Answer: X`.",
    )
    parser.add_argument(
        "--include-frame-timestamps",
        action="store_true",
        help="List each final image's video timestamp in the answer prompt.",
    )
    parser.add_argument("--judge-all", action="store_true")
    parser.add_argument(
        "--hide-candidate-suggestions",
        action="store_true",
        help="Do not expose previous model answers in the judge prompt.",
    )
    parser.add_argument(
        "--judge-plurality-ties-only",
        action="store_true",
        help="Call the judge only when multiple answers share the highest vote count.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()
    if len(args.candidate_predictions) != len(args.candidate_labels):
        parser.error("candidate prediction and label counts must match")
    if len(args.candidate_predictions) < 2:
        parser.error("at least two candidate prediction sets are required")
    if (
        args.evidence_strategy == "mixed"
        and args.pivot_quota + args.uniform_quota + args.uncertainty_quota
        > args.max_frames
    ):
        parser.error("evidence quotas cannot exceed --max-frames")
    if not 0 <= args.sparse_global_quota <= args.max_frames:
        parser.error("--sparse-global-quota must be within the frame budget")
    if args.sparse_neighbor_radius < 0:
        parser.error("--sparse-neighbor-radius cannot be negative")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.thinking_token_budget is not None and args.thinking_token_budget < 0:
        parser.error("--thinking-token-budget cannot be negative")
    return args


def _fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "candidates": [os.path.abspath(path) for path in args.candidate_predictions],
        "fallback": os.path.abspath(args.fallback_predictions),
        "proofpack": os.path.abspath(args.proofpack),
        "uncertainty": os.path.abspath(args.uncertainty_selection),
        "model": args.llm_model,
        "max_frames": args.max_frames,
        "quotas": [args.pivot_quota, args.uniform_quota, args.uncertainty_quota],
        "evidence_strategy": args.evidence_strategy,
        "sparse_global_quota": args.sparse_global_quota,
        "sparse_neighbor_radius": args.sparse_neighbor_radius,
        "judge_all": args.judge_all,
        "hide_candidate_suggestions": args.hide_candidate_suggestions,
        "judge_plurality_ties_only": args.judge_plurality_ties_only,
        "max_new_tokens": args.max_new_tokens,
        "thinking_token_budget": args.thinking_token_budget,
        "require_final_answer_marker": args.require_final_answer_marker,
        "subset": os.path.abspath(args.subset_file) if args.subset_file else None,
    }
    if args.include_frame_timestamps:
        payload["include_frame_timestamps"] = True
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def main() -> None:
    from model import create_model, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa_grounded import _run_eval

    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    candidate_sets = {
        label: _index_jsonl(_resolve_path(path))
        for label, path in zip(args.candidate_labels, args.candidate_predictions)
    }
    fallback = _index_jsonl(_resolve_path(args.fallback_predictions))
    reference_rows = load_jsonl(_resolve_path(args.proofpack_reference))
    proofpacks = index_row_aligned_metadata(
        load_jsonl(_resolve_path(args.proofpack)), reference_rows, "judge proof pack"
    )
    uncertainty = _index_jsonl(_resolve_path(args.uncertainty_selection))
    required = {sample_key(row) for row in rows}
    for label, indexed in {
        **candidate_sets,
        "fallback": fallback,
        "proofpack": proofpacks,
        "uncertainty": uncertainty,
    }.items():
        missing = required - set(indexed)
        if missing:
            raise RuntimeError(f"{label} is missing {len(missing)} required rows")

    fingerprint = _fingerprint(args)
    existing = [] if args.no_resume or not os.path.exists(output_path) else load_jsonl(output_path)
    start = 0
    for index, record in enumerate(existing[: len(rows)]):
        if (
            sample_key(record) != sample_key(rows[index])
            or record.get("multicandidate_judge_fingerprint") != fingerprint
        ):
            break
        start += 1
    if len(existing) != start:
        with open(output_path, "w") as handle:
            for record in existing[:start]:
                handle.write(json.dumps(record) + "\n")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def should_judge_row(row: dict[str, Any]) -> bool:
        answers = [
            _answer(indexed[sample_key(row)]) for indexed in candidate_sets.values()
        ]
        counts = Counter(answers)
        if args.judge_all:
            return True
        if len(counts) <= 1:
            return False
        if not args.judge_plurality_ties_only:
            return True
        maximum = max(counts.values())
        return sum(count == maximum for count in counts.values()) > 1

    judge_count = sum(should_judge_row(row) for row in rows)
    print(
        f"Multi-candidate judge: rows={len(rows)}, calls={judge_count}, resume={start}, "
        f"model={args.llm_model}, fingerprint={fingerprint}"
    )
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
    calls = 0
    with model, open(output_path, "a" if start else "w") as handle:
        for index, row in enumerate(rows[start:], start=start):
            row_begun = time.perf_counter()
            key = sample_key(row)
            answers = {label: _answer(indexed[key]) for label, indexed in candidate_sets.items()}
            should_judge = should_judge_row(row)
            fallback_answer = _answer(fallback[key])
            if should_judge:
                video_path = os.path.join(video_folder, str(row["video_path"]))
                _fps, total_frames = _video_metadata(video_path)
                if total_frames <= 0:
                    raise RuntimeError(f"Could not read video metadata: {video_path}")
                if args.evidence_strategy == "balanced_option_sparse":
                    indices, contribution = build_balanced_option_sparse_indices(
                        proofpacks[key],
                        total_frames,
                        args.max_frames,
                        args.sparse_global_quota,
                        args.sparse_neighbor_radius,
                    )
                else:
                    indices, contribution = build_combined_indices(
                        proofpacks[key]["selected"],
                        uncertainty[key],
                        total_frames,
                        args.max_frames,
                        args.pivot_quota,
                        args.uniform_quota,
                        args.uncertainty_quota,
                    )
                frames = extract_frames_by_indices(video_path, indices)
                frame_timestamps = (
                    [frame_index / max(_fps, 1e-6) for frame_index in indices]
                    if args.include_frame_timestamps
                    else None
                )
                messages = [
                    {
                        "role": "user",
                        "content": build_judge_prompt(
                            row,
                            answers,
                            show_candidate_suggestions=(
                                not args.hide_candidate_suggestions
                            ),
                            require_final_answer_marker=(
                                args.require_final_answer_marker
                            ),
                            frame_timestamps=frame_timestamps,
                            balanced_sparse_evidence=(
                                args.evidence_strategy == "balanced_option_sparse"
                            ),
                        ),
                    }
                ]
                raw = str(
                    model.generate(
                        frames,
                        messages,
                        max_new_tokens=args.max_new_tokens,
                        thinking_token_budget=args.thinking_token_budget,
                    )
                )
                completion_error = None
                try:
                    raw = _complete_missing_final_answers(
                        model,
                        [frames],
                        [messages],
                        [raw],
                        args.require_final_answer_marker,
                    )[0]
                except RuntimeError as error:
                    completion_error = str(error)
                parsed = normalize_answer(raw)
                valid_answer = parsed in {"A", "B", "C", "D"} and (
                    not args.require_final_answer_marker
                    or has_unambiguous_final_answer(raw)
                )
                selected = parsed if valid_answer else fallback_answer
                prediction = build_prediction_row(
                    row, selected, prompt_variant="multicandidate_large_judge"
                )
                prediction.update(
                    {
                        "multicandidate_judge_applied": True,
                        "multicandidate_judge_raw": str(raw),
                        "multicandidate_judge_parsed": parsed if valid_answer else None,
                        "multicandidate_judge_fallback": not valid_answer,
                        "multicandidate_judge_candidate_blind": (
                            args.hide_candidate_suggestions
                        ),
                        "multicandidate_judge_max_new_tokens": args.max_new_tokens,
                        "multicandidate_judge_thinking_token_budget": (
                            args.thinking_token_budget
                        ),
                        "multicandidate_judge_final_answer_marker": (
                            has_final_answer_marker(raw)
                        ),
                        "multicandidate_judge_final_answer_validated": (
                            has_unambiguous_final_answer(raw)
                        ),
                        "multicandidate_judge_completion_error": completion_error,
                        "multicandidate_judge_timestamps_in_prompt": (
                            args.include_frame_timestamps
                        ),
                        "multicandidate_judge_frame_timestamps": (
                            frame_timestamps or []
                        ),
                        "multicandidate_judge_frames": indices,
                        "multicandidate_judge_frame_sources": contribution,
                        "multicandidate_judge_evidence_strategy": (
                            args.evidence_strategy
                        ),
                    }
                )
                calls += 1
            else:
                prediction = build_prediction_row(
                    row, fallback_answer, prompt_variant="multicandidate_judge_agreement"
                )
                prediction.update(
                    {
                        "multicandidate_judge_applied": False,
                        "multicandidate_judge_raw": None,
                        "multicandidate_judge_parsed": None,
                        "multicandidate_judge_fallback": False,
                        "multicandidate_judge_candidate_blind": (
                            args.hide_candidate_suggestions
                        ),
                        "multicandidate_judge_frames": [],
                        "multicandidate_judge_frame_sources": {},
                    }
                )
            vote_counts = Counter(answers.values())
            prediction.update(
                {
                    "candidate_answers": answers,
                    "candidate_vote_counts": dict(vote_counts),
                    "candidate_agreement_pattern": (
                        "unanimous"
                        if len(vote_counts) == 1
                        else "split_vote"
                        if max(vote_counts.values()) > 1
                        else "all_different"
                    ),
                    "multicandidate_judge_previous": fallback_answer,
                    "multicandidate_judge_fingerprint": fingerprint,
                    "multicandidate_judge_seconds": round(
                        time.perf_counter() - row_begun, 3
                    ),
                }
            )
            handle.write(json.dumps(prediction) + "\n")
            handle.flush()
            print(f"  Judge progress: {index + 1}/{len(rows)} calls={calls}/{judge_count}")

    print(f"Judge calls this invocation: {calls}")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(
            input_path,
            output_path,
            _resolve_path(args.eval_output) if args.eval_output else None,
        )


if __name__ == "__main__":
    main()
