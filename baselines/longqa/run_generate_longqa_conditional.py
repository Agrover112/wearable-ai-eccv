#!/usr/bin/env python3
"""Run targeted LongQA interventions only when strong systems disagree.

Rows outside the configured disagreement gate copy the deterministic majority
answer. This keeps each output directly comparable with the fixed ensemble
while limiting expensive visual inference to the rows an intervention can
actually change.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from longqa_utils import (
    apply_subset,
    build_prediction_row,
    index_row_aligned_metadata,
    normalize_answer,
    parse_mcq_options,
    sample_key,
)
from run_generate_longqa_grounded import (
    _run_eval,
    extract_frames_by_indices,
    load_jsonl,
)
from run_generate_longqa_object_hints import (
    _top_detections,
    choose_detail_frames,
    crop_detection,
)
from run_generate_longqa_proofpack import resize_frame_to_max_pixels
from run_generate_longqa_uncertainty import (
    _compact_score,
    _index_jsonl,
    _index_proofpack,
    _letter_probability,
    _video_metadata,
    build_crop_context,
    build_entropy_prompt,
    score_collections,
    select_uncertainty_dynamic_tcot,
)
from run_generate_longqa_verifier import build_verifier_frame_indices


MODES = ("candidate_arbiter", "uncertainty_tcot", "delta_crops")


def _resolve_path(path: str | None) -> str | None:
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _answer(row: dict[str, Any]) -> str:
    return str(
        row.get("mcq_answer_parsed") or row.get("mcq_answer", "")
    ).strip().upper()


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def majority_answer(answers: list[str]) -> tuple[str, dict[str, int], bool]:
    counts = Counter(answers)
    top_count = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == top_count}
    chosen = next(answer for answer in answers if answer in tied)
    return chosen, dict(counts), len(tied) > 1


def should_intervene(answers: list[str], scope: str) -> bool:
    unique = len(set(answers))
    if scope == "all_different":
        return unique == len(answers)
    if scope == "any_disagreement":
        return unique > 1
    raise ValueError(f"Unknown disagreement scope: {scope}")


def build_candidate_prompt(row: dict[str, Any], candidates: list[str]) -> str:
    options = parse_mcq_options(row["mcq_options"])
    candidates = sorted(set(candidates))
    if len(candidates) < 2 or any(candidate not in options for candidate in candidates):
        raise ValueError("Candidate arbiter requires at least two valid option letters")
    candidate_text = "\n".join(
        f"{letter}. {options[letter]}" for letter in candidates
    )
    allowed = " or ".join(candidates)
    return (
        "Several strong video-question answering systems disagreed. Use the "
        "chronological visual evidence to choose between only the candidate "
        "answers below. Check visible support, contradiction, and the order of "
        "events requested by the question. Do not choose an unlisted answer.\n\n"
        f"Question: {row['question']}\n\n"
        f"Candidate answers:\n{candidate_text}\n\n"
        f"Return ONLY {allowed}."
    )


def parse_candidate_response(response: object, candidates: list[str]) -> str | None:
    answer = normalize_answer(response)
    return answer if answer in set(candidates) else None


def select_delta_gated_crops(
    model: Any,
    row: dict[str, Any],
    video_path: str,
    selected: list[dict[str, Any]],
    detection_record: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[object], dict[str, Any]]:
    selected_indices = {int(item["frame_index"]) for item in selected}
    candidates = choose_detail_frames(
        [
            item
            for item in detection_record["frames"]
            if int(item["frame_index"]) in selected_indices
        ],
        args.crop_candidates,
    )
    candidate_indices = [int(item["frame_index"]) for item in candidates]
    source_frames = extract_frames_by_indices(video_path, candidate_indices)
    if len(source_frames) != len(candidate_indices):
        raise RuntimeError("Could not extract every delta-crop candidate frame")
    detection_map = {
        int(item["frame_index"]): item.get("detections", [])
        for item in detection_record["frames"]
    }

    descriptors: list[list[int]] = []
    collections: list[list[object]] = []
    valid_candidates: list[dict[str, Any]] = []
    for item, frame in zip(candidates, source_frames):
        frame_index = int(item["frame_index"])
        detections = _top_detections(detection_map.get(frame_index, []), 2)
        if not detections:
            continue
        timestamp = float(item["timestamp"])
        crop = crop_detection(frame, detections[0], timestamp)
        descriptors.extend(([frame_index], [frame_index, -frame_index - 1]))
        collections.extend(
            (
                [resize_frame_to_max_pixels(frame, args.scoring_max_pixels)],
                [
                    resize_frame_to_max_pixels(frame, args.scoring_max_pixels),
                    resize_frame_to_max_pixels(crop, args.scoring_max_pixels),
                ],
            )
        )
        valid_candidates.append(item)

    if not valid_candidates:
        final_frames, context_meta = build_crop_context(
            video_path,
            selected,
            detection_record,
            [],
            args.max_frames,
        )
        return final_frames, {
            "crop_candidates": [],
            "admitted_crop_indices": [],
            "min_entropy_reduction": args.min_entropy_reduction,
            "score_cache_hit": False,
            "no_valid_crop_candidates": True,
            **context_meta,
        }

    scores, cache_hit = score_collections(
        model,
        row,
        descriptors,
        collections,
        build_entropy_prompt(row),
        args,
    )
    candidates_scored: list[dict[str, Any]] = []
    allowed_answers = set(args.current_candidate_answers)
    for position, item in enumerate(valid_candidates):
        source_score = scores[2 * position]
        pair_score = scores[2 * position + 1]
        delta = float(source_score["entropy_lower_bound"]) - float(
            pair_score["entropy_lower_bound"]
        )
        pair_letter = max(
            "ABCD",
            key=lambda letter: _letter_probability(pair_score, letter),
        )
        candidates_scored.append(
            {
                "item": item,
                "source_score": _compact_score(source_score),
                "pair_score": _compact_score(pair_score),
                "entropy_reduction": delta,
                "pair_letter": pair_letter,
                "candidate_consistent": pair_letter in allowed_answers,
            }
        )
    admitted = [
        record
        for record in candidates_scored
        if record["entropy_reduction"] >= args.min_entropy_reduction
        and record["candidate_consistent"]
    ]
    admitted.sort(
        key=lambda record: (
            -float(record["entropy_reduction"]),
            float(record["item"]["timestamp"]),
        )
    )
    chosen = admitted[: args.detail_count]
    detail_frames = sorted(
        [record["item"] for record in chosen],
        key=lambda item: float(item["timestamp"]),
    )
    final_frames, context_meta = build_crop_context(
        video_path,
        selected,
        detection_record,
        detail_frames,
        args.max_frames,
    )
    return final_frames, {
        "crop_candidates": [
            {
                "frame_index": int(record["item"]["frame_index"]),
                "timestamp": float(record["item"]["timestamp"]),
                "entropy_reduction": record["entropy_reduction"],
                "pair_letter": record["pair_letter"],
                "candidate_consistent": record["candidate_consistent"],
                "source_score": record["source_score"],
                "pair_score": record["pair_score"],
            }
            for record in candidates_scored
        ],
        "admitted_crop_indices": [
            int(record["item"]["frame_index"]) for record in chosen
        ],
        "min_entropy_reduction": args.min_entropy_reduction,
        "score_cache_hit": cache_hit,
        **context_meta,
    }


def _fingerprint(args: argparse.Namespace) -> str:
    excluded = {
        "output",
        "eval_output",
        "audit_output",
        "no_resume",
    }
    payload = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded and key != "current_candidate_answers"
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
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--disagreement-scope",
        choices=["all_different", "any_disagreement"],
        default="all_different",
    )
    parser.add_argument("--primary-predictions", required=True)
    parser.add_argument("--secondary-predictions", required=True)
    parser.add_argument("--tertiary-predictions", required=True)
    parser.add_argument("--proofpack", required=True)
    parser.add_argument("--proofpack-reference", required=True)
    parser.add_argument("--detections-input", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--score-cache-dir", default=None)
    parser.add_argument("--tcot-cache-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")

    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--proofpack-quota", type=int, default=32)
    parser.add_argument("--candidate-frames", type=int, default=128)
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--uncertainty-per-segment", type=int, default=4)
    parser.add_argument("--scoring-max-pixels", type=int, default=50176)
    parser.add_argument("--top-logprobs", type=int, default=100)
    parser.add_argument("--tcot-segments", type=int, default=4)
    parser.add_argument("--max-selected-per-segment", type=int, default=6)
    parser.add_argument("--selector-max-pixels", type=int, default=50176)
    parser.add_argument("--neighborhood-radius", type=int, default=1)
    parser.add_argument("--selected-quota", type=int, default=48)
    parser.add_argument("--uniform-quota", type=int, default=16)

    parser.add_argument("--crop-candidates", type=int, default=24)
    parser.add_argument("--detail-count", type=int, default=8)
    parser.add_argument("--min-entropy-reduction", type=float, default=0.05)

    parser.add_argument("--llm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "uncertainty_tcot" and (
        not args.score_cache_dir or not args.tcot_cache_dir
    ):
        parser.error(
            "--score-cache-dir and --tcot-cache-dir are required for uncertainty_tcot"
        )
    if args.mode == "delta_crops" and (
        not args.detections_input or not args.score_cache_dir
    ):
        parser.error(
            "--detections-input and --score-cache-dir are required for delta_crops"
        )
    return args


def main() -> None:
    import time

    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    eval_output = _resolve_path(args.eval_output)
    audit_output = _resolve_path(args.audit_output)
    args.score_cache_dir = _resolve_path(args.score_cache_dir)
    args.tcot_cache_dir = _resolve_path(args.tcot_cache_dir)

    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    prediction_sets = [
        _index_jsonl(_resolve_path(args.primary_predictions)),
        _index_jsonl(_resolve_path(args.secondary_predictions)),
        _index_jsonl(_resolve_path(args.tertiary_predictions)),
    ]
    proofpacks = _index_proofpack(
        _resolve_path(args.proofpack),
        _resolve_path(args.proofpack_reference),
    )
    detections = (
        _index_jsonl(_resolve_path(args.detections_input))
        if args.detections_input
        else {}
    )
    required_keys = {sample_key(row) for row in rows}
    inputs = {
        "primary predictions": prediction_sets[0],
        "secondary predictions": prediction_sets[1],
        "tertiary predictions": prediction_sets[2],
        "proof pack": proofpacks,
    }
    if args.mode == "delta_crops":
        inputs["detections"] = detections
    for label, indexed in inputs.items():
        missing = required_keys - set(indexed)
        if missing:
            raise RuntimeError(f"{label} is missing {len(missing)} required rows")

    fingerprint = _fingerprint(args)
    existing = (
        []
        if args.no_resume or not os.path.exists(output_path)
        else load_jsonl(output_path)
    )
    audits = (
        []
        if args.no_resume or not os.path.exists(audit_output)
        else load_jsonl(audit_output)
    )
    start = 0
    for index, prediction in enumerate(existing[: len(rows)]):
        if (
            sample_key(prediction) != sample_key(rows[index])
            or prediction.get("conditional_fingerprint") != fingerprint
        ):
            break
        start += 1
    audit_start = 0
    for index, record in enumerate(audits[: len(rows)]):
        if (
            record.get("sample_key") != sample_key(rows[index])
            or record.get("conditional_fingerprint") != fingerprint
        ):
            break
        audit_start += 1
    start = min(start, audit_start)
    if len(existing) != start:
        _write_jsonl(output_path, existing[:start])
    if len(audits) != start:
        _write_jsonl(audit_output, audits[:start])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(audit_output) or ".", exist_ok=True)
    if args.score_cache_dir:
        Path(args.score_cache_dir).mkdir(parents=True, exist_ok=True)
    if args.tcot_cache_dir:
        Path(args.tcot_cache_dir).mkdir(parents=True, exist_ok=True)

    gated_count = sum(
        should_intervene(
            [_answer(predictions[sample_key(row)]) for predictions in prediction_sets],
            args.disagreement_scope,
        )
        for row in rows
    )
    print(
        f"Conditional config: mode={args.mode}, rows={len(rows)}, "
        f"scope={args.disagreement_scope}, interventions={gated_count}, "
        f"resume={start}, fingerprint={fingerprint}"
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
    interventions = 0
    with (
        model,
        open(output_path, "a" if start else "w") as prediction_handle,
        open(audit_output, "a" if start else "w") as audit_handle,
    ):
        for row_index, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            candidate_answers = [
                _answer(predictions[key]) for predictions in prediction_sets
            ]
            majority, vote_counts, majority_tied = majority_answer(candidate_answers)
            intervene = should_intervene(candidate_answers, args.disagreement_scope)
            metadata: dict[str, Any] = {}
            raw_response: object = majority
            parsed: str | None = majority
            if intervene:
                interventions += 1
                video_path = os.path.join(video_folder, str(row["video_path"]))
                _fps, total_frames = _video_metadata(video_path)
                if total_frames <= 0:
                    raise RuntimeError(f"Could not read video metadata: {video_path}")
                candidates = sorted(set(candidate_answers))
                args.current_candidate_answers = candidates
                if args.mode == "candidate_arbiter":
                    final_indices, metadata = build_verifier_frame_indices(
                        proofpacks[key]["selected"],
                        total_frames,
                        max_frames=args.max_frames,
                        proofpack_quota=args.proofpack_quota,
                    )
                    final_frames = extract_frames_by_indices(video_path, final_indices)
                elif args.mode == "uncertainty_tcot":
                    final_indices, metadata = select_uncertainty_dynamic_tcot(
                        model, row, video_path, total_frames, args
                    )
                    final_frames = extract_frames_by_indices(video_path, final_indices)
                else:
                    final_frames, metadata = select_delta_gated_crops(
                        model,
                        row,
                        video_path,
                        proofpacks[key]["selected"],
                        detections[key],
                        args,
                    )
                    final_indices = list(metadata["base_indices"])
                raw_response = model.generate(
                    final_frames,
                    [{"role": "user", "content": build_candidate_prompt(row, candidates)}],
                    max_new_tokens=8,
                )
                parsed = parse_candidate_response(raw_response, candidates)
            chosen = parsed or majority
            prediction = build_prediction_row(
                row,
                chosen,
                prompt_variant=f"conditional_{args.mode}",
            )
            prediction.update(
                {
                    "conditional_fingerprint": fingerprint,
                    "conditional_mode": args.mode,
                    "conditional_scope": args.disagreement_scope,
                    "conditional_applied": intervene,
                    "candidate_answers": candidate_answers,
                    "candidate_vote_counts": vote_counts,
                    "majority_answer": majority,
                    "majority_tied": majority_tied,
                    "conditional_raw_response": str(raw_response),
                    "conditional_parsed": parsed,
                    "conditional_fallback_majority": intervene and parsed is None,
                }
            )
            audit = {
                "sample_key": key,
                "video_path": row.get("video_path"),
                "conditional_fingerprint": fingerprint,
                "conditional_mode": args.mode,
                "conditional_scope": args.disagreement_scope,
                "conditional_applied": intervene,
                "candidate_answers": candidate_answers,
                "candidate_vote_counts": vote_counts,
                "majority_answer": majority,
                "selected_answer": chosen,
                "metadata": metadata,
            }
            prediction_handle.write(json.dumps(prediction) + "\n")
            prediction_handle.flush()
            audit_handle.write(json.dumps(audit) + "\n")
            audit_handle.flush()
            print(
                f"  Conditional progress: {row_index + 1}/{len(rows)} "
                f"(calls={interventions})"
            )

    print(f"Interventions this invocation: {interventions}")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
