#!/usr/bin/env python3
"""Suppress answers supported only on the wrong side of a temporal pivot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from longqa_utils import (
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    compute_diagnostics,
    load_jsonl,
    normalize_answer,
    sample_key,
)
from run_generate_longqa_grounded import extract_frames_by_indices
from run_generate_longqa_proofpack import compile_temporal_program_v2
from run_generate_longqa_uncertainty import _video_metadata


LETTERS = "ABCD"
LAMBDAS = (0.25, 0.5, 1.0)


def _index(paths: list[str], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            key = str(row.get("sample_key") or sample_key(row))
            if key in indexed:
                raise ValueError(f"Duplicate {label} key: {key}")
            indexed[key] = row
    return indexed


def _answer(row: dict[str, Any]) -> str:
    value = normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))
    if value not in LETTERS:
        raise ValueError(f"Invalid answer {value!r} for {sample_key(row)}")
    return value


def _evenly(values: list[int], count: int) -> list[int]:
    values = sorted(set(values))
    if len(values) <= count:
        return values
    if count <= 1:
        return [values[len(values) // 2]]
    return [
        values[int(round(position * (len(values) - 1) / (count - 1)))]
        for position in range(count)
    ]


def _endpoint_candidates(total_frames: int, count: int) -> list[int]:
    count = min(max(1, count), total_frames)
    if count == 1:
        return [0]
    return sorted(
        {
            int(round(position * (total_frames - 1) / (count - 1)))
            for position in range(count)
        }
    )


def temporal_side_packs(
    total_frames: int,
    pivots: list[int],
    direction: str,
    candidate_frames: int,
    max_frames: int,
    pivot_context: int,
) -> tuple[list[int], list[int]]:
    """Return requested-side and opposite-side packs around the nearest pivot."""
    candidates = _endpoint_candidates(total_frames, candidate_frames)
    pivots = sorted({min(max(0, int(value)), total_frames - 1) for value in pivots})
    if direction not in {"forward", "backward"} or not pivots:
        return [], []
    valid: list[int] = []
    wrong: list[int] = []
    for frame_index in candidates:
        pivot = min(pivots, key=lambda value: (abs(frame_index - value), value))
        requested = frame_index > pivot if direction == "forward" else frame_index < pivot
        (valid if requested else wrong).append(frame_index)

    shared = sorted(
        candidates,
        key=lambda frame_index: min(abs(frame_index - pivot) for pivot in pivots),
    )[:pivot_context]
    def pack(side: list[int]) -> list[int]:
        side_only = [frame_index for frame_index in side if frame_index not in shared]
        side_budget = max(1, max_frames - len(shared))
        selected = _evenly(side_only, side_budget) + shared
        return sorted(set(selected))[:max_frames]

    return pack(valid), pack(wrong)


def _normalized_logprobs(scores: dict[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    denominator = maximum + math.log(
        sum(math.exp(float(scores[letter]) - maximum) for letter in LETTERS)
    )
    return {letter: float(scores[letter]) - denominator for letter in LETTERS}


def _argmax(scores: dict[str, float], fallback: str) -> str:
    maximum = max(scores.values())
    tied = [letter for letter in LETTERS if abs(scores[letter] - maximum) < 1e-12]
    return fallback if fallback in tied else tied[0]


def _fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "baseline": os.path.abspath(args.baseline_predictions),
        "proofpacks": sorted(os.path.abspath(path) for path in args.proofpack),
        "subset": os.path.abspath(args.subset_file) if args.subset_file else None,
        "model": args.llm_model,
        "candidate_frames": args.candidate_frames,
        "max_frames": args.max_frames,
        "pivot_context": args.pivot_context,
        "schema": 1,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--proofpack", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--candidate-frames", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--pivot-context", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    rows = apply_subset(load_jsonl(args.input), args.subset_file)
    baseline = _index([args.baseline_predictions], "baseline")
    proofpacks = _index(args.proofpack, "proofpack")
    required = {sample_key(row) for row in rows}
    for label, indexed in (("baseline", baseline), ("proofpack", proofpacks)):
        missing = required - set(indexed)
        if missing:
            raise RuntimeError(f"{label} is missing {len(missing)} rows")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.jsonl"
    fingerprint = _fingerprint(args)
    existing = [] if args.no_resume or not feature_path.exists() else load_jsonl(str(feature_path))
    start = 0
    for index, record in enumerate(existing[: len(rows)]):
        if record.get("sample_key") != sample_key(rows[index]) or record.get("fingerprint") != fingerprint:
            break
        start += 1
    if len(existing) != start:
        with feature_path.open("w") as handle:
            for record in existing[:start]:
                handle.write(json.dumps(record) + "\n")

    model = VLLMModel(
        args.llm_model,
        tp_size=1,
        concurrency=args.concurrency,
        max_frames=args.max_frames,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    begun = time.time()
    with model, feature_path.open("a" if start else "w") as handle:
        for index, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            fallback = _answer(baseline[key])
            proof = proofpacks[key]
            metadata = proof.get("selection_meta", {})
            program_data = metadata.get("temporal_program", {})
            program = compile_temporal_program_v2(row.get("question"))
            direction = str(program_data.get("direction") or program.direction)
            pivots = [int(value) for value in metadata.get("pivot_centers", [])]
            video_path = os.path.join(args.video_folder, str(row["video_path"]))
            _, total_frames = _video_metadata(video_path)
            valid_indices, wrong_indices = temporal_side_packs(
                total_frames,
                pivots,
                direction,
                args.candidate_frames,
                args.max_frames,
                args.pivot_context,
            )
            record: dict[str, Any] = {
                "sample_key": key,
                "video_path": row.get("video_path"),
                "question": row.get("question"),
                "baseline_answer": fallback,
                "fingerprint": fingerprint,
                "operator": str(program_data.get("operator") or program.operator),
                "direction": direction,
                "pivot_centers": pivots,
                "contrast_applied": bool(valid_indices and wrong_indices),
                "valid_indices": valid_indices,
                "wrong_indices": wrong_indices,
            }
            if valid_indices and wrong_indices:
                prompt = build_longqa_prompt(row["question"], row["mcq_options"], "baseline")
                valid_scores = _normalized_logprobs(
                    model.score_choice_letters(
                        extract_frames_by_indices(video_path, valid_indices),
                        [{"role": "user", "content": prompt}],
                    )
                )
                wrong_scores = _normalized_logprobs(
                    model.score_choice_letters(
                        extract_frames_by_indices(video_path, wrong_indices),
                        [{"role": "user", "content": prompt}],
                    )
                )
                record["valid_logprobs"] = valid_scores
                record["wrong_logprobs"] = wrong_scores
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"  Temporal contrast: {index + 1}/{len(rows)} applied={record['contrast_applied']}")

    features = load_jsonl(str(feature_path))
    policies = {"baseline": [] , "valid_only": []}
    for value in LAMBDAS:
        policies[f"contrast_{value:g}"] = []
    for row, feature in zip(rows, features):
        fallback = str(feature["baseline_answer"])
        policies["baseline"].append(build_prediction_row(row, fallback, "temporal_contrast_baseline"))
        if not feature.get("contrast_applied"):
            choices = {name: fallback for name in policies if name != "baseline"}
        else:
            valid = feature["valid_logprobs"]
            wrong = feature["wrong_logprobs"]
            choices = {"valid_only": _argmax(valid, fallback)}
            for value in LAMBDAS:
                scores = {
                    letter: float(valid[letter]) - value * float(wrong[letter])
                    for letter in LETTERS
                }
                choices[f"contrast_{value:g}"] = _argmax(scores, fallback)
        for name, choice in choices.items():
            policies[name].append(build_prediction_row(row, choice, f"temporal_{name}"))

    summary: dict[str, Any] = {
        "rows": len(rows),
        "applied": sum(bool(row.get("contrast_applied")) for row in features),
        "uses_labels_for_predictions": False,
        "policies": {},
    }
    for name, predictions in policies.items():
        diagnostics = compute_diagnostics(rows, predictions, run_id=name)
        summary["policies"][name] = diagnostics
        with (output_dir / f"predictions_{name}.jsonl").open("w") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction) + "\n")
        print(f"{name}: {diagnostics['correct']}/{diagnostics['total']} ({diagnostics['accuracy']:.4f})")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())


if __name__ == "__main__":
    main()
