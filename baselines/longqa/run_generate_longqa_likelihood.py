#!/usr/bin/env python3
"""Score LongQA options with video-conditioned minus video-blind log probabilities."""

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
    load_subset_keys,
    sample_key,
)
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices, load_jsonl


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def build_scoring_prompt(row: dict[str, Any], video_conditioned: bool) -> str:
    evidence = (
        "Use the supplied chronological video frames as the decisive evidence."
        if video_conditioned
        else "No video evidence is supplied. Choose using only the question and answer text."
    )
    return (
        f"{evidence} Select the best multiple-choice answer.\n\n"
        f"Question: {row['question']}\n\nOptions:\n{row['mcq_options']}\n\n"
        "Return ONLY one option letter (A, B, C, or D).\nAnswer:"
    )


def calibrated_choice(
    visual_scores: dict[str, float], blind_scores: dict[str, float], blind_weight: float
) -> tuple[str, dict[str, float]]:
    calibrated = {
        letter: float(visual_scores[letter] - blind_weight * blind_scores[letter])
        for letter in "ABCD"
    }
    return max(calibrated, key=calibrated.__getitem__), calibrated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl")
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument(
        "--exclude-subset-file",
        default=None,
        help="Exclude stable sample keys listed in this subset manifest.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--proofpack", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--blind-weight", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--llm-model", default="Qwen/Qwen3-VL-8B-Instruct")
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
    proofpack_path = _resolve_path(args.proofpack)
    output_path = _resolve_path(args.output)
    eval_output = _resolve_path(args.eval_output) if args.eval_output else None
    all_rows = load_jsonl(input_path)
    included_rows = apply_subset(all_rows, args.subset_file)
    rows = list(included_rows)
    excluded_keys = load_subset_keys(args.exclude_subset_file)
    if excluded_keys is not None:
        rows = [row for row in rows if sample_key(row) not in excluded_keys]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    pack_rows = load_jsonl(proofpack_path)
    if len(pack_rows) == len(all_rows):
        pack_reference = all_rows
    elif len(pack_rows) == len(included_rows):
        pack_reference = included_rows
    else:
        raise RuntimeError(
            f"Proof pack has {len(pack_rows)} rows; expected either "
            f"{len(all_rows)} full rows or {len(included_rows)} included rows"
        )
    packs = index_row_aligned_metadata(pack_rows, pack_reference, "proof pack")
    missing = [sample_key(row) for row in rows if sample_key(row) not in packs]
    if missing:
        raise RuntimeError(f"Proof pack is missing {len(missing)} required rows")

    payload = {
        "proofpack": os.path.abspath(proofpack_path),
        "blind_weight": args.blind_weight,
        "model": args.llm_model,
        "max_frames": args.max_frames,
        "subset_file": os.path.abspath(args.subset_file) if args.subset_file else None,
        "exclude_subset_file": (
            os.path.abspath(args.exclude_subset_file) if args.exclude_subset_file else None
        ),
    }
    fingerprint = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    existing = [] if args.no_resume or not os.path.exists(output_path) else load_jsonl(output_path)
    start = 0
    for index, pred in enumerate(existing[: len(rows)]):
        if sample_key(pred) != sample_key(rows[index]):
            break
        if pred.get("likelihood_fingerprint") != fingerprint:
            break
        start += 1
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(
        f"Likelihood config: rows={len(rows)}, blind_weight={args.blind_weight}, "
        f"resume={start}, fingerprint={fingerprint}"
    )
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
            pack = packs[sample_key(row)]
            indices = [int(item["frame_index"]) for item in pack["selected"]][: args.max_frames]
            frames = extract_frames_by_indices(
                os.path.join(video_folder, str(row["video_path"])), indices
            )
            visual_scores = model.score_choice_letters(
                frames,
                [{"role": "user", "content": build_scoring_prompt(row, True)}],
            )
            blind_scores = model.score_choice_letters(
                [],
                [{"role": "user", "content": build_scoring_prompt(row, False)}],
            )
            answer, scores = calibrated_choice(visual_scores, blind_scores, args.blind_weight)
            pred = build_prediction_row(row, answer, prompt_variant="likelihood_blind_corrected")
            pred.update(
                {
                    "visual_option_logprobs": visual_scores,
                    "blind_option_logprobs": blind_scores,
                    "calibrated_option_scores": scores,
                    "blind_weight": args.blind_weight,
                    "likelihood_frames": len(frames),
                    "likelihood_fingerprint": fingerprint,
                }
            )
            handle.write(json.dumps(pred) + "\n")
            handle.flush()
            print(f"  Likelihood progress: {index + 1}/{len(rows)}")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
