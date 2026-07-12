#!/usr/bin/env python3
"""Verify disagreements between two completed LongQA candidate runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

from longqa_utils import apply_subset, build_longqa_prompt, build_prediction_row, sample_key
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices, load_jsonl
from run_generate_longqa_proofpack import baseline_uniform_indices


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


def build_verifier_prompt(row: dict[str, Any], first: str, second: str) -> str:
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


def _fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "primary": os.path.abspath(args.primary_predictions),
        "secondary": os.path.abspath(args.secondary_predictions),
        "proofpack": os.path.abspath(args.primary_proofpack),
        "model": args.llm_model,
        "max_frames": args.max_frames,
        "proofpack_quota": args.proofpack_quota,
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
    primary = _load_by_key(primary_path)
    secondary = _load_by_key(secondary_path)
    proofpack_rows = load_jsonl(proofpack_path)
    proofpack = {str(row.get("video_path", "")): row for row in proofpack_rows}
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
        str(row["video_path"])
        for row in disagreements
        if str(row["video_path"]) not in proofpack
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
    with model, open(output_path, mode) as handle:
        for idx, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            first = _answer(primary[key])
            second = _answer(secondary[key])
            if first == second:
                pred = build_prediction_row(row, first, prompt_variant="verifier_agreement")
                pred["verifier_applied"] = False
                pred["verifier_frames"] = 0
            else:
                video_path = os.path.join(video_folder, str(row["video_path"]))
                _fps, total_frames = _video_metadata(video_path)
                frame_indices, frame_meta = build_verifier_frame_indices(
                    proofpack[str(row["video_path"])]["selected"],
                    total_frames,
                    max_frames=args.max_frames,
                    proofpack_quota=args.proofpack_quota,
                )
                frames = extract_frames_by_indices(video_path, frame_indices)
                response = model.generate(
                    frames,
                    [{"role": "user", "content": build_verifier_prompt(row, first, second)}],
                    max_new_tokens=16,
                )
                pred = build_prediction_row(row, response, prompt_variant="disagreement_verifier")
                pred["verifier_applied"] = True
                pred["verifier_frames"] = len(frames)
                pred["verifier_frame_meta"] = frame_meta
                verified += 1
            pred["candidate_answers"] = [first, second]
            pred["verifier_fingerprint"] = fingerprint
            handle.write(json.dumps(pred) + "\n")
            handle.flush()
            print(f"  Verifier progress: {idx + 1}/{len(rows)}")

    print(f"Verifier calls this invocation: {verified}")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
