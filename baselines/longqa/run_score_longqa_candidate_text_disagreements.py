#!/usr/bin/env python3
"""Score complete candidate answer text on fixed-ensemble disagreements."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

from longqa_utils import (
    apply_subset,
    index_row_aligned_metadata,
    parse_mcq_options,
    sample_key,
)
from run_generate_longqa_grounded import extract_frames_by_indices, load_jsonl
from run_generate_longqa_proofpack import baseline_uniform_indices
from run_generate_longqa_semantic_likelihood import build_semantic_scoring_prompt
from run_generate_longqa_uncertainty import _index_jsonl, _video_metadata
from run_generate_longqa_verifier import build_verifier_frame_indices


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _answer(row: dict[str, Any]) -> str:
    return str(
        row.get("mcq_answer_parsed") or row.get("mcq_answer", "")
    ).strip().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl",
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--primary-predictions", required=True)
    parser.add_argument("--secondary-predictions", required=True)
    parser.add_argument("--tertiary-predictions", required=True)
    parser.add_argument("--proofpack", required=True)
    parser.add_argument("--proofpack-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--proofpack-quota", type=int, default=32)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    import time

    from model import (
        VLLMModel,
        reset_prompt_token_stats,
        summarize_prompt_token_stats,
    )
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    prediction_sets = {
        "q35_pivot": _index_jsonl(_resolve_path(args.primary_predictions)),
        "q35_uniform": _index_jsonl(_resolve_path(args.secondary_predictions)),
        "q3_verifier": _index_jsonl(_resolve_path(args.tertiary_predictions)),
    }
    proofpack_reference = load_jsonl(_resolve_path(args.proofpack_reference))
    proofpacks = index_row_aligned_metadata(
        load_jsonl(_resolve_path(args.proofpack)),
        proofpack_reference,
        "candidate-text proof pack",
    )
    disagreement_rows = []
    for row in rows:
        key = sample_key(row)
        answers = {
            _answer(predictions[key]) for predictions in prediction_sets.values()
        }
        if len(answers) > 1:
            disagreement_rows.append(row)

    fingerprint_payload = {
        "primary": os.path.abspath(args.primary_predictions),
        "secondary": os.path.abspath(args.secondary_predictions),
        "tertiary": os.path.abspath(args.tertiary_predictions),
        "proofpack": os.path.abspath(args.proofpack),
        "model": args.llm_model,
        "max_frames": args.max_frames,
        "proofpack_quota": args.proofpack_quota,
        "min_pixels": os.environ.get("QWEN_MIN_PIXELS"),
        "max_pixels": os.environ.get("QWEN_MAX_PIXELS"),
        "max_model_len": os.environ.get("VLLM_QWEN_MAX_MODEL_LEN"),
        "subset": os.path.abspath(args.subset_file) if args.subset_file else None,
        "scoring": "complete_candidate_text_mean_logprob_v1",
    }
    fingerprint = hashlib.sha1(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    existing = (
        []
        if args.no_resume or not os.path.exists(output_path)
        else load_jsonl(output_path)
    )
    start = 0
    for index, record in enumerate(existing[: len(disagreement_rows)]):
        if (
            record.get("sample_key") != sample_key(disagreement_rows[index])
            or record.get("score_fingerprint") != fingerprint
        ):
            break
        start += 1
    if len(existing) != start:
        with open(output_path, "w") as handle:
            for record in existing[:start]:
                handle.write(json.dumps(record) + "\n")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(
        f"Candidate-text scoring: rows={len(disagreement_rows)}, "
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
    with model, open(output_path, "a" if start else "w") as handle:
        for index, row in enumerate(disagreement_rows[start:], start=start):
            key = sample_key(row)
            video_path = os.path.join(video_folder, str(row["video_path"]))
            _fps, total_frames = _video_metadata(video_path)
            if total_frames <= 0:
                raise RuntimeError(f"Could not read video metadata: {video_path}")
            pivot_indices = sorted(
                int(item["frame_index"])
                for item in proofpacks[key]["selected"][: args.max_frames]
            )
            uniform_indices = baseline_uniform_indices(
                total_frames, args.max_frames
            )
            mixed_indices, mixed_meta = build_verifier_frame_indices(
                proofpacks[key]["selected"],
                total_frames,
                max_frames=args.max_frames,
                proofpack_quota=args.proofpack_quota,
            )
            contexts = {
                "pivot": extract_frames_by_indices(video_path, pivot_indices),
                "uniform": extract_frames_by_indices(
                    video_path, uniform_indices
                ),
                "mixed": extract_frames_by_indices(video_path, mixed_indices),
            }
            candidate_answers = {
                name: _answer(predictions[key])
                for name, predictions in prediction_sets.items()
            }
            options = parse_mcq_options(str(row.get("mcq_options", "")))
            candidates = sorted(set(candidate_answers.values()))
            candidate_texts = {
                candidate: options[candidate] for candidate in candidates
            }
            prompt = [
                {
                    "role": "user",
                    "content": build_semantic_scoring_prompt(row),
                }
            ]
            view_scores = {
                name: model.score_candidate_texts(
                    frames, prompt, candidate_texts
                )
                for name, frames in contexts.items()
            }
            blind_scores = model.score_candidate_texts(
                [], prompt, candidate_texts
            )
            record = {
                "sample_key": key,
                "video_path": row.get("video_path"),
                "question": row.get("question"),
                "candidate_answers": candidate_answers,
                "candidate_texts": candidate_texts,
                "view_scores": view_scores,
                "blind_scores": blind_scores,
                "frame_counts": {
                    "pivot": len(pivot_indices),
                    "uniform": len(uniform_indices),
                    "mixed": len(mixed_indices),
                },
                "mixed_frame_meta": mixed_meta,
                "score_fingerprint": fingerprint,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(
                f"  Candidate-text progress: {index + 1}/"
                f"{len(disagreement_rows)}"
            )

    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())


if __name__ == "__main__":
    main()
