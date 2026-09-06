#!/usr/bin/env python3
"""Compare raw and deduplicated LongQA evidence packs with a fixed answerer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from longqa_utils import (
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    compute_diagnostics,
    index_row_aligned_metadata,
    load_jsonl,
    sample_key,
)
from run_generate_longqa_grounded import extract_frames_by_indices
from run_generate_longqa_object_hints import _selected_priority


MODES = ("raw48", "temporal_dedup48", "visual_dedup48")
SCHEMA_VERSION = 1


def resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def perceptual_bits(image: Any) -> np.ndarray:
    gray = image.convert("L").resize((9, 8))
    values = np.asarray(gray, dtype=np.int16)
    return (values[:, 1:] > values[:, :-1]).reshape(-1)


def perceptual_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return 1.0 - float(np.count_nonzero(left != right)) / float(left.size)


def ranked_unique(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(selected, key=_selected_priority)
    output = []
    seen = set()
    for item in ranked:
        frame_index = int(item["frame_index"])
        if frame_index in seen:
            continue
        seen.add(frame_index)
        output.append(item)
    return output


def choose_temporal(
    selected: list[dict[str, Any]], max_frames: int, gap_seconds: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = []
    rejected = []
    for item in selected:
        timestamp = float(item["timestamp"])
        if any(abs(timestamp - float(other["timestamp"])) < gap_seconds for other in kept):
            rejected.append(item)
            continue
        kept.append(item)
        if len(kept) >= max_frames:
            break
    return kept, rejected


def choose_visual(
    selected: list[dict[str, Any]],
    images: dict[int, Any],
    max_frames: int,
    window_seconds: float,
    similarity_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = []
    rejected = []
    signatures: dict[int, np.ndarray] = {}
    for item in selected:
        frame_index = int(item["frame_index"])
        image = images.get(frame_index)
        if image is None:
            continue
        signature = signatures.setdefault(frame_index, perceptual_bits(image))
        duplicate = False
        for other in kept:
            if abs(float(item["timestamp"]) - float(other["timestamp"])) > window_seconds:
                continue
            other_index = int(other["frame_index"])
            other_signature = signatures.setdefault(
                other_index, perceptual_bits(images[other_index])
            )
            if perceptual_similarity(signature, other_signature) >= similarity_threshold:
                duplicate = True
                break
        if duplicate:
            rejected.append(item)
            continue
        kept.append(item)
        if len(kept) >= max_frames:
            break
    return kept, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl")
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--proofpack", required=True)
    parser.add_argument("--proofpack-reference", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--temporal-gap-seconds", type=float, default=8.0)
    parser.add_argument("--visual-window-seconds", type=float, default=15.0)
    parser.add_argument("--visual-similarity", type=float, default=0.90)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def fingerprint(args: argparse.Namespace) -> str:
    ignored = {"output", "eval_output", "no_resume"}
    payload = {key: value for key, value in vars(args).items() if key not in ignored}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = resolve(args.input)
    video_folder = resolve(args.video_folder)
    output_path = resolve(args.output)
    eval_path = resolve(args.eval_output)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    packs = index_row_aligned_metadata(
        load_jsonl(resolve(args.proofpack)),
        load_jsonl(resolve(args.proofpack_reference)),
        "density-ablation proof pack",
    )
    required = {sample_key(row) for row in rows}
    missing = required - set(packs)
    if missing:
        raise RuntimeError(f"proof pack is missing {len(missing)} density-ablation rows")

    run_fingerprint = fingerprint(args)
    existing = [] if args.no_resume or not os.path.exists(output_path) else load_jsonl(output_path)
    start = 0
    for index, prediction in enumerate(existing[: len(rows)]):
        if (
            sample_key(prediction) != sample_key(rows[index])
            or prediction.get("density_fingerprint") != run_fingerprint
        ):
            break
        start += 1
    write_jsonl(output_path, existing[:start])

    model = VLLMModel(
        args.llm_model,
        tp_size=1,
        concurrency=1,
        max_frames=args.max_frames,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    begun = time.time()
    with model, open(output_path, "a") as handle:
        for index, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            video_path = os.path.join(video_folder, str(row["video_path"]))
            ranked = ranked_unique(packs[key]["selected"])
            all_indices = [int(item["frame_index"]) for item in ranked]
            all_images = extract_frames_by_indices(video_path, all_indices)
            image_map = dict(zip(all_indices, all_images))
            if args.mode == "raw48":
                kept, rejected = ranked[: args.max_frames], []
            elif args.mode == "temporal_dedup48":
                kept, rejected = choose_temporal(
                    ranked, args.max_frames, args.temporal_gap_seconds
                )
            else:
                kept, rejected = choose_visual(
                    ranked,
                    image_map,
                    args.max_frames,
                    args.visual_window_seconds,
                    args.visual_similarity,
                )
            kept = sorted(kept, key=lambda item: float(item["timestamp"]))
            frames = [image_map[int(item["frame_index"])] for item in kept]
            response = model.generate(
                frames,
                [{"role": "user", "content": build_longqa_prompt(row["question"], row["mcq_options"])}],
                max_new_tokens=16,
            )
            prediction = build_prediction_row(
                row, response, prompt_variant=f"density_{args.mode}"
            )
            prediction.update(
                {
                    "density_schema": SCHEMA_VERSION,
                    "density_mode": args.mode,
                    "density_fingerprint": run_fingerprint,
                    "density_input_frames": len(ranked),
                    "density_final_frames": len(kept),
                    "density_rejected_frames": len(rejected),
                    "density_frame_indices": [int(item["frame_index"]) for item in kept],
                }
            )
            handle.write(json.dumps(prediction) + "\n")
            handle.flush()
            print(
                f"  Density progress: {index + 1}/{len(rows)} "
                f"frames={len(kept)} rejected={len(rejected)}"
            )

    predictions = load_jsonl(output_path)
    result = compute_diagnostics(rows, predictions, run_id=f"density_{args.mode}")
    Path(eval_path).write_text(json.dumps(result, indent=2) + "\n")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    print(f"accuracy={result['accuracy_raw']:.4f} correct={result['correct']}/{result['total']}")


if __name__ == "__main__":
    main()
