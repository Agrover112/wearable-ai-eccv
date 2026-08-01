#!/usr/bin/env python3
"""Run a matched LongQA ablation using only the central video frame."""

from __future__ import annotations

import argparse
import json
import os

from longqa_utils import apply_subset, build_longqa_prompt, build_prediction_row, query_hash, sample_key
from run_generate_longqa import _print_context_summary
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices, load_jsonl


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(os.path.dirname(__file__), path)


def central_index(video_path: str) -> int:
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError(f"Invalid video length: {video_path}")
        return total // 2
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl")
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-resume-predictions", action="store_true")
    args = parser.parse_args()

    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats

    input_path = resolve(args.input)
    video_folder = resolve(args.video_folder)
    output_path = resolve(args.output)
    eval_output = resolve(args.eval_output)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    fingerprint = query_hash(json.dumps({"model": args.llm_model, "ablation": "central_frame"}, sort_keys=True))
    cached = [] if args.no_resume_predictions or not os.path.exists(output_path) else load_jsonl(output_path)
    valid = []
    for index, prediction in enumerate(cached[: len(rows)]):
        if sample_key(prediction) != sample_key(rows[index]):
            break
        if prediction.get("single_frame_fingerprint") != fingerprint:
            break
        valid.append(prediction)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if len(valid) != len(cached):
        with open(output_path, "w") as handle:
            for row in valid:
                handle.write(json.dumps(row) + "\n")
    if len(valid) == len(rows):
        _run_eval(input_path, output_path, eval_output)
        return

    model = VLLMModel(args.llm_model, tp_size=1, concurrency=args.concurrency, max_frames=1, model_type="qwen")
    reset_prompt_token_stats()
    with model, open(output_path, "a" if valid else "w") as handle:
        for index, row in enumerate(rows[len(valid):], start=len(valid)):
            video_path = os.path.join(video_folder, str(row["video_path"]))
            frame_index = central_index(video_path)
            frames = extract_frames_by_indices(video_path, [frame_index])
            response = model.generate(
                frames,
                [{"role": "user", "content": build_longqa_prompt(row["question"], row["mcq_options"], prompt_variant="baseline")}],
                max_new_tokens=16,
            )
            prediction = build_prediction_row(row, response, "baseline")
            prediction.update({"ablation": "central_frame", "frame_index": frame_index, "frames": 1, "single_frame_fingerprint": fingerprint})
            handle.write(json.dumps(prediction) + "\n")
            handle.flush()
            print(f"  Central-frame progress: {index + 1}/{len(rows)}")
    _print_context_summary(summarize_prompt_token_stats())
    _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
