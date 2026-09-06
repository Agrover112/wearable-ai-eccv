#!/usr/bin/env python3
"""Run NVILA-HD-Video with AutoGaze on an EgoLongQA subset."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from longqa_utils import (
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    load_jsonl,
    sample_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model", default="nvidia/NVILA-8B-HD-Video")
    parser.add_argument("--autogaze-model", default="nvidia/AutoGaze")
    parser.add_argument("--num-video-frames", type=int, default=128)
    parser.add_argument("--num-thumbnail-frames", type=int, default=64)
    parser.add_argument("--max-tiles-video", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModel, AutoProcessor

    rows = apply_subset(load_jsonl(args.input), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [] if args.no_resume or not output_path.exists() else load_jsonl(str(output_path))
    resume_count = 0
    for index, prediction in enumerate(existing[: len(rows)]):
        if sample_key(prediction) != sample_key(rows[index]):
            break
        if str(prediction.get("mcq_answer_parsed", "")) not in {"A", "B", "C", "D"}:
            break
        resume_count += 1
    if len(existing) != resume_count:
        with output_path.open("w") as handle:
            for prediction in existing[:resume_count]:
                handle.write(json.dumps(prediction) + "\n")

    processor = AutoProcessor.from_pretrained(
        args.model,
        autogaze_model_id=args.autogaze_model,
        num_video_frames=args.num_video_frames,
        num_video_frames_thumbnail=args.num_thumbnail_frames,
        max_tiles_video=args.max_tiles_video,
        gazing_ratio_tile=[0.2] + [0.06] * 15,
        gazing_ratio_thumbnail=1,
        task_loss_requirement_tile=0.6,
        task_loss_requirement_thumbnail=None,
        max_batch_size_autogaze=16,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        max_batch_size_siglip=32,
    ).to("cuda").eval()
    video_token = processor.tokenizer.video_token

    mode = "a" if resume_count else "w"
    with output_path.open(mode) as handle, torch.inference_mode():
        for index, row in enumerate(rows[resume_count:], start=resume_count):
            video_path = os.path.join(args.video_folder, str(row["video_path"]))
            if not os.path.isfile(video_path):
                raise RuntimeError(f"Missing video: {video_path}")
            prompt = build_longqa_prompt(row["question"], row["mcq_options"], "baseline")
            started = time.perf_counter()
            inputs = processor(
                text=f"{video_token}\n\n{prompt}",
                videos=video_path,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(model.device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
            response = processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )[0].strip()
            seconds = time.perf_counter() - started

            prediction = build_prediction_row(row, response, prompt_variant="baseline")
            prediction.update(
                {
                    "model_id": args.model,
                    "autogaze_model_id": args.autogaze_model,
                    "media_mode": "native_video_autogaze",
                    "num_video_frames": args.num_video_frames,
                    "num_thumbnail_frames": args.num_thumbnail_frames,
                    "max_tiles_video": args.max_tiles_video,
                    "generation_seconds": round(seconds, 3),
                }
            )
            handle.write(json.dumps(prediction) + "\n")
            handle.flush()
            print(
                f"Progress: {index + 1}/{len(rows)} answer="
                f"{prediction['mcq_answer_parsed']} seconds={seconds:.2f}"
            )

    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
