#!/usr/bin/env python3
"""Run a timestamp-grounded InternVideo3 pilot from native video files."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from longqa_utils import (
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    load_jsonl,
    normalize_answer,
    sample_key,
)


DEFAULT_REVISION = "c4602918b65225650d152db2850fe34e01d21fcd"


def build_messages(video_path: str, row: dict[str, object]) -> list[dict[str, object]]:
    prompt = build_longqa_prompt(
        row["question"],
        row["mcq_options"],
        prompt_variant="timestamp_grounded",
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def prepare_inputs(
    processor: object,
    messages: list[dict[str, object]],
    num_frames: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[object, dict[str, object]]:
    # InternVideo3 expresses its resize budget across the complete video.
    processor.video_processor.size = {
        "shortest_edge": min_pixels * num_frames,
        "longest_edge": max_pixels * num_frames,
    }
    processor.video_processor.max_frames = num_frames
    processor.video_processor.fps = None

    start = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        do_sample_frames=True,
        num_frames=num_frames,
    )
    processor_seconds = time.perf_counter() - start

    input_ids = inputs["input_ids"][0].tolist()
    text_ids = [token for token in input_ids if token != processor.video_token_id]
    rendered_text = processor.tokenizer.decode(text_ids, skip_special_tokens=False)
    timestamps = [float(value) for value in re.findall(r"<([0-9.]+) seconds>", rendered_text)]
    grid_t, grid_h, grid_w = inputs["video_grid_thw"][0].tolist()
    profile = {
        "processor_seconds": round(processor_seconds, 4),
        "prompt_tokens": int(inputs["attention_mask"].sum()),
        "sampled_frames": int(grid_t * processor.video_processor.temporal_patch_size),
        "video_grid_thw": [int(grid_t), int(grid_h), int(grid_w)],
        "timestamp_count": len(timestamps),
        "first_timestamp_seconds": timestamps[0],
        "last_timestamp_seconds": timestamps[-1],
    }
    return inputs, profile


def generate_one(
    model: object,
    processor: object,
    messages: list[dict[str, object]],
    num_frames: int,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
) -> tuple[str, dict[str, object]]:
    import torch

    inputs, profile = prepare_inputs(
        processor,
        messages,
        num_frames,
        min_pixels,
        max_pixels,
    )
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    inputs = inputs.to(model.device)
    torch.cuda.synchronize()
    profile["transfer_seconds"] = round(time.perf_counter() - start, 4)

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    profile["inference_seconds"] = round(time.perf_counter() - start, 4)
    profile["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    profile["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    response = processor.decode(new_tokens, skip_special_tokens=True).strip()
    profile["output_tokens"] = int(new_tokens.numel())
    return response, profile


def percentile(values: list[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 4)


def write_summary(
    predictions: list[dict[str, object]],
    output_path: Path,
    model_load_seconds: float,
    attn_implementation: str,
) -> None:
    import torch

    correct = sum(
        normalize_answer(row["mcq_answer_parsed"]) == normalize_answer(row["gold_answer"])
        for row in predictions
    )
    processor_times = [float(row["profile"]["processor_seconds"]) for row in predictions]
    inference_times = [float(row["profile"]["inference_seconds"]) for row in predictions]
    summary = {
        "samples": len(predictions),
        "correct": correct,
        "accuracy": round(correct / len(predictions), 4),
        "attention_implementation": attn_implementation,
        "gpu": torch.cuda.get_device_name(),
        "model_load_seconds": round(model_load_seconds, 4),
        "processor_seconds": {
            "mean": round(sum(processor_times) / len(processor_times), 4),
            "p50": percentile(processor_times, 50),
            "p95": percentile(processor_times, 95),
            "max": round(max(processor_times), 4),
        },
        "inference_seconds": {
            "mean": round(sum(inference_times) / len(inference_times), 4),
            "p50": percentile(inference_times, 50),
            "p95": percentile(inference_times, 95),
            "max": round(max(inference_times), 4),
        },
    }
    summary_path = output_path.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="yanziang/InternVideo3-8B-Instruct")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=131072)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    rows = apply_subset(load_jsonl(args.input), args.subset_file)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = load_jsonl(str(output_path)) if output_path.exists() else []
    completed = {sample_key(row) for row in predictions}

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="auto",
    )
    model_load_seconds = time.perf_counter() - load_start

    with output_path.open("a") as output_file:
        for index, row in enumerate(rows, start=1):
            if sample_key(row) in completed:
                continue
            video_path = os.path.join(args.video_folder, str(row["video_path"]))
            response, profile = generate_one(
                model,
                processor,
                build_messages(video_path, row),
                args.frames,
                args.min_pixels,
                args.max_pixels,
                args.max_new_tokens,
            )
            prediction = build_prediction_row(row, response, "timestamp_grounded")
            prediction["gold_answer"] = row["mcq_answer"]
            prediction["attention_implementation"] = args.attn_implementation
            prediction["profile"] = profile
            output_file.write(json.dumps(prediction) + "\n")
            output_file.flush()
            predictions.append(prediction)
            correct = normalize_answer(prediction["mcq_answer_parsed"]) == normalize_answer(
                prediction["gold_answer"]
            )
            print(
                f"{index}/{len(rows)} {row['video_path']} "
                f"answer={prediction['mcq_answer_parsed']} correct={correct} "
                f"processor={profile['processor_seconds']:.2f}s "
                f"inference={profile['inference_seconds']:.2f}s",
                flush=True,
            )

    write_summary(predictions, output_path, model_load_seconds, args.attn_implementation)


if __name__ == "__main__":
    main()
