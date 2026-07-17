#!/usr/bin/env python3
"""Profile one timestamp-grounded InternVideo3 LongQA query."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path

from longqa_utils import build_longqa_prompt, normalize_answer


DEFAULT_REVISION = "c4602918b65225650d152db2850fe34e01d21fcd"


def load_sample(input_path: str, video_name: str) -> dict[str, object]:
    with open(input_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return next(row for row in rows if row["video_path"] == video_name)


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
    # Qwen3-VL's video resize budget is expressed across the complete video.
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
    metadata = {
        "processor_seconds": round(processor_seconds, 4),
        "prompt_tokens": int(inputs["attention_mask"].sum()),
        "sampled_frames": int(grid_t * processor.video_processor.temporal_patch_size),
        "video_grid_thw": [int(grid_t), int(grid_h), int(grid_w)],
        "visual_tokens": int(grid_t * grid_h * grid_w / processor.video_processor.merge_size**2),
        "timestamp_count": len(timestamps),
        "first_timestamp_seconds": timestamps[0],
        "last_timestamp_seconds": timestamps[-1],
    }
    return inputs, metadata


def profile_configuration(
    model: object,
    processor: object,
    messages: list[dict[str, object]],
    num_frames: int,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
) -> dict[str, object]:
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    inputs, result = prepare_inputs(
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
    result["transfer_seconds"] = round(time.perf_counter() - start, 4)

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    result["inference_seconds"] = round(time.perf_counter() - start, 4)
    result["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    result["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    response = processor.decode(new_tokens, skip_special_tokens=True).strip()
    result["output_tokens"] = int(new_tokens.numel())
    result["response"] = response
    result["parsed_answer"] = normalize_answer(response)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--video-name", required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--min-pixels", type=int, default=64 * 32 * 32)
    parser.add_argument("--max-pixels", type=int, default=128 * 32 * 32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--model", default="yanziang/InternVideo3-8B-Instruct")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--processor-only", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    row = load_sample(args.input, args.video_name)
    video_path = os.path.join(args.video_folder, args.video_name)
    messages = build_messages(video_path, row)

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"
    model = None
    if not args.processor_only:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            device_map="auto",
        )
    load_seconds = time.perf_counter() - load_start

    payload = {
        "model": args.model,
        "revision": args.revision,
        "attn_implementation": args.attn_implementation,
        "video_name": args.video_name,
        "question": row["question"],
        "gold_answer": row["mcq_answer"],
        "min_pixels_per_frame": args.min_pixels,
        "max_pixels_per_frame": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "model_load_seconds": round(load_seconds, 4),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "profiles": [],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.processor_only:
        print("Running a 4-frame CUDA warmup...", flush=True)
        payload["warmup"] = profile_configuration(
            model,
            processor,
            messages,
            4,
            args.min_pixels,
            args.max_pixels,
            1,
        )

    for num_frames in args.frames:
        if args.processor_only:
            _, profile = prepare_inputs(
                processor,
                messages,
                num_frames,
                args.min_pixels,
                args.max_pixels,
            )
        else:
            profile = profile_configuration(
                model,
                processor,
                messages,
                num_frames,
                args.min_pixels,
                args.max_pixels,
                args.max_new_tokens,
            )
        profile["requested_frames"] = num_frames
        payload["profiles"].append(profile)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(profile, indent=2), flush=True)


if __name__ == "__main__":
    main()
