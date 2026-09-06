#!/usr/bin/env python3
"""Answer EgoLongQA using frame indices recorded by a selection experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from longqa_utils import apply_subset, build_longqa_prompt, build_prediction_row, load_jsonl, sample_key
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices


def _nested(record: dict[str, Any], key: str) -> object:
    value: object = record
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _indices(record: dict[str, Any], key: str) -> list[int]:
    values = _nested(record, key)
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("frame_index")
        if value is not None:
            result.append(int(value))
    return result


def build_vertical_detail_panel(image: object, size: int = 672) -> object:
    """Pair a full view with enlarged upper/lower halves from the same frame."""
    from PIL import Image, ImageOps

    image = image.convert("RGB")
    width, height = image.size
    upper = image.crop((0, 0, width, max(1, height // 2)))
    lower = image.crop((0, height // 2, width, height))
    half = size // 2
    panel = Image.new("RGB", (size, size), (0, 0, 0))
    panel.paste(ImageOps.fit(image, (half, size)), (0, 0))
    panel.paste(ImageOps.fit(upper, (size - half, half)), (half, 0))
    panel.paste(ImageOps.fit(lower, (size - half, size - half)), (half, half))
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--indices-key", default="final_indices")
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--prompt-variant", default="baseline")
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--detail-panel-centers-key",
        default=None,
        help="Optional nested selection key whose frame indices receive full/detail panels.",
    )
    parser.add_argument("--detail-panel-size", type=int, default=672)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    rows = apply_subset(load_jsonl(args.input), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    selections = {
        str(row.get("sample_key") or sample_key(row)): row
        for row in load_jsonl(args.selection)
    }
    missing = {sample_key(row) for row in rows} - set(selections)
    if missing:
        raise RuntimeError(f"Selection file is missing {len(missing)} rows")
    fingerprint = hashlib.sha1(
        json.dumps(
            {
                "selection": os.path.abspath(args.selection),
                "indices_key": args.indices_key,
                "model": args.llm_model,
                "prompt": args.prompt_variant,
                "detail_panel_centers_key": args.detail_panel_centers_key,
                "detail_panel_size": args.detail_panel_size,
                "schema": 1,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = [] if args.no_resume or not output.exists() else load_jsonl(str(output))
    start = 0
    for index, prediction in enumerate(existing[: len(rows)]):
        if sample_key(prediction) != sample_key(rows[index]) or prediction.get("selection_answer_fingerprint") != fingerprint:
            break
        start += 1
    if len(existing) != start:
        with output.open("w") as handle:
            for prediction in existing[:start]:
                handle.write(json.dumps(prediction) + "\n")

    model = VLLMModel(
        args.llm_model,
        tp_size=1,
        concurrency=args.concurrency,
        max_frames=args.max_frames,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    begun = time.time()
    with model, output.open("a" if start else "w") as handle:
        for index, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            indices = _indices(selections[key], args.indices_key)
            if not indices or len(indices) > args.max_frames or len(indices) != len(set(indices)):
                raise RuntimeError(f"Invalid selected frame pack for {key}: {len(indices)} frames")
            video_path = os.path.join(args.video_folder, str(row["video_path"]))
            frames = extract_frames_by_indices(video_path, indices)
            detail_indices = set(
                _indices(selections[key], args.detail_panel_centers_key)
                if args.detail_panel_centers_key
                else []
            )
            if detail_indices:
                frames = [
                    build_vertical_detail_panel(frame, args.detail_panel_size)
                    if frame_index in detail_indices
                    else frame
                    for frame_index, frame in zip(indices, frames)
                ]
            prompt = build_longqa_prompt(row["question"], row["mcq_options"], args.prompt_variant)
            response = model.generate(
                frames, [{"role": "user", "content": prompt}], max_new_tokens=16
            )
            prediction = build_prediction_row(row, response, f"selection_{args.prompt_variant}")
            prediction.update(
                {
                    "selection_answer_fingerprint": fingerprint,
                    "selection_source": os.path.abspath(args.selection),
                    "selection_indices_key": args.indices_key,
                    "selection_frame_indices": indices,
                    "selection_detail_panel_indices": sorted(detail_indices),
                }
            )
            handle.write(json.dumps(prediction) + "\n")
            handle.flush()
            print(f"  Selection answer: {index + 1}/{len(rows)}")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    _run_eval(args.input, str(output), args.eval_output)


if __name__ == "__main__":
    main()
