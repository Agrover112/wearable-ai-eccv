#!/usr/bin/env python3
"""Q-Gate-lite LongQA run using generated timestamped narrative anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Any

import numpy as np

from longqa_utils import (
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    classify_question_types,
    sample_key,
)
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices, load_jsonl
from run_generate_longqa_openqa import MiniLMTextEncoder
from run_generate_longqa_proofpack import baseline_uniform_indices


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _video_metadata(video_path: str) -> tuple[float, int]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        return float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def build_caption_prompt(row: dict[str, Any], timestamps: list[float]) -> str:
    labels = ", ".join(
        f"image {index + 1}={timestamp:.1f}s" for index, timestamp in enumerate(timestamps)
    )
    return (
        "Caption each supplied image independently in chronological order. Focus on "
        "visible actions, handled objects, people, places, signs, and state changes "
        "that could help answer the question. Do not answer the question. Return "
        "exactly one short line per image as `N: caption`.\n\n"
        f"Image timestamps: {labels}\n\nQuestion: {row['question']}\n\n"
        f"Options:\n{row['mcq_options']}"
    )


def parse_numbered_captions(raw: object, count: int) -> list[str]:
    parsed: dict[int, str] = {}
    fallback: list[str] = []
    for line in str(raw).splitlines():
        clean = line.strip().lstrip("-* ")
        if not clean:
            continue
        match = re.match(r"(?:image\s*)?(\d+)\s*[:.)-]\s*(.+)", clean, re.IGNORECASE)
        if match:
            number = int(match.group(1)) - 1
            if 0 <= number < count:
                parsed[number] = match.group(2).strip()
        else:
            fallback.append(clean)
    captions: list[str] = []
    fallback_index = 0
    for index in range(count):
        if index in parsed:
            captions.append(parsed[index])
        elif fallback_index < len(fallback):
            captions.append(fallback[fallback_index])
            fallback_index += 1
        else:
            captions.append("No reliable caption generated.")
    return captions


def select_narrative_indices(
    query: str,
    captions: list[str],
    anchor_indices: list[int],
    encoder: MiniLMTextEncoder,
    count: int,
) -> tuple[list[int], list[float]]:
    embeddings = encoder.encode([query] + captions)
    scores = embeddings[1:] @ embeddings[0]
    order = np.argsort(-scores)[: min(count, len(anchor_indices))]
    return [anchor_indices[int(index)] for index in order], [float(scores[index]) for index in order]


def build_final_indices(
    proofpack: dict[str, Any],
    narrative_indices: list[int],
    total_frames: int,
    max_frames: int,
    visual_quota: int,
) -> list[int]:
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
    ranked = sorted(
        proofpack["selected"],
        key=lambda item: (
            priority.get(str(item.get("source", "")), 6),
            -(float(item["score"]) if item.get("score") is not None else -1e9),
        ),
    )
    pools = [
        [int(item["frame_index"]) for item in ranked[:visual_quota]],
        narrative_indices,
        baseline_uniform_indices(total_frames, max_frames),
        [int(item["frame_index"]) for item in ranked[visual_quota:]],
    ]
    chosen: list[int] = []
    seen: set[int] = set()
    for pool in pools:
        for index in pool:
            if index not in seen:
                chosen.append(index)
                seen.add(index)
            if len(chosen) == max_frames:
                return sorted(chosen)
    return sorted(chosen)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl")
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--proofpack", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--narrative-output", required=True)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--caption-frames", type=int, default=16)
    parser.add_argument("--narrative-frames", type=int, default=12)
    parser.add_argument("--non-temporal-narrative-frames", type=int, default=8)
    parser.add_argument("--visual-quota", type=int, default=44)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--text-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    import time

    from model import create_model, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    proofpack_path = _resolve_path(args.proofpack)
    output_path = _resolve_path(args.output)
    narrative_path = _resolve_path(args.narrative_output)
    eval_output = _resolve_path(args.eval_output) if args.eval_output else None
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    packs = {str(item.get("video_path", "")): item for item in load_jsonl(proofpack_path)}
    payload = {
        "proofpack": os.path.abspath(proofpack_path),
        "caption_frames": args.caption_frames,
        "narrative_frames": args.narrative_frames,
        "non_temporal_narrative_frames": args.non_temporal_narrative_frames,
        "visual_quota": args.visual_quota,
        "max_frames": args.max_frames,
        "text_encoder": args.text_encoder,
        "model": args.llm_model,
    }
    fingerprint = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    existing = [] if args.no_resume or not os.path.exists(output_path) else load_jsonl(output_path)
    narrative_existing = (
        [] if args.no_resume or not os.path.exists(narrative_path) else load_jsonl(narrative_path)
    )
    start = 0
    for index, pred in enumerate(existing[: len(rows)]):
        if sample_key(pred) != sample_key(rows[index]) or pred.get("narrative_fingerprint") != fingerprint:
            break
        start += 1
    if len(narrative_existing) < start:
        raise RuntimeError("Narrative cache is shorter than the prediction resume prefix")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(narrative_path) or ".", exist_ok=True)
    encoder = MiniLMTextEncoder(args.text_encoder, device="cpu", batch_size=64)
    model = create_model(
        "qwen", args.llm_model, backend="vllm", tp_size=args.tp,
        concurrency=args.concurrency, max_frames=args.max_frames,
    )
    reset_prompt_token_stats()
    begun = time.time()
    with model, open(output_path, "a" if start else "w") as pred_handle, open(
        narrative_path, "a" if start else "w"
    ) as narrative_handle:
        for index, row in enumerate(rows[start:], start=start):
            video_path = os.path.join(video_folder, str(row["video_path"]))
            fps, total_frames = _video_metadata(video_path)
            anchor_indices = baseline_uniform_indices(total_frames, args.caption_frames)
            anchor_frames = extract_frames_by_indices(video_path, anchor_indices)
            timestamps = [frame_index / max(fps, 1e-6) for frame_index in anchor_indices]
            raw_captions = model.generate(
                anchor_frames,
                [{"role": "user", "content": build_caption_prompt(row, timestamps)}],
                max_new_tokens=max(128, args.caption_frames * 18),
            )
            captions = parse_numbered_captions(raw_captions, len(anchor_indices))
            query = f"{row['question']}\n{row['mcq_options']}"
            temporal = "cross_time_ordering" in classify_question_types(row["question"])
            narrative_count = (
                args.narrative_frames if temporal else args.non_temporal_narrative_frames
            )
            narrative_indices, narrative_scores = select_narrative_indices(
                query, captions, anchor_indices, encoder, narrative_count
            )
            final_indices = build_final_indices(
                packs[str(row["video_path"])], narrative_indices, total_frames,
                args.max_frames, args.visual_quota,
            )
            final_frames = extract_frames_by_indices(video_path, final_indices)
            top_notes = [
                f"- {anchor_indices.index(frame_index) + 1}@{frame_index / max(fps, 1e-6):.1f}s: "
                f"{captions[anchor_indices.index(frame_index)]}"
                for frame_index in narrative_indices
            ]
            answer_prompt = (
                "The images are chronological. Use the timestamped caption notes as retrieval "
                "hints, but trust visible image evidence when a note conflicts.\n"
                + "\n".join(top_notes)
                + "\n\n"
                + build_longqa_prompt(row["question"], row["mcq_options"])
            )
            response = model.generate(
                final_frames, [{"role": "user", "content": answer_prompt}], max_new_tokens=16
            )
            pred = build_prediction_row(row, response, prompt_variant="narrative_gate")
            pred.update(
                {
                    "narrative_fingerprint": fingerprint,
                    "narrative_route": "temporal" if temporal else "visual_dominant",
                    "narrative_selected_indices": narrative_indices,
                    "narrative_final_frames": len(final_frames),
                }
            )
            meta = {
                "sample_key": sample_key(row),
                "video_path": row.get("video_path", ""),
                "narrative_fingerprint": fingerprint,
                "anchor_indices": anchor_indices,
                "timestamps": timestamps,
                "captions": captions,
                "raw_captions": str(raw_captions),
                "selected_indices": narrative_indices,
                "selected_scores": narrative_scores,
                "final_indices": final_indices,
            }
            pred_handle.write(json.dumps(pred) + "\n")
            narrative_handle.write(json.dumps(meta) + "\n")
            pred_handle.flush()
            narrative_handle.flush()
            print(f"  Narrative-gate progress: {index + 1}/{len(rows)}")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
