#!/usr/bin/env python3
"""Run matched Temporal Chain-of-Thought experiments for EgoLongQA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from longqa_utils import (
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    normalize_answer,
    sample_key,
)
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices, load_jsonl
from run_generate_longqa_proofpack import (
    baseline_uniform_indices,
    resize_frame_to_max_pixels,
)


SELECTOR_PROMPT_VERSION = "tcot-direct-v1"
ANSWER_PROMPT_VERSION = "tcot-answer-v2"


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _video_metadata(video_path: str, attempts: int = 3) -> tuple[float, int]:
    import cv2

    for attempt in range(attempts):
        capture = cv2.VideoCapture(video_path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if capture.isOpened() and fps > 0 and total_frames > 0:
                return fps, total_frames
        finally:
            capture.release()
        if attempt + 1 < attempts:
            time.sleep(2**attempt)
    return 0.0, 0


def split_positions(length: int, segments: int) -> list[list[int]]:
    """Split a chronological candidate sequence into contiguous segments."""
    if length <= 0 or segments <= 0:
        raise ValueError("length and segments must be positive")
    segments = min(length, segments)
    boundaries = [round(index * length / segments) for index in range(segments + 1)]
    return [list(range(boundaries[i], boundaries[i + 1])) for i in range(segments)]


def selector_schema(frame_count: int, max_selected: int) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "frame_ids": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": frame_count},
                "minItems": 0,
                "maxItems": min(frame_count, max_selected),
            },
            "justification": {"type": "string", "maxLength": 600},
        },
        "required": ["frame_ids", "justification"],
        "additionalProperties": False,
    }


def build_selector_prompt(row: dict[str, Any], frame_count: int, max_selected: int) -> str:
    return (
        f"You are given {frame_count} images sampled chronologically from one segment "
        "of a long egocentric video. The images supplied before this text are FrameID "
        f"1 through FrameID {frame_count}, in exactly that order.\n\n"
        "Select the smallest sufficient set of frames that contains visible evidence "
        "needed to answer the question. Include both the temporal anchor and the requested "
        "before/after event when relevant. For first, last, repeated, returned, again, or "
        "counting questions, retain every potentially relevant occurrence visible in this "
        "segment. If this segment contains no relevant visible evidence, return an empty "
        "frame_ids list. Do not answer the question. Do not select a frame solely because "
        "an answer option sounds plausible.\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n{row['mcq_options']}\n\n"
        f"Return at most {max_selected} FrameIDs and a short visible-evidence justification."
    )


def build_answer_prompt(row: dict[str, Any], retrieved: bool) -> str:
    prefix = (
        "The images are chronological frames selected by a visual evidence agent. "
        if retrieved
        else "The images are chronological frames sampled uniformly from the video. "
    )


def has_final_answer_marker(response: object) -> bool:
    return bool(
        re.search(
            r"\bfinal\s+answer\s*:\s*\(?[A-Da-d]\)?\b",
            str(response),
            re.IGNORECASE,
        )
    )
    return (
        prefix
        + "Identify the relevant event or events and reason about their temporal order. "
        "Use only visible evidence and compare all answer options. Give a concise evidence "
        "summary, then finish with exactly `Final Answer: X`, where X is A, B, C, or D.\n\n"
        f"Question: {row['question']}\n\n"
        f"Options:\n{row['mcq_options']}"
    )


def normalize_local_ids(raw: object, frame_count: int, max_selected: int) -> list[int]:
    if not isinstance(raw, list):
        return []
    selected: list[int] = []
    for value in raw:
        try:
            frame_id = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= frame_id <= frame_count and frame_id not in selected:
            selected.append(frame_id)
        if len(selected) == max_selected:
            break
    return sorted(selected)


def uniformly_cap(values: list[int], limit: int) -> list[int]:
    values = sorted(set(values))
    if limit <= 0:
        return []
    if len(values) <= limit:
        return values
    positions = baseline_uniform_indices(len(values), limit)
    return [values[position] for position in positions]


def expand_candidate_neighborhoods(
    candidate_indices: list[int], selected_indices: list[int], radius: int
) -> list[int]:
    position_by_frame = {frame_index: position for position, frame_index in enumerate(candidate_indices)}
    expanded: set[int] = set()
    for frame_index in selected_indices:
        if frame_index not in position_by_frame:
            continue
        center = position_by_frame[frame_index]
        for position in range(max(0, center - radius), min(len(candidate_indices), center + radius + 1)):
            expanded.add(candidate_indices[position])
    return sorted(expanded)


def build_final_pack(
    candidate_indices: list[int],
    selected_indices: list[int],
    total_frames: int,
    neighborhood_radius: int,
    selected_quota: int,
    uniform_quota: int,
    max_frames: int,
) -> tuple[list[int], dict[str, Any]]:
    expanded = expand_candidate_neighborhoods(
        candidate_indices, selected_indices, neighborhood_radius
    )
    selected = uniformly_cap(expanded, min(selected_quota, max_frames))
    target_uniform = min(uniform_quota, max_frames - len(selected))
    uniform_added: list[int] = []
    pool_size = min(total_frames, max(target_uniform * 4, target_uniform))
    while len(uniform_added) < target_uniform and pool_size > 0:
        for frame_index in baseline_uniform_indices(total_frames, pool_size):
            if frame_index not in selected and frame_index not in uniform_added:
                uniform_added.append(frame_index)
            if len(uniform_added) == target_uniform:
                break
        if pool_size == total_frames:
            break
        pool_size = min(total_frames, max(pool_size + 1, pool_size * 2))
    final_indices = sorted(selected + uniform_added)
    return final_indices, {
        "expanded_indices": expanded,
        "selected_pack_indices": selected,
        "uniform_added_indices": uniform_added,
    }


def selection_fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "version": SELECTOR_PROMPT_VERSION,
        "model": args.llm_model,
        "candidate_frames": args.candidate_frames,
        "segments": args.segments,
        "max_selected_per_segment": args.max_selected_per_segment,
        "selector_max_pixels": args.selector_max_pixels,
        "question_input": "question_options",
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def inference_fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "version": ANSWER_PROMPT_VERSION,
        "mode": args.mode,
        "model": args.llm_model,
        "max_frames": args.max_frames,
        "candidate_frames": args.candidate_frames,
        "segments": args.segments,
        "max_selected_per_segment": args.max_selected_per_segment,
        "selector_max_pixels": args.selector_max_pixels,
        "neighborhood_radius": args.neighborhood_radius,
        "selected_quota": args.selected_quota,
        "uniform_quota": args.uniform_quota,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _cache_path(cache_dir: str, fingerprint: str, key: str) -> Path:
    digest = hashlib.sha1(key.encode()).hexdigest()
    return Path(cache_dir) / fingerprint / f"{digest}.json"


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload) + "\n")
    os.replace(temporary, path)


def select_with_qwen(
    model: Any,
    row: dict[str, Any],
    video_path: str,
    candidate_indices: list[int],
    args: argparse.Namespace,
    fingerprint: str,
) -> dict[str, Any]:
    key = sample_key(row)
    cache_path = _cache_path(args.selection_cache_dir, fingerprint, key)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if (
            cached.get("selection_fingerprint") == fingerprint
            and cached.get("sample_key") == key
            and cached.get("candidate_indices") == candidate_indices
        ):
            cached["cache_hit"] = True
            return cached

    candidate_frames = extract_frames_by_indices(video_path, candidate_indices)
    candidate_frames = [
        resize_frame_to_max_pixels(frame, args.selector_max_pixels)
        for frame in candidate_frames
    ]
    position_segments = split_positions(len(candidate_indices), args.segments)
    batch_frames: list[list[object]] = []
    batch_messages: list[list[dict[str, str]]] = []
    for positions in position_segments:
        batch_frames.append([candidate_frames[position] for position in positions])
        batch_messages.append(
            [
                {
                    "role": "user",
                    "content": build_selector_prompt(
                        row, len(positions), args.max_selected_per_segment
                    ),
                }
            ]
        )
    schema = selector_schema(
        max(len(positions) for positions in position_segments),
        args.max_selected_per_segment,
    )
    responses = model.generate_json_batch(
        batch_frames,
        batch_messages,
        schema,
        schema_name="tcot_frame_selection",
        max_new_tokens=256,
    )
    segment_records: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    for segment_index, (positions, response) in enumerate(zip(position_segments, responses)):
        local_ids = normalize_local_ids(
            response.get("frame_ids", []),
            len(positions),
            args.max_selected_per_segment,
        )
        if not local_ids and args.segments == 1:
            local_ids = baseline_uniform_indices(
                len(positions), min(2, len(positions))
            )
            local_ids = [position + 1 for position in local_ids]
        global_indices = [candidate_indices[positions[frame_id - 1]] for frame_id in local_ids]
        selected_indices.extend(global_indices)
        segment_records.append(
            {
                "segment": segment_index,
                "candidate_start": candidate_indices[positions[0]],
                "candidate_end": candidate_indices[positions[-1]],
                "local_frame_ids": local_ids,
                "selected_indices": global_indices,
                "justification": str(response.get("justification", "")).strip(),
            }
        )
    fallback_used = not selected_indices
    if fallback_used:
        fallback_positions = baseline_uniform_indices(
            len(candidate_indices), min(args.max_frames, len(candidate_indices))
        )
        selected_indices = [candidate_indices[position] for position in fallback_positions]
    record = {
        "sample_key": key,
        "video_path": row.get("video_path", ""),
        "selection_fingerprint": fingerprint,
        "candidate_indices": candidate_indices,
        "selected_indices": sorted(set(selected_indices)),
        "segments": segment_records,
        "all_empty_fallback": fallback_used,
        "cache_hit": False,
    }
    _write_atomic(cache_path, record)
    return record


def _resume_prefix(
    rows: list[dict[str, Any]], existing: list[dict[str, Any]], fingerprint: str
) -> int:
    start = 0
    for index, prediction in enumerate(existing[: len(rows)]):
        if (
            sample_key(prediction) != sample_key(rows[index])
            or prediction.get("tcot_inference_fingerprint") != fingerprint
        ):
            break
        start += 1
    return start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl")
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-output", default=None)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--selection-cache-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["uniform_answer_cot", "single_step", "dynamic_segment"],
        required=True,
    )
    parser.add_argument("--candidate-frames", type=int, default=128)
    parser.add_argument("--segments", type=int, default=1)
    parser.add_argument("--max-selected-per-segment", type=int, default=12)
    parser.add_argument("--selector-max-pixels", type=int, default=50176)
    parser.add_argument("--neighborhood-radius", type=int, default=2)
    parser.add_argument("--selected-quota", type=int, default=64)
    parser.add_argument("--uniform-quota", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--llm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()
    if args.mode == "uniform_answer_cot":
        args.candidate_frames = args.max_frames
        args.segments = 1
        args.selected_quota = 0
        args.uniform_quota = args.max_frames
    if args.selected_quota + args.uniform_quota > args.max_frames:
        parser.error("selected-quota + uniform-quota cannot exceed max-frames")
    if args.mode != "uniform_answer_cot" and not args.selection_output:
        parser.error("selection-output is required for TCoT selection modes")
    if args.candidate_frames < args.segments:
        parser.error("candidate-frames must be at least segments")
    return args


def main() -> None:
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    eval_output = _resolve_path(args.eval_output)
    args.selection_cache_dir = _resolve_path(args.selection_cache_dir)
    selection_output = _resolve_path(args.selection_output) if args.selection_output else None
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    infer_fingerprint = inference_fingerprint(args)
    select_fingerprint = selection_fingerprint(args)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if selection_output:
        os.makedirs(os.path.dirname(selection_output) or ".", exist_ok=True)
    existing = [] if args.no_resume or not os.path.exists(output_path) else load_jsonl(output_path)
    start = _resume_prefix(rows, existing, infer_fingerprint)
    selection_existing: list[dict[str, Any]] = []
    if selection_output:
        selection_existing = (
            []
            if args.no_resume or not os.path.exists(selection_output)
            else load_jsonl(selection_output)
        )
        selection_start = 0
        for index, record in enumerate(selection_existing[: len(rows)]):
            if (
                record.get("sample_key") != sample_key(rows[index])
                or record.get("tcot_inference_fingerprint") != infer_fingerprint
                or record.get("selection_fingerprint") != select_fingerprint
            ):
                break
            selection_start += 1
        start = min(start, selection_start)
    if len(existing) != start:
        with open(output_path, "w") as handle:
            for prediction in existing[:start]:
                handle.write(json.dumps(prediction) + "\n")
    if selection_output and len(selection_existing) != start:
        with open(selection_output, "w") as handle:
            for record in selection_existing[:start]:
                handle.write(json.dumps(record) + "\n")

    max_images_per_call = max(args.max_frames, math.ceil(args.candidate_frames / args.segments))
    model = VLLMModel(
        args.llm_model,
        tp_size=1,
        concurrency=args.concurrency,
        max_frames=max_images_per_call,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    begun = time.time()
    selection_mode = "a" if selection_output and start else "w"
    selection_handle = open(selection_output, selection_mode) if selection_output else None
    try:
        with model, open(output_path, "a" if start else "w") as prediction_handle:
            for row_index, row in enumerate(rows[start:], start=start):
                video_path = os.path.join(video_folder, str(row["video_path"]))
                _fps, total_frames = _video_metadata(video_path)
                if total_frames <= 0:
                    raise RuntimeError(f"Could not read video metadata: {video_path}")

                if args.mode == "uniform_answer_cot":
                    final_indices = baseline_uniform_indices(total_frames, args.max_frames)
                    selection_record: dict[str, Any] | None = None
                    pack_meta = {
                        "expanded_indices": [],
                        "selected_pack_indices": [],
                        "uniform_added_indices": final_indices,
                    }
                else:
                    candidate_indices = baseline_uniform_indices(
                        total_frames, args.candidate_frames
                    )
                    selection_record = select_with_qwen(
                        model,
                        row,
                        video_path,
                        candidate_indices,
                        args,
                        select_fingerprint,
                    )
                    final_indices, pack_meta = build_final_pack(
                        candidate_indices,
                        selection_record["selected_indices"],
                        total_frames,
                        args.neighborhood_radius,
                        args.selected_quota,
                        args.uniform_quota,
                        args.max_frames,
                    )
                    audited = dict(selection_record)
                    audited.update(pack_meta)
                    audited["final_indices"] = final_indices
                    audited["tcot_inference_fingerprint"] = infer_fingerprint
                    if selection_handle is not None:
                        selection_handle.write(json.dumps(audited) + "\n")
                        selection_handle.flush()

                final_frames = extract_frames_by_indices(video_path, final_indices)
                if args.mode == "uniform_answer_cot":
                    response = model.generate(
                        final_frames,
                        [{"role": "user", "content": build_answer_prompt(row, False)}],
                        max_new_tokens=512,
                    )
                    if not has_final_answer_marker(response):
                        terse_response = model.generate(
                            final_frames,
                            [
                                {
                                    "role": "user",
                                    "content": build_longqa_prompt(
                                        row["question"], row["mcq_options"]
                                    ),
                                }
                            ],
                            max_new_tokens=16,
                        )
                        terse_answer = normalize_answer(terse_response)
                        if terse_answer not in tuple("ABCD"):
                            raise RuntimeError(
                                "Answer-CoT response was truncated and terse retry did not "
                                "produce a valid option letter"
                            )
                        response = f"{response.rstrip()}\n\nFinal Answer: {terse_answer}"
                else:
                    response = model.generate(
                        final_frames,
                        [
                            {
                                "role": "user",
                                "content": build_longqa_prompt(
                                    row["question"], row["mcq_options"]
                                ),
                            }
                        ],
                        max_new_tokens=16,
                    )
                prediction = build_prediction_row(row, response, prompt_variant=args.mode)
                prediction.update(
                    {
                        "tcot_inference_fingerprint": infer_fingerprint,
                        "tcot_selection_fingerprint": (
                            select_fingerprint if selection_record else None
                        ),
                        "tcot_mode": args.mode,
                        "tcot_final_indices": final_indices,
                        "tcot_final_frames": len(final_indices),
                        "tcot_selected_seed_count": (
                            len(selection_record["selected_indices"])
                            if selection_record
                            else 0
                        ),
                        "tcot_selected_pack_count": len(
                            pack_meta["selected_pack_indices"]
                        ),
                        "tcot_uniform_added_count": len(
                            pack_meta["uniform_added_indices"]
                        ),
                    }
                )
                prediction_handle.write(json.dumps(prediction) + "\n")
                prediction_handle.flush()
                print(
                    f"  TCoT progress: {row_index + 1}/{len(rows)} "
                    f"({len(final_indices)} final frames)"
                )
    finally:
        if selection_handle is not None:
            selection_handle.close()

    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
