#!/usr/bin/env python3
"""Answer EgoLongQA from normalized HieraMamba temporal proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from longqa_utils import apply_subset, build_longqa_prompt, build_prediction_row, sample_key
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices, load_jsonl
from run_generate_longqa_proofpack import baseline_uniform_indices
from run_generate_longqa_tcot import (
    build_final_pack,
    select_with_qwen,
    selection_fingerprint,
)


STRATEGIES = ("interval_uniform", "interval_pivot", "dynamic_tcot")


def _resolve(path: str | None) -> str | None:
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _video_metadata(video_path: str) -> tuple[float, int]:
    import cv2

    capture = cv2.VideoCapture(video_path)
    try:
        return (
            float(capture.get(cv2.CAP_PROP_FPS)),
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def _proposal_center(proposal: dict[str, Any]) -> float:
    start, end = proposal["segment_seconds"]
    return (float(start) + float(end)) / 2


def select_query_proposals(
    record: dict[str, Any], top_k: int, padding_seconds: float
) -> tuple[list[tuple[float, float]], list[dict[str, Any]]]:
    """Apply lightweight relation filtering to ranked per-query proposals."""
    selected_by_query: dict[str, list[dict[str, Any]]] = {}
    audit: list[dict[str, Any]] = []
    duration = float(record["duration_seconds"])

    for query in record.get("queries", []):
        query_id = str(query["id"])
        limit = 1 if query.get("retrieve") == "top1" else top_k
        candidates = list(query.get("proposals", []))[: max(1, limit)]
        relation = str(query.get("relation", "none"))
        reference_id = None
        direction = None
        if relation.startswith("after:"):
            direction, reference_id = "after", relation.split(":", 1)[1]
        elif relation.startswith("before:"):
            direction, reference_id = "before", relation.split(":", 1)[1]
        if reference_id and selected_by_query.get(reference_id):
            boundary = _proposal_center(selected_by_query[reference_id][0])
            filtered = [
                proposal
                for proposal in candidates
                if (
                    _proposal_center(proposal) > boundary
                    if direction == "after"
                    else _proposal_center(proposal) < boundary
                )
            ]
            if filtered:
                candidates = filtered

        if relation == "first_occurrence" and candidates:
            candidates = [min(candidates, key=_proposal_center)]
        elif relation == "last_occurrence" and candidates:
            candidates = [max(candidates, key=_proposal_center)]
        elif relation == "second_occurrence" and candidates:
            ordered = sorted(candidates, key=_proposal_center)
            candidates = [ordered[min(1, len(ordered) - 1)]]

        selected_by_query[query_id] = candidates
        for proposal in candidates:
            start, end = (float(value) for value in proposal["segment_seconds"])
            window = str(query.get("evidence_window", "around"))
            before = padding_seconds if window in {"around", "before"} else 0.0
            after = padding_seconds if window in {"around", "after"} else 0.0
            interval = (max(0.0, start - before), min(duration, end + after))
            audit.append(
                {
                    "query_id": query_id,
                    "relation": relation,
                    "rank": int(proposal.get("rank", 0)),
                    "score": float(proposal.get("score", 0.0)),
                    "raw_segment_seconds": [start, end],
                    "evidence_interval_seconds": list(interval),
                }
            )

    intervals = sorted(
        {
            (float(item["evidence_interval_seconds"][0]), float(item["evidence_interval_seconds"][1]))
            for item in audit
            if item["evidence_interval_seconds"][1] > item["evidence_interval_seconds"][0]
        }
    )
    return intervals, audit


def sample_interval_frames(
    intervals: list[tuple[float, float]], fps: float, total_frames: int, budget: int
) -> list[int]:
    if not intervals or budget <= 0:
        return []
    per_interval = max(2, math.ceil(budget / len(intervals)))
    candidates: set[int] = set()
    for start, end in intervals:
        first = max(0, min(total_frames - 1, int(round(start * fps))))
        last = max(first, min(total_frames - 1, int(round(end * fps))))
        if per_interval == 1 or first == last:
            candidates.add(first)
            continue
        for position in range(per_interval):
            candidates.add(
                int(round(first + position * (last - first) / (per_interval - 1)))
            )
    ordered = sorted(candidates)
    if len(ordered) <= budget:
        return ordered
    positions = baseline_uniform_indices(len(ordered), budget)
    return [ordered[position] for position in positions]


def add_global_frames(
    selected: list[int], total_frames: int, global_quota: int, max_frames: int
) -> tuple[list[int], list[int]]:
    final = list(dict.fromkeys(selected[:max_frames]))
    added: list[int] = []
    target = min(global_quota, max_frames - len(final))
    pool = min(total_frames, max(target * 4, target))
    while len(added) < target and pool > 0:
        for frame_index in baseline_uniform_indices(total_frames, pool):
            if frame_index not in final and frame_index not in added:
                added.append(frame_index)
            if len(added) >= target:
                break
        if pool == total_frames:
            break
        pool = min(total_frames, max(pool + 1, pool * 2))
    return sorted(final + added), sorted(added)


def select_pivot_inside_intervals(
    proofpack: dict[str, Any],
    intervals: list[tuple[float, float]],
    fps: float,
    total_frames: int,
    local_budget: int,
) -> list[int]:
    inside = []
    for item in proofpack.get("selected", []):
        frame_index = int(item["frame_index"])
        timestamp = float(item.get("timestamp", frame_index / fps))
        if any(start <= timestamp <= end for start, end in intervals):
            inside.append(frame_index)
    inside = list(dict.fromkeys(inside))
    if len(inside) < local_budget:
        interval_fill = sample_interval_frames(
            intervals, fps, total_frames, local_budget * 2
        )
        inside.extend(index for index in interval_fill if index not in inside)
    if len(inside) > local_budget:
        positions = baseline_uniform_indices(len(inside), local_budget)
        inside = [inside[position] for position in positions]
    return sorted(inside)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--proofpack", default=None)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--padding-seconds", type=float, default=2.0)
    parser.add_argument("--candidate-frames", type=int, default=128)
    parser.add_argument("--local-frames", type=int, default=48)
    parser.add_argument("--global-frames", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--tcot-segments", type=int, default=4)
    parser.add_argument("--max-selected-per-segment", type=int, default=6)
    parser.add_argument("--selector-max-pixels", type=int, default=50176)
    parser.add_argument("--neighborhood-radius", type=int, default=1)
    parser.add_argument("--selection-cache-dir", default=None)
    parser.add_argument("--llm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.local_frames + args.global_frames > args.max_frames:
        parser.error("--local-frames + --global-frames cannot exceed --max-frames")
    if args.strategy == "interval_pivot" and not args.proofpack:
        parser.error("--proofpack is required for interval_pivot")
    if args.strategy == "dynamic_tcot" and not args.selection_cache_dir:
        parser.error("--selection-cache-dir is required for dynamic_tcot")
    return args


def main() -> None:
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve(args.input)
    video_folder = _resolve(args.video_folder)
    output_path = _resolve(args.output)
    selection_path = _resolve(args.selection_output)
    eval_path = _resolve(args.eval_output)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    proposal_rows = {sample_key(row): row for row in load_jsonl(_resolve(args.proposals))}
    proofpacks = (
        {sample_key(row): row for row in load_jsonl(_resolve(args.proofpack))}
        if args.proofpack
        else {}
    )
    missing = [sample_key(row) for row in rows if sample_key(row) not in proposal_rows]
    if missing:
        raise RuntimeError(f"Missing HieraMamba proposals for {len(missing)} rows")

    proposal_hash = hashlib.sha1(Path(_resolve(args.proposals)).read_bytes()).hexdigest()[:12]
    fingerprint_data = {
        "strategy": args.strategy,
        "top_k": args.top_k,
        "padding_seconds": args.padding_seconds,
        "candidate_frames": args.candidate_frames,
        "local_frames": args.local_frames,
        "global_frames": args.global_frames,
        "max_frames": args.max_frames,
        "model": args.llm_model,
        "proposal_hash": proposal_hash,
    }
    fingerprint = hashlib.sha1(
        json.dumps(fingerprint_data, sort_keys=True).encode()
    ).hexdigest()[:12]
    existing = [] if args.no_resume else load_jsonl(output_path)
    selections = [] if args.no_resume else load_jsonl(selection_path)
    start = 0
    for index, prediction in enumerate(existing[: len(rows)]):
        if (
            sample_key(prediction) != sample_key(rows[index])
            or prediction.get("hieramamba_fingerprint") != fingerprint
        ):
            break
        start += 1
    start = min(start, len(selections))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(selection_path), exist_ok=True)

    selector_args = SimpleNamespace(
        llm_model=args.llm_model,
        candidate_frames=args.candidate_frames,
        segments=args.tcot_segments,
        max_selected_per_segment=args.max_selected_per_segment,
        selector_max_pixels=args.selector_max_pixels,
        selection_cache_dir=_resolve(args.selection_cache_dir),
        max_frames=args.max_frames,
    )
    selector_fp = (
        selection_fingerprint(selector_args)
        if args.strategy == "dynamic_tcot"
        else None
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
    with (
        model,
        open(output_path, "a" if start else "w") as prediction_handle,
        open(selection_path, "a" if start else "w") as selection_handle,
    ):
        for index, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            video_path = os.path.join(video_folder, str(row["video_path"]))
            fps, total_frames = _video_metadata(video_path)
            if fps <= 0 or total_frames <= 0:
                raise RuntimeError(f"Could not read video metadata: {video_path}")
            intervals, proposal_audit = select_query_proposals(
                proposal_rows[key], args.top_k, args.padding_seconds
            )
            if not intervals:
                local = []
            elif args.strategy == "interval_pivot":
                if key not in proofpacks:
                    raise RuntimeError(f"Missing proof pack for {key}")
                local = select_pivot_inside_intervals(
                    proofpacks[key],
                    intervals,
                    fps,
                    total_frames,
                    args.local_frames,
                )
            else:
                local = sample_interval_frames(
                    intervals,
                    fps,
                    total_frames,
                    (
                        args.candidate_frames
                        if args.strategy == "dynamic_tcot"
                        else args.local_frames
                    ),
                )

            tcot_record = None
            if args.strategy == "dynamic_tcot" and local:
                selector_args.candidate_frames = len(local)
                selector_args.segments = min(args.tcot_segments, len(local))
                tcot_record = select_with_qwen(
                    model,
                    row,
                    video_path,
                    local,
                    selector_args,
                    selector_fp,
                )
                final_indices, pack = build_final_pack(
                    local,
                    tcot_record["selected_indices"],
                    total_frames,
                    args.neighborhood_radius,
                    args.local_frames,
                    args.global_frames,
                    args.max_frames,
                )
                global_added = pack["uniform_added_indices"]
            else:
                final_indices, global_added = add_global_frames(
                    local, total_frames, args.global_frames, args.max_frames
                )
            if len(final_indices) < args.max_frames:
                final_indices, extra = add_global_frames(
                    final_indices,
                    total_frames,
                    args.max_frames - len(final_indices),
                    args.max_frames,
                )
                global_added = sorted(set(global_added + extra))

            frames = extract_frames_by_indices(video_path, final_indices)
            response = model.generate(
                frames,
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
            prediction = build_prediction_row(
                row, response, prompt_variant=f"hieramamba_{args.strategy}"
            )
            prediction["hieramamba_fingerprint"] = fingerprint
            selection = {
                "sample_key": key,
                "video_path": row["video_path"],
                "hieramamba_fingerprint": fingerprint,
                "strategy": args.strategy,
                "top_k": args.top_k,
                "intervals_seconds": [list(interval) for interval in intervals],
                "proposal_audit": proposal_audit,
                "local_indices": local,
                "global_added_indices": global_added,
                "final_indices": final_indices,
                "tcot": tcot_record,
            }
            prediction_handle.write(json.dumps(prediction) + "\n")
            prediction_handle.flush()
            selection_handle.write(json.dumps(selection) + "\n")
            selection_handle.flush()
            print(f"[{index + 1}/{len(rows)}] {key} -> {prediction['prediction']}")

    stats = summarize_prompt_token_stats()
    _print_context_summary(stats)
    Path(output_path).with_name("results_summary.json").write_text(
        json.dumps(
            {
                "runtime_seconds": time.time() - begun,
                "hieramamba_fingerprint": fingerprint,
                "context": stats,
            },
            indent=2,
        )
    )
    _run_eval(input_path, output_path, eval_path)


if __name__ == "__main__":
    main()
