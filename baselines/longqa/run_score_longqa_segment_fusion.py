#!/usr/bin/env python3
"""Score chronological 16-frame blocks from endpoint and option-quota views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from longqa_utils import apply_subset, build_longqa_prompt, load_jsonl, sample_key
from run_generate_longqa_grounded import extract_frames_by_indices
from run_generate_longqa_uncertainty import _video_metadata
from run_score_longqa_evidence_rank_fusion import (
    _answer,
    _index,
    _scored_view,
    _trigger_details,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--challenger-predictions", required=True)
    parser.add_argument("--option-proofpack", action="append", required=True)
    parser.add_argument("--trigger-predictions", action="append", default=[])
    parser.add_argument("--trigger-min-votes", type=int, default=3)
    parser.add_argument(
        "--target-mode",
        choices=("disagreement_and_consensus", "consensus_only"),
        default="disagreement_and_consensus",
        help="Choose whether to include ordinary endpoint/option disagreements.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--frames-per-block", type=int, default=16)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "baseline": os.path.abspath(args.baseline_predictions),
        "challenger": os.path.abspath(args.challenger_predictions),
        "proofpacks": sorted(os.path.abspath(path) for path in args.option_proofpack),
        "triggers": sorted(os.path.abspath(path) for path in args.trigger_predictions),
        "trigger_min_votes": args.trigger_min_votes,
        "target_mode": args.target_mode,
        "subset": os.path.abspath(args.subset_file),
        "model": args.llm_model,
        "blocks": args.blocks,
        "frames_per_block": args.frames_per_block,
        "prompt": "baseline_choice_letter_v1",
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _split_blocks(indices: list[int], blocks: int, frames_per_block: int) -> list[list[int]]:
    expected = blocks * frames_per_block
    if len(indices) != expected or len(set(indices)) != expected:
        raise ValueError(f"Expected {expected} unique frame indices, received {len(indices)}")
    ordered = sorted(indices)
    return [
        ordered[offset : offset + frames_per_block]
        for offset in range(0, expected, frames_per_block)
    ]


def main() -> None:
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats, uniform_full_video_indices
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    if args.blocks < 2 or args.frames_per_block < 1:
        raise ValueError("At least two non-empty blocks are required")
    rows = apply_subset(load_jsonl(args.input), args.subset_file)
    baseline = _index(load_jsonl(args.baseline_predictions), "baseline")
    challenger = _index(load_jsonl(args.challenger_predictions), "challenger")
    proofpacks: dict[str, dict[str, Any]] = {}
    for path in args.option_proofpack:
        for key, record in _index(load_jsonl(path), f"proofpack {path}").items():
            if key in proofpacks:
                raise ValueError(f"Proofpack key occurs in multiple files: {key}")
            proofpacks[key] = record
    trigger_sources = [
        _index(load_jsonl(path), f"trigger predictions {path}")
        for path in args.trigger_predictions
    ]
    if trigger_sources and not 1 <= args.trigger_min_votes <= len(trigger_sources):
        raise ValueError(
            f"--trigger-min-votes must be between 1 and {len(trigger_sources)}"
        )
    for row in rows:
        key = sample_key(row)
        if key not in baseline or key not in challenger or key not in proofpacks:
            raise RuntimeError(f"Missing endpoint, option, or proofpack artifact for {key}")
        if any(key not in source for source in trigger_sources):
            raise RuntimeError(f"Missing trigger prediction for {key}")

    fingerprint = _fingerprint(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = [] if args.no_resume or not output.exists() else load_jsonl(str(output))
    start = 0
    for index, record in enumerate(existing[: len(rows)]):
        if (
            record.get("sample_key") != sample_key(rows[index])
            or record.get("segment_fusion_fingerprint") != fingerprint
        ):
            break
        start += 1
    if len(existing) != start:
        with output.open("w") as handle:
            for record in existing[:start]:
                handle.write(json.dumps(record) + "\n")

    targets = 0
    consensus_targets = 0
    for row in rows:
        key = sample_key(row)
        endpoint_answer = _answer(baseline[key])
        option_answer = _answer(challenger[key])
        trigger = _trigger_details(
            key,
            endpoint_answer,
            option_answer,
            trigger_sources,
            args.trigger_min_votes,
        )
        direct_target = (
            endpoint_answer != option_answer
            and args.target_mode == "disagreement_and_consensus"
        )
        targets += int(direct_target or trigger["applied"])
        consensus_targets += int(trigger["applied"])
    calls_per_target = args.blocks * 2
    print(
        f"Segment fusion: rows={len(rows)} targets={targets} "
        f"consensus_challenges={consensus_targets} calls={targets * calls_per_target} "
        f"resume={start} fingerprint={fingerprint}"
    )

    model = VLLMModel(
        args.llm_model,
        tp_size=1,
        concurrency=args.concurrency,
        max_frames=args.frames_per_block,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    begun = time.time()
    calls = 0
    with model, output.open("a" if start else "w") as handle:
        for index, row in enumerate(rows[start:], start=start):
            row_begun = time.perf_counter()
            key = sample_key(row)
            endpoint_answer = _answer(baseline[key])
            option_answer = _answer(challenger[key])
            trigger = _trigger_details(
                key,
                endpoint_answer,
                option_answer,
                trigger_sources,
                args.trigger_min_votes,
            )
            direct_target = (
                endpoint_answer != option_answer
                and args.target_mode == "disagreement_and_consensus"
            )
            should_score = direct_target or trigger["applied"]
            record: dict[str, Any] = {
                "sample_key": key,
                "video_path": row.get("video_path"),
                "question": row.get("question"),
                "endpoint_answer": endpoint_answer,
                "option_answer": option_answer,
                "segment_fusion_applied": should_score,
                "direct_disagreement": endpoint_answer != option_answer,
                "direct_disagreement_targeted": direct_target,
                "consensus_challenge": trigger,
                "segment_fusion_fingerprint": fingerprint,
            }
            if should_score:
                video_path = os.path.join(args.video_folder, str(row["video_path"]))
                _, total_frames = _video_metadata(video_path)
                total_selected = args.blocks * args.frames_per_block
                endpoint_indices = uniform_full_video_indices(
                    total_frames,
                    total_selected,
                    total_selected,
                    "endpoint_inclusive",
                )
                option_indices = [
                    int(item["frame_index"])
                    for item in proofpacks[key].get("selected", [])
                ]
                endpoint_blocks = _split_blocks(
                    endpoint_indices, args.blocks, args.frames_per_block
                )
                option_blocks = _split_blocks(
                    option_indices, args.blocks, args.frames_per_block
                )
                prompt = build_longqa_prompt(row["question"], row["mcq_options"], "baseline")
                messages = [{"role": "user", "content": prompt}]
                scored_endpoint = []
                scored_option = []
                for block_index, block_indices in enumerate(endpoint_blocks):
                    frames = extract_frames_by_indices(video_path, block_indices)
                    scored_endpoint.append(
                        {
                            "block_index": block_index,
                            "frame_indices": block_indices,
                            **_scored_view(model.score_choice_letters(frames, messages)),
                        }
                    )
                    calls += 1
                for block_index, block_indices in enumerate(option_blocks):
                    frames = extract_frames_by_indices(video_path, block_indices)
                    scored_option.append(
                        {
                            "block_index": block_index,
                            "frame_indices": block_indices,
                            **_scored_view(model.score_choice_letters(frames, messages)),
                        }
                    )
                    calls += 1
                record.update(
                    {
                        "endpoint_blocks": scored_endpoint,
                        "option_blocks": scored_option,
                    }
                )
            record["segment_fusion_seconds"] = round(time.perf_counter() - row_begun, 3)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"  Segment progress: {index + 1}/{len(rows)} calls={calls}")

    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())


if __name__ == "__main__":
    main()
