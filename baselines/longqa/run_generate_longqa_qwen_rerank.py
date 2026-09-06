#!/usr/bin/env python3
"""Build LongQA proof packs with SigLIP2 recall and Qwen-VL window reranking."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

from longqa_utils import apply_subset, parse_mcq_options, query_hash, sample_key
from run_generate_longqa_grounded import (
    CandidateFrame,
    _uniform_positions,
    create_text_image_grounder,
    extract_frames_by_indices,
    load_jsonl,
    load_jsonl_if_exists,
    load_or_encode_grounder_features,
)
from run_generate_longqa_proofpack import (
    TemporalProgram,
    build_token_safe_retrieval_queries,
    compile_temporal_program_v2,
)

logger = logging.getLogger(__name__)
STRATEGY = "qwen_window_rerank"
SCHEMA = 1


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _rank_nms(
    scores: list[float],
    candidates: list[CandidateFrame],
    count: int,
    nms_seconds: float,
    eligible: set[int] | None = None,
) -> list[int]:
    ranked = sorted(
        (idx for idx in range(len(scores)) if eligible is None or idx in eligible),
        key=lambda idx: (-scores[idx], idx),
    )
    selected: list[int] = []
    for idx in ranked:
        timestamp = candidates[idx].timestamp
        if all(
            abs(timestamp - candidates[other].timestamp) >= nms_seconds
            for other in selected
        ):
            selected.append(idx)
            if len(selected) == count:
                return selected
    for idx in ranked:
        if idx not in selected:
            selected.append(idx)
            if len(selected) == count:
                break
    return selected


def build_siglip_shortlist(
    candidates: list[CandidateFrame],
    component_scores: dict[str, list[float]],
    pivot_scores: list[float],
    max_windows: int,
    nms_seconds: float,
) -> list[int]:
    """Take balanced recall from the pivot, target, and every option query."""
    selected: list[int] = []
    seen: set[int] = set()

    def add(scores: list[float], count: int) -> None:
        for idx in _rank_nms(scores, candidates, count, nms_seconds):
            if idx not in seen and len(selected) < max_windows:
                seen.add(idx)
                selected.append(idx)

    if pivot_scores:
        add(pivot_scores, 4)
    if "target" in component_scores:
        add(component_scores["target"], 4)
    for label in sorted(label for label in component_scores if label.startswith("option_")):
        add(component_scores[label], 4)

    aggregate = [
        max(scores[idx] for scores in component_scores.values())
        for idx in range(len(candidates))
    ]
    if pivot_scores:
        aggregate = [max(value, pivot_scores[idx]) for idx, value in enumerate(aggregate)]
    add(aggregate, max_windows)
    return sorted(selected[:max_windows])


def _temporal_eligible(
    program: TemporalProgram,
    centers: list[int],
    candidates: list[CandidateFrame],
    anchor_center: int | None,
) -> set[int]:
    if anchor_center is None or program.operator not in {"AFTER", "BEFORE"}:
        return set(centers)
    anchor_time = candidates[anchor_center].timestamp
    if program.operator == "AFTER":
        return {idx for idx in centers if candidates[idx].timestamp > anchor_time}
    return {idx for idx in centers if candidates[idx].timestamp < anchor_time}


def select_reranked_centers(
    centers: list[int],
    candidates: list[CandidateFrame],
    option_scores: dict[str, list[float]],
    pivot_scores: list[float] | None,
    program: TemporalProgram,
    mode: str,
    final_centers: int,
    contrast_weight: float,
    nms_seconds: float,
) -> tuple[list[int], dict[str, Any]]:
    """Select unique centers with equal option opportunity and optional time gating."""
    labels = sorted(option_scores)
    if not labels:
        raise RuntimeError("Qwen reranking requires four option score lists")
    if any(len(option_scores[label]) != len(centers) for label in labels):
        raise ValueError("option score length does not match shortlist")

    pivot_centers: list[int] = []
    anchor_center: int | None = None
    if pivot_scores:
        pivot_local = _rank_nms(
            pivot_scores,
            [candidates[idx] for idx in centers],
            2,
            nms_seconds,
        )
        pivot_centers = [centers[idx] for idx in pivot_local]
        anchor_center = pivot_centers[0] if pivot_centers else None

    eligible = set(centers)
    if mode == "temporal":
        eligible = _temporal_eligible(program, centers, candidates, anchor_center)
    fallback = False
    if len(eligible) < len(labels):
        eligible = set(centers)
        fallback = True

    adjusted: dict[str, dict[int, float]] = defaultdict(dict)
    raw_by_center: dict[str, dict[int, float]] = defaultdict(dict)
    for local_idx, center in enumerate(centers):
        values = {label: option_scores[label][local_idx] for label in labels}
        for label, raw in values.items():
            competing = max(value for other, value in values.items() if other != label)
            adjusted[label][center] = raw + contrast_weight * (raw - competing)
            raw_by_center[label][center] = raw

    temporal_question = program.operator != "GLOBAL" and bool(pivot_centers)
    reserved_pivots = pivot_centers[: min(2, final_centers)] if temporal_question else []
    option_budget = final_centers - len(reserved_pivots)
    base_quota = max(1, option_budget // len(labels))
    selected: list[int] = list(reserved_pivots)
    selected_set = set(selected)
    option_selected: dict[str, list[int]] = {label: [] for label in labels}

    proposals = sorted(
        [
            (
            adjusted[label][center],
            label,
            center,
            )
            for label in labels
            for center in eligible
        ],
        key=lambda item: (-item[0], item[1], item[2]),
    )
    for _score, label, center in proposals:
        if len(option_selected[label]) >= base_quota or center in selected_set:
            continue
        if any(
            abs(candidates[center].timestamp - candidates[other].timestamp) < nms_seconds
            for other in option_selected[label]
        ):
            continue
        selected.append(center)
        selected_set.add(center)
        option_selected[label].append(center)
        if all(len(option_selected[item]) >= base_quota for item in labels):
            break

    # Fill any missing quota without NMS before assigning optional extra centers.
    for label in labels:
        for center in sorted(eligible, key=lambda idx: (-adjusted[label][idx], idx)):
            if len(option_selected[label]) >= base_quota:
                break
            if center not in selected_set:
                selected.append(center)
                selected_set.add(center)
                option_selected[label].append(center)

    extras = sorted(
        [
            (
                adjusted[label][center],
                label,
                center,
            )
            for label in labels
            for center in eligible
            if center not in selected_set
        ],
        key=lambda item: (-item[0], item[1], item[2]),
    )
    for _score, label, center in extras:
        if len(selected) >= final_centers:
            break
        selected.append(center)
        selected_set.add(center)
        option_selected[label].append(center)

    if len(selected) < final_centers:
        aggregate = sorted(
            centers,
            key=lambda center: (
                -max(adjusted[label][center] for label in labels),
                center,
            ),
        )
        for center in aggregate:
            if center not in selected_set:
                selected.append(center)
                selected_set.add(center)
                if len(selected) == final_centers:
                    break
    if len(selected) != final_centers:
        raise RuntimeError(f"selected {len(selected)}/{final_centers} reranked windows")

    return sorted(selected), {
        "mode": mode,
        "operator": program.operator,
        "anchor_center": anchor_center,
        "pivot_centers": pivot_centers,
        "temporal_filter_fallback": fallback,
        "eligible_centers": sorted(eligible),
        "option_centers": {label: sorted(values) for label, values in option_selected.items()},
        "raw_scores": {
            label: {str(center): round(value, 6) for center, value in values.items()}
            for label, values in raw_by_center.items()
        },
        "adjusted_scores": {
            label: {str(center): round(value, 6) for center, value in values.items()}
            for label, values in adjusted.items()
        },
    }


def build_final_frame_indices(
    candidates: list[CandidateFrame],
    centers: list[int],
    total_video_frames: int,
    window_radius: int,
    global_frames: int,
    final_frames: int,
) -> tuple[list[int], set[int], list[int]]:
    retrieval: set[int] = set()
    for center in centers:
        for candidate_pos in range(
            max(0, center - window_radius),
            min(len(candidates), center + window_radius + 1),
        ):
            retrieval.add(candidates[candidate_pos].index)
    if len(retrieval) > final_frames - global_frames:
        raise RuntimeError("reranked windows exceed the reserved retrieval budget")

    global_indices = _uniform_positions(total_video_frames, global_frames)
    selected = set(retrieval)
    selected.update(global_indices)
    for index in _uniform_positions(total_video_frames, final_frames):
        if len(selected) >= final_frames:
            break
        selected.add(index)
    if len(selected) < final_frames:
        for index in range(total_video_frames):
            if len(selected) >= final_frames:
                break
            selected.add(index)
    if len(selected) != final_frames:
        raise RuntimeError(f"selected {len(selected)}/{final_frames} final frames")
    return sorted(selected), retrieval, global_indices


class QwenVLWindowReranker:
    """Batched yes/no relevance scoring following the official Qwen reranker."""

    def __init__(
        self,
        model_id: str,
        revision: str | None,
        batch_size: int,
        max_pixels: int,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.batch_size = batch_size
        self.max_pixels = max_pixels
        lm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda")
        self.model = lm.model
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
        )
        vocabulary = self.processor.tokenizer.get_vocab()
        yes_id = vocabulary["yes"]
        no_id = vocabulary["no"]
        self.yes_weight = lm.lm_head.weight[yes_id].detach()
        self.no_weight = lm.lm_head.weight[no_id].detach()

    def _conversation(
        self,
        query: str,
        frames: list[object],
        timestamps: list[float],
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "<Instruct>: Judge whether this short chronological video window "
                    "contains direct visual evidence for the candidate hypothesis at "
                    "the time requested by the question. Do not reward a matching "
                    "event from a different occurrence or time.\n<Query>: " + query
                    + "\n<Document>: Frames are ordered from earliest to latest."
                ),
            }
        ]
        for frame, timestamp in zip(frames, timestamps):
            content.append(
                {"type": "text", "text": f"Frame at {timestamp:.1f} seconds:"}
            )
            content.append(
                {
                    "type": "image",
                    "image": frame,
                    "min_pixels": 3136,
                    "max_pixels": self.max_pixels,
                }
            )
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Judge whether the Document meets the requirements based "
                            "on the Query and Instruct. Answer only yes or no."
                        ),
                    }
                ],
            },
            {"role": "user", "content": content},
        ]

    def score(
        self,
        query: str,
        windows: list[list[object]],
        timestamps: list[list[float]],
    ) -> list[float]:
        from qwen_vl_utils import process_vision_info

        results: list[float] = []
        for start in range(0, len(windows), self.batch_size):
            conversations = [
                self._conversation(query, frames, times)
                for frames, times in zip(
                    windows[start : start + self.batch_size],
                    timestamps[start : start + self.batch_size],
                )
            ]
            texts = self.processor.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=True,
            )
            images, videos = process_vision_info(conversations, image_patch_size=16)
            inputs = self.processor(
                text=texts,
                images=images,
                videos=videos,
                padding=True,
                truncation=False,
                do_resize=False,
                return_tensors="pt",
            )
            inputs = {
                key: value.to("cuda") if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with self.torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state[:, -1]
                logits = hidden @ (self.yes_weight - self.no_weight)
                scores = self.torch.sigmoid(logits).float().cpu().tolist()
            results.extend(float(value) for value in scores)
        return results


def _video_metadata(video_path: str) -> tuple[int, float]:
    import cv2

    capture = cv2.VideoCapture(video_path)
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if total <= 0 or fps <= 0:
        raise RuntimeError(f"invalid video metadata: {video_path}")
    return total, fps


def _fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "schema": SCHEMA,
        "mode": args.mode,
        "candidate_frames": args.candidate_frames,
        "shortlist_windows": args.shortlist_windows,
        "final_windows": args.final_windows,
        "window_radius": args.window_radius,
        "global_frames": args.global_frames,
        "final_frames": args.final_frames,
        "contrast_weight": args.contrast_weight,
        "nms_seconds": args.nms_seconds,
        "grounder_model": args.grounder_model,
        "reranker_model": args.reranker_model,
        "reranker_revision": args.reranker_revision,
        "reranker_max_pixels": args.reranker_max_pixels,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode()).hexdigest()[:12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("balanced", "temporal"), required=True)
    parser.add_argument("--candidate-frames", type=int, default=128)
    parser.add_argument("--shortlist-windows", type=int, default=24)
    parser.add_argument("--final-windows", type=int, default=8)
    parser.add_argument("--window-radius", type=int, default=1)
    parser.add_argument("--global-frames", type=int, default=40)
    parser.add_argument("--final-frames", type=int, default=64)
    parser.add_argument("--contrast-weight", type=float, default=0.5)
    parser.add_argument("--nms-seconds", type=float, default=8.0)
    parser.add_argument("--grounder-model", default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--grounder-cache-dir", required=True)
    parser.add_argument("--grounder-batch-size", type=int, default=32)
    parser.add_argument("--reranker-model", default="Qwen/Qwen3-VL-Reranker-2B")
    parser.add_argument(
        "--reranker-revision",
        default="93eac850736c677b682c67fc0302b03e552a7b16",
    )
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--reranker-max-pixels", type=int, default=100352)
    return parser.parse_args()


def main() -> None:
    import torch

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.final_windows * (2 * args.window_radius + 1) > (
        args.final_frames - args.global_frames
    ):
        raise ValueError("window budget exceeds final retrieval-frame reservation")

    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    cache_dir = _resolve_path(args.grounder_cache_dir)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    fingerprint = _fingerprint(args)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cached = load_jsonl_if_exists(output_path)
    records: list[dict[str, Any]] = []
    for index, record in enumerate(cached[: len(rows)]):
        if (
            int(record.get("index", -1)) != index
            or str(record.get("sample_key", "")) != sample_key(rows[index])
            or str(record.get("proofpack_fingerprint", "")) != fingerprint
        ):
            break
        records.append(record)
    if len(records) != len(cached):
        with open(output_path, "w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    if records:
        print(f"Resuming Qwen reranking from {len(records)}/{len(rows)} rows")
    if len(records) == len(rows):
        print(f"Qwen rerank proof pack already complete: {output_path}")
        return

    grounder = create_text_image_grounder(
        args.grounder_model,
        device="cuda",
        batch_size=args.grounder_batch_size,
        dtype="bfloat16",
        max_pixels=200704,
    )
    reranker = QwenVLWindowReranker(
        args.reranker_model,
        args.reranker_revision,
        args.reranker_batch_size,
        args.reranker_max_pixels,
    )

    mode = "a" if records else "w"
    with open(output_path, mode) as handle:
        for row_index, row in enumerate(rows[len(records) :], start=len(records)):
            video_path = os.path.join(video_folder, str(row["video_path"]))
            candidates, features, cache_hit = load_or_encode_grounder_features(
                video_path,
                args.candidate_frames,
                grounder,
                cache_dir,
            )
            program = compile_temporal_program_v2(row["question"])
            queries = build_token_safe_retrieval_queries(row, program)
            component_scores = {
                query.label: grounder.score_embeddings(query.text, features)
                for query in queries
            }
            pivot_query = (
                "Find the temporal reference event in the video.\n"
                f"Reference event: {program.pivot or row['question']}"
            )
            pivot_siglip_scores = grounder.score_embeddings(pivot_query, features)
            shortlist = build_siglip_shortlist(
                candidates,
                component_scores,
                pivot_siglip_scores,
                args.shortlist_windows,
                args.nms_seconds,
            )

            window_positions = [
                list(
                    range(
                        max(0, center - args.window_radius),
                        min(len(candidates), center + args.window_radius + 1),
                    )
                )
                for center in shortlist
            ]
            needed_indices = sorted(
                {
                    candidates[position].index
                    for positions in window_positions
                    for position in positions
                }
            )
            images = extract_frames_by_indices(video_path, needed_indices)
            if len(images) != len(needed_indices):
                raise RuntimeError(
                    f"extracted {len(images)}/{len(needed_indices)} reranker frames"
                )
            image_by_index = dict(zip(needed_indices, images))
            windows = [
                [image_by_index[candidates[position].index] for position in positions]
                for positions in window_positions
            ]
            timestamps = [
                [candidates[position].timestamp for position in positions]
                for positions in window_positions
            ]

            options = parse_mcq_options(row["mcq_options"])
            option_scores: dict[str, list[float]] = {}
            for letter, option in sorted(options.items()):
                query = f"Question: {row['question']}\nCandidate answer: {option}"
                option_scores[f"option_{letter}"] = reranker.score(
                    query, windows, timestamps
                )
            pivot_scores = None
            if program.operator != "GLOBAL":
                pivot_scores = reranker.score(
                    f"Temporal reference event: {program.pivot or row['question']}",
                    windows,
                    timestamps,
                )
            selected_centers, rerank_meta = select_reranked_centers(
                shortlist,
                candidates,
                option_scores,
                pivot_scores,
                program,
                args.mode,
                args.final_windows,
                args.contrast_weight,
                args.nms_seconds,
            )
            total_video_frames, fps = _video_metadata(video_path)
            final_indices, retrieval_indices, global_indices = build_final_frame_indices(
                candidates,
                selected_centers,
                total_video_frames,
                args.window_radius,
                args.global_frames,
                args.final_frames,
            )
            center_score = {
                center: max(
                    rerank_meta["adjusted_scores"][label][str(center)]
                    for label in rerank_meta["adjusted_scores"]
                )
                for center in selected_centers
            }
            selected = []
            for frame_index in final_indices:
                nearest = min(
                    selected_centers,
                    key=lambda center: abs(candidates[center].index - frame_index),
                )
                is_retrieval = frame_index in retrieval_indices
                selected.append(
                    {
                        "frame_index": frame_index,
                        "timestamp": round(frame_index / fps, 3),
                        "score": round(center_score[nearest], 6) if is_retrieval else None,
                        "source": "qwen_reranked_window" if is_retrieval else "endpoint_global",
                    }
                )
            record = {
                "index": row_index,
                "sample_key": sample_key(row),
                "video_path": row.get("video_path", ""),
                "strategy": STRATEGY,
                "candidate_frames": len(candidates),
                "selected_frames": len(selected),
                "proofpack_fingerprint": fingerprint,
                "feature_cache_hit": cache_hit,
                "grounder_model": args.grounder_model,
                "reranker_model": args.reranker_model,
                "reranker_revision": args.reranker_revision,
                "retrieval_query_mode": "token_safe_balanced_then_cross_encoder",
                "queries": [
                    {"label": query.label, "hash": query_hash(query.text)}
                    for query in queries
                ],
                "selection_meta": {
                    **rerank_meta,
                    "shortlist_centers": shortlist,
                    "selected_centers": selected_centers,
                    "global_frame_indices": global_indices,
                    "retrieval_frame_indices": sorted(retrieval_indices),
                    "temporal_program": {
                        "operator": program.operator,
                        "pivot": program.pivot,
                        "direction": program.direction,
                        "target": program.target,
                    },
                    "route": f"siglip2_recall_qwen_rerank_{args.mode}",
                },
                "selected": selected,
            }
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"Qwen rerank progress: {row_index + 1}/{len(rows)}")

    del reranker
    del grounder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Qwen rerank proof pack complete: {len(records)} rows -> {output_path}")


if __name__ == "__main__":
    main()
