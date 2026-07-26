#!/usr/bin/env python3
"""Generate LongQA predictions from structured temporal proof packs.

This experimental runner leaves the established uniform and grounded baselines
untouched. It reuses cached text-image candidate embeddings and supports:

* eventlet_hybrid: global anchors plus local triplets around relevant events,
* option_contrastive: balanced, discriminative eventlets for every option,
* temporal_pivot: pivot, directional target, bridge, and global evidence,
* qca: dynamic segment budgets balancing relevance and content deviation,
* multi_event: separate event-clause retrieval with chronological bridges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from longqa_utils import (
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    parse_mcq_options,
    query_hash,
    sample_key,
)
from run_generate_longqa_grounded import (
    CandidateFrame,
    GroundingQuery,
    SelectedFrame,
    TextImageGrounder,
    _run_eval,
    _uniform_positions,
    extract_frames_by_indices,
    load_jsonl,
    load_jsonl_if_exists,
    load_or_encode_grounder_features,
)

logger = logging.getLogger(__name__)

STRATEGIES = (
    "adaq",
    "eventlet_hybrid",
    "focus",
    "mixed_resolution",
    "option_contrastive",
    "temporal_pivot",
    "operator_router",
    "qca",
    "qca_router",
    "multi_event",
    "multi_event_router",
)
PROOFPACK_SCHEMA = 2


@dataclass(frozen=True)
class TemporalProgram:
    operator: str
    pivot: str
    direction: str
    target: str


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def compile_temporal_program(question: object) -> TemporalProgram:
    """Compile explicit temporal wording into a deterministic retrieval policy."""
    text = " ".join(str(question).split())
    leading = re.match(r"(?i)^(after|before)\s+(.+?),\s*(.+)$", text)
    if leading:
        operator = leading.group(1).upper()
        return TemporalProgram(
            operator=operator,
            pivot=leading.group(2).strip(),
            direction="forward" if operator == "AFTER" else "backward",
            target=leading.group(3).rstrip(" ?"),
        )

    embedded = re.search(r"(?i)\b(after|before)\s+(.+?)(?:\?|$)", text)
    if embedded:
        operator = embedded.group(1).upper()
        return TemporalProgram(
            operator=operator,
            pivot=embedded.group(2).strip(" ,.?"),
            direction="forward" if operator == "AFTER" else "backward",
            target=text.rstrip(" ?"),
        )

    if re.search(r"(?i)\b(first|earliest|at the start|beginning)\b", text):
        # Most LongQA "first" questions compare multiple occurrences. Preserve
        # all relevant eventlets so Qwen can establish their order.
        return TemporalProgram("FIRST", text, "multi_event", text)
    if re.search(r"(?i)\b(last|latest|finally|at the end|end of)\b", text):
        return TemporalProgram("LAST", text, "latest", text)
    if re.search(r"(?i)\b(chang(?:e|ed)|different|compared|between)\b", text):
        return TemporalProgram("STATE_CHANGE", text, "bidirectional", text)
    return TemporalProgram("GLOBAL", "", "global", text)


def _question_options_query(row: dict[str, Any]) -> str:
    return (
        "Find video evidence needed to answer this multiple-choice question.\n"
        f"Question: {row['question']}\nOptions:\n{row['mcq_options']}"
    )


def build_option_hypotheses(row: dict[str, Any]) -> list[GroundingQuery]:
    options = parse_mcq_options(row["mcq_options"])
    return [
        GroundingQuery(
            f"option_{letter}",
            "Find visual and temporal evidence that specifically supports this "
            "candidate answer rather than the alternatives.\n"
            f"Question: {row['question']}\n"
            f"Hypothesis {letter}: {option}",
        )
        for letter, option in sorted(options.items())
    ]


def build_multi_event_queries(row: dict[str, Any], max_events: int = 3) -> list[GroundingQuery]:
    """Extract distinct event clauses without asking an LLM to rewrite the question."""
    text = " ".join(str(row["question"]).split())
    fragments = re.split(
        r"(?i)(?:[.;,?]|\b(?:and then|then|later|previously|afterwards)\b)", text
    )
    queries: list[GroundingQuery] = []
    seen: set[str] = set()
    for fragment in fragments:
        fragment = fragment.strip(" ,.?-")
        fragment = re.sub(r"(?i)^(?:what|which|where|when|how)\s+", "", fragment)
        normalized = fragment.lower()
        if len(fragment.split()) < 3 or normalized in seen:
            continue
        seen.add(normalized)
        queries.append(
            GroundingQuery(
                f"event_{len(queries) + 1}",
                "Find the moment in the video corresponding to this event clause.\n"
                f"Event: {fragment}",
            )
        )
        if len(queries) >= max_events:
            break
    if len(queries) < 2:
        return [
            GroundingQuery(
                "event_1",
                "Find the principal event described by this question.\n"
                f"Question: {text}",
            )
        ]
    return queries


def _rank_with_temporal_nms(
    scores: list[float],
    candidates: list[CandidateFrame],
    count: int,
    nms_seconds: float,
    allowed: set[int] | None = None,
) -> list[int]:
    ranked = sorted(
        (idx for idx in range(len(scores)) if allowed is None or idx in allowed),
        key=lambda idx: scores[idx],
        reverse=True,
    )
    selected: list[int] = []
    for idx in ranked:
        if all(
            abs(candidates[idx].timestamp - candidates[other].timestamp) >= nms_seconds
            for other in selected
        ):
            selected.append(idx)
            if len(selected) >= count:
                break
    if len(selected) < count:
        for idx in ranked:
            if idx not in selected:
                selected.append(idx)
                if len(selected) >= count:
                    break
    return selected


def _boundary_scores(image_features: np.ndarray) -> list[float]:
    if len(image_features) == 0:
        return []
    scores = [0.0]
    for idx in range(1, len(image_features)):
        scores.append(float(1.0 - np.dot(image_features[idx], image_features[idx - 1])))
    return scores


def baseline_uniform_indices(total_frames: int, frame_count: int) -> list[int]:
    """Match model.extract_frames() for one full-video interval."""
    if total_frames <= 0 or frame_count <= 0:
        return []
    end_frame = total_frames - 1
    count = min(frame_count, total_frames)
    step = end_frame / count
    return sorted({int(idx * step) for idx in range(count)})


def select_baseline_uniform_frames(
    video_path: str, frame_count: int
) -> tuple[list[SelectedFrame], dict[str, Any]]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return [], {"uniform_frames": 0}
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if fps <= 0 or total_frames <= 0:
        return [], {"uniform_frames": 0}
    selected = [
        SelectedFrame(
            CandidateFrame(index=index, timestamp=index / fps, image=None),
            None,
            "uniform_global",
        )
        for index in baseline_uniform_indices(total_frames, frame_count)
    ]
    return selected, {"uniform_frames": len(selected)}


def _add_frame(
    selected: dict[int, SelectedFrame],
    candidates: list[CandidateFrame],
    candidate_pos: int,
    score: float | None,
    source: str,
    priority: int,
    priorities: dict[int, int],
) -> None:
    if candidate_pos < 0 or candidate_pos >= len(candidates):
        return
    source_index = candidates[candidate_pos].index
    previous_priority = priorities.get(source_index, 10_000)
    if source_index not in selected or priority < previous_priority:
        selected[source_index] = SelectedFrame(
            candidate=candidates[candidate_pos], score=score, source=source
        )
        priorities[source_index] = priority


def _add_eventlet(
    selected: dict[int, SelectedFrame],
    priorities: dict[int, int],
    candidates: list[CandidateFrame],
    center: int,
    score: float,
    radius: int,
    source: str,
    priority: int,
) -> None:
    for idx in range(center - radius, center + radius + 1):
        frame_source = source if idx == center else f"{source}_context"
        _add_frame(
            selected,
            candidates,
            idx,
            score,
            frame_source,
            priority if idx == center else priority + 1,
            priorities,
        )


def _finalize_selection(
    selected: dict[int, SelectedFrame],
    priorities: dict[int, int],
    final_max_frames: int,
) -> list[SelectedFrame]:
    values = list(selected.values())
    if len(values) > final_max_frames:
        values = sorted(
            values,
            key=lambda frame: (
                priorities.get(frame.candidate.index, 10_000),
                -(frame.score if frame.score is not None else -1e9),
                frame.candidate.timestamp,
            ),
        )[:final_max_frames]
    return sorted(values, key=lambda frame: frame.candidate.timestamp)


def _fill_boundaries(
    selected: dict[int, SelectedFrame],
    priorities: dict[int, int],
    candidates: list[CandidateFrame],
    boundary_scores: list[float],
    target_count: int,
) -> None:
    for idx in sorted(range(len(boundary_scores)), key=boundary_scores.__getitem__, reverse=True):
        if len(selected) >= target_count:
            break
        _add_frame(
            selected,
            candidates,
            idx,
            boundary_scores[idx],
            "semantic_boundary",
            5,
            priorities,
        )


def _fill_uniform_coverage(
    selected: dict[int, SelectedFrame],
    priorities: dict[int, int],
    candidates: list[CandidateFrame],
    target_count: int,
) -> None:
    """Greedily fill the largest uncovered temporal gaps."""
    source_to_position = {frame.index: idx for idx, frame in enumerate(candidates)}
    selected_positions = {
        source_to_position[source_index]
        for source_index in selected
        if source_index in source_to_position
    }
    while len(selected) < target_count and len(selected_positions) < len(candidates):
        remaining = [idx for idx in range(len(candidates)) if idx not in selected_positions]
        if not selected_positions:
            next_position = len(candidates) // 2
        else:
            next_position = max(
                remaining,
                key=lambda idx: (
                    min(abs(idx - chosen) for chosen in selected_positions),
                    -idx,
                ),
            )
        _add_frame(
            selected,
            candidates,
            next_position,
            None,
            "coverage_fill",
            4,
            priorities,
        )
        selected_positions.add(next_position)


def select_eventlet_hybrid(
    candidates: list[CandidateFrame],
    relevance_scores: list[float],
    image_features: np.ndarray,
    event_centers: int,
    eventlet_radius: int,
    anchor_k: int,
    boundary_k: int,
    final_max_frames: int,
    temporal_nms_seconds: float,
    fill_mode: str = "semantic_boundary",
) -> tuple[list[SelectedFrame], dict[str, Any]]:
    selected: dict[int, SelectedFrame] = {}
    priorities: dict[int, int] = {}
    centers = _rank_with_temporal_nms(
        relevance_scores,
        candidates,
        event_centers,
        temporal_nms_seconds,
    )
    for center in centers:
        _add_eventlet(
            selected,
            priorities,
            candidates,
            center,
            relevance_scores[center],
            eventlet_radius,
            "event_center",
            0,
        )
    for idx in _uniform_positions(len(candidates), anchor_k):
        _add_frame(selected, candidates, idx, relevance_scores[idx], "anchor", 3, priorities)

    boundaries = _boundary_scores(image_features)
    boundary_ranked = sorted(range(len(boundaries)), key=boundaries.__getitem__, reverse=True)
    added_boundaries = 0
    for idx in boundary_ranked:
        if added_boundaries >= boundary_k:
            break
        before = len(selected)
        _add_frame(
            selected, candidates, idx, boundaries[idx], "semantic_boundary", 4, priorities
        )
        added_boundaries += int(len(selected) > before)
    if fill_mode == "uniform_coverage":
        _fill_uniform_coverage(selected, priorities, candidates, final_max_frames)
    else:
        _fill_boundaries(selected, priorities, candidates, boundaries, final_max_frames)
    return _finalize_selection(selected, priorities, final_max_frames), {
        "event_centers": [candidates[idx].index for idx in centers],
        "eventlet_radius": eventlet_radius,
        "boundary_frames": added_boundaries,
        "fill_mode": fill_mode,
    }


def _zscore(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    std = float(array.std())
    if std < 1e-6:
        return np.zeros_like(array)
    return (array - float(array.mean())) / std


def _minmax(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    span = float(array.max() - array.min())
    if span < 1e-12:
        return np.zeros_like(array)
    return (array - float(array.min())) / span


def select_adaq_pack(
    candidates: list[CandidateFrame],
    relevance_scores: list[float],
    budget: int,
    var_scale: float,
    p_threshold: float,
    rng: np.random.Generator,
) -> tuple[list[SelectedFrame], dict[str, Any]]:
    """AdaQ's official variance-scaled top-p probabilistic sampler."""
    scores = np.asarray(relevance_scores, dtype=np.float64)
    if not 0 < budget <= len(scores):
        raise ValueError("AdaQ budget must be between 1 and the candidate count")
    if var_scale < 0 or not 0 < p_threshold <= 1:
        raise ValueError("AdaQ expects var_scale >= 0 and 0 < p_threshold <= 1")

    variance = float(np.var(_minmax(scores)))
    tau = max(variance * var_scale, 0.01)
    shifted = scores / tau if var_scale else scores
    shifted -= shifted.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()

    ranked = np.argsort(-probabilities)
    cumulative = np.cumsum(probabilities[ranked])
    keep_count = int(np.searchsorted(cumulative, p_threshold, side="left")) + 1
    filtered = ranked[:keep_count]
    removed = np.setdiff1d(np.arange(len(scores)), filtered)
    if len(filtered) < budget:
        need = min(budget - len(filtered), len(removed))
        supplement_prob = probabilities[removed]
        supplement_prob /= supplement_prob.sum()
        supplement = rng.choice(removed, size=need, replace=False, p=supplement_prob)
        chosen = np.concatenate([filtered, supplement])
    elif len(filtered) > budget:
        filtered_prob = probabilities[filtered]
        filtered_prob /= filtered_prob.sum()
        chosen = rng.choice(filtered, size=budget, replace=False, p=filtered_prob)
    else:
        chosen = filtered

    selected = [
        SelectedFrame(candidates[int(index)], float(scores[index]), "adaq")
        for index in sorted(int(index) for index in chosen)
    ]
    return selected, {
        "adaq_variance": variance,
        "adaq_tau": tau,
        "adaq_top_p_candidates": len(filtered),
        "adaq_var_scale": var_scale,
        "adaq_p_threshold": p_threshold,
    }


def select_focus_pack(
    candidates: list[CandidateFrame],
    relevance_scores: list[float],
    budget: int,
    num_arms: int,
    zoom_ratio: float,
    extra_samples_per_arm: int,
    top_ratio: float,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[list[SelectedFrame], dict[str, Any]]:
    """Apply FOCUS's Bernstein-UCB allocation to cached candidate scores."""
    scores = np.asarray(relevance_scores, dtype=np.float64)
    if not 0 < budget <= len(scores):
        raise ValueError("FOCUS budget must be between 1 and the candidate count")
    arms = [part for part in np.array_split(np.arange(len(scores)), num_arms) if len(part)]
    sampled_by_arm: list[np.ndarray] = []
    arm_stats: list[dict[str, float | int]] = []
    for arm_id, arm in enumerate(arms):
        center = int(arm[len(arm) // 2])
        available = arm[arm != center]
        extra_count = min(extra_samples_per_arm, len(available))
        extras = (
            rng.choice(available, size=extra_count, replace=False)
            if extra_count
            else np.asarray([], dtype=int)
        )
        sampled = np.concatenate([[center], extras]).astype(int)
        sampled_by_arm.append(sampled)
        values = scores[sampled]
        arm_stats.append(
            {
                "arm": arm_id,
                "start": int(arm[0]),
                "end": int(arm[-1]),
                "samples": len(sampled),
                "mean": float(values.mean()),
                "variance": float(values.var()) if len(values) > 1 else 0.0,
            }
        )

    total_samples = sum(int(item["samples"]) for item in arm_stats)
    for item in arm_stats:
        count = int(item["samples"])
        variance = max(float(item["variance"]), 1e-6)
        item["focus_score"] = float(item["mean"]) + math.sqrt(
            2 * math.log(max(total_samples, 2)) * variance / count
        ) + 3 * math.log(max(total_samples, 2)) / count

    selected_arm_count = max(4, int(math.ceil(len(arms) * zoom_ratio)))
    selected_arm_count = min(selected_arm_count, len(arms))
    selected_arm_ids = [
        int(item["arm"])
        for item in sorted(arm_stats, key=lambda item: float(item["focus_score"]), reverse=True)[
            :selected_arm_count
        ]
    ]
    computed = sorted(
        set(
            int(index)
            for arm_id in selected_arm_ids
            for index in np.concatenate([sampled_by_arm[arm_id], arms[arm_id]])
        )
    )
    top_count = min(budget, max(1, int(round(top_ratio * min(budget, len(computed))))))
    chosen = sorted(computed, key=lambda index: scores[index], reverse=True)[:top_count]
    chosen_set = set(chosen)

    remaining = budget - len(chosen)
    base, remainder = divmod(remaining, selected_arm_count)
    for rank, arm_id in enumerate(selected_arm_ids):
        need = base + int(rank < remainder)
        available = [int(index) for index in arms[arm_id] if int(index) not in chosen_set]
        if not available or need <= 0:
            continue
        values = _minmax(scores[available])
        logits = values / max(temperature, 1e-6)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        take = min(need, len(available))
        additions = rng.choice(available, size=take, replace=False, p=probabilities)
        chosen.extend(int(index) for index in additions)
        chosen_set.update(int(index) for index in additions)

    if len(chosen) < budget:
        for index in np.argsort(-scores):
            if int(index) not in chosen_set:
                chosen.append(int(index))
                chosen_set.add(int(index))
            if len(chosen) == budget:
                break
    selected = [
        SelectedFrame(candidates[index], float(scores[index]), "focus")
        for index in sorted(chosen[:budget])
    ]
    return selected, {
        "focus_arms": arm_stats,
        "focus_selected_arms": selected_arm_ids,
        "focus_zoom_ratio": zoom_ratio,
        "focus_top_ratio": top_ratio,
    }


def select_mixed_resolution_pack(
    candidates: list[CandidateFrame],
    relevance_scores: list[float],
    high_frames: int,
    medium_frames: int,
    low_frames: int,
    high_pixels: int,
    medium_pixels: int,
    low_pixels: int,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[list[SelectedFrame], dict[str, Any]]:
    """Q-Frame-style Gumbel sampling with relevance-ranked pixel budgets."""
    scores = np.asarray(relevance_scores, dtype=np.float64)
    budget = high_frames + medium_frames + low_frames
    if not 0 < budget <= len(scores):
        raise ValueError("mixed-resolution budget exceeds candidate count")
    logits = scores / max(temperature, 1e-6)
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    gumbel = rng.gumbel(size=len(scores))
    chosen = np.argsort(-(np.log(np.maximum(probabilities, 1e-300)) + gumbel))[:budget]
    relevance_ranked = sorted((int(index) for index in chosen), key=lambda index: scores[index], reverse=True)
    assignments: dict[int, tuple[str, int]] = {}
    cut_high = high_frames
    cut_medium = high_frames + medium_frames
    for rank, index in enumerate(relevance_ranked):
        if rank < cut_high:
            assignments[index] = ("qframe_high", high_pixels)
        elif rank < cut_medium:
            assignments[index] = ("qframe_medium", medium_pixels)
        else:
            assignments[index] = ("qframe_low", low_pixels)
    selected = [
        SelectedFrame(candidates[index], float(scores[index]), assignments[index][0])
        for index in sorted(assignments)
    ]
    return selected, {
        "mixed_resolution_counts": {
            "high": high_frames,
            "medium": medium_frames,
            "low": low_frames,
        },
        "mixed_resolution_pixels": {
            "high": high_pixels,
            "medium": medium_pixels,
            "low": low_pixels,
        },
        "frame_max_pixels": {
            str(candidates[index].index): pixels
            for index, (_source, pixels) in assignments.items()
        },
        "qframe_temperature": temperature,
    }


def resize_frame_to_max_pixels(frame: object, max_pixels: int) -> object:
    """Downscale a PIL frame to a per-frame area budget without upscaling."""
    width, height = frame.size
    area = width * height
    if area <= max_pixels:
        return frame
    scale = math.sqrt(max_pixels / area)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    from PIL import Image

    return frame.resize(target, Image.Resampling.LANCZOS)


def select_option_contrastive_eventlets(
    candidates: list[CandidateFrame],
    option_scores: dict[str, list[float]],
    image_features: np.ndarray,
    centers_per_option: int,
    eventlet_radius: int,
    anchor_k: int,
    final_max_frames: int,
    temporal_nms_seconds: float,
) -> tuple[list[SelectedFrame], dict[str, Any]]:
    if not option_scores:
        return [], {"option_centers": {}}
    labels = sorted(option_scores)
    normalized = {label: _zscore(option_scores[label]) for label in labels}
    selected: dict[int, SelectedFrame] = {}
    priorities: dict[int, int] = {}
    option_centers: dict[str, list[int]] = {}
    for label in labels:
        others = [normalized[other] for other in labels if other != label]
        competing = np.max(np.stack(others), axis=0) if others else 0.0
        contrast = normalized[label] - competing
        contrast_scores = [float(value) for value in contrast]
        centers = _rank_with_temporal_nms(
            contrast_scores,
            candidates,
            centers_per_option,
            temporal_nms_seconds,
        )
        option_centers[label] = [candidates[idx].index for idx in centers]
        for center in centers:
            _add_eventlet(
                selected,
                priorities,
                candidates,
                center,
                contrast_scores[center],
                eventlet_radius,
                label,
                0,
            )
    merged_relevance = np.max(np.stack([normalized[label] for label in labels]), axis=0)
    for idx in _uniform_positions(len(candidates), anchor_k):
        _add_frame(
            selected,
            candidates,
            idx,
            float(merged_relevance[idx]),
            "anchor",
            3,
            priorities,
        )
    boundaries = _boundary_scores(image_features)
    _fill_boundaries(selected, priorities, candidates, boundaries, final_max_frames)
    return _finalize_selection(selected, priorities, final_max_frames), {
        "option_centers": option_centers,
        "centers_per_option": centers_per_option,
        "eventlet_radius": eventlet_radius,
    }


def select_qca_pack(
    candidates: list[CandidateFrame],
    relevance_scores: list[float],
    image_features: np.ndarray,
    budget: int,
    num_segments: int,
    alpha: float,
    beta: float,
    temperature: float,
    relevance_threshold: float,
) -> tuple[list[SelectedFrame], dict[str, Any]]:
    """Adapt the teammate QCA pilot to cached proof-pack candidates."""
    features = np.asarray(image_features, dtype=np.float32)
    relevance = np.asarray(relevance_scores, dtype=np.float32)
    if features.ndim != 2 or relevance.shape != (len(features),):
        raise ValueError("expected features [frames, dim] and relevance [frames]")
    if not 0 < budget <= len(features):
        raise ValueError("QCA budget must be between 1 and the candidate count")
    if alpha < 0 or beta < 0 or alpha + beta <= 0:
        raise ValueError("QCA alpha and beta must have a positive sum")

    segments = [
        segment
        for segment in np.array_split(np.arange(len(features)), num_segments)
        if len(segment)
    ]
    global_mean = features.mean(axis=0)
    matching = np.asarray([relevance[segment].mean() for segment in segments])
    deviation = np.asarray(
        [
            np.square(features[segment].mean(axis=0) - global_mean).sum()
            + features[segment].var(axis=0).sum()
            for segment in segments
        ]
    )

    def softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - values.max()
        weights = np.exp(shifted)
        return weights / max(float(weights.sum()), 1e-12)

    scale = alpha + beta
    contribution = (alpha / scale) * softmax(matching) + (beta / scale) * softmax(
        deviation
    )
    weights = np.power(np.maximum(contribution, 1e-12), temperature)
    weights /= weights.sum()
    raw_quotas = weights * budget
    quotas = np.floor(raw_quotas).astype(int)
    remainder = budget - int(quotas.sum())
    for index in np.argsort(-(raw_quotas - quotas))[:remainder]:
        quotas[index] += 1

    selected_positions: list[int] = []
    segment_anchors: list[int | None] = []
    for segment, quota in zip(segments, quotas):
        if quota <= 0:
            segment_anchors.append(None)
            continue
        anchor = int(segment[np.argmax(relevance[segment])])
        segment_anchors.append(anchor)
        chosen = [anchor]
        threshold = relevance_threshold
        pool = segment[relevance[segment] >= threshold * relevance[anchor]]
        while len(pool) < quota and threshold > 0:
            threshold = max(0.0, threshold - 0.1)
            pool = segment[relevance[segment] >= threshold * relevance[anchor]]
        if len(pool) < quota:
            pool = segment
        while len(chosen) < quota:
            remaining = [int(index) for index in pool if int(index) not in chosen]
            if not remaining:
                break
            diversity = np.asarray(
                [
                    min(1.0 - float(np.dot(features[index], features[other])) for other in chosen)
                    for index in remaining
                ]
            )
            chosen.append(remaining[int(diversity.argmax())])
        selected_positions.extend(chosen)

    if len(selected_positions) < budget:
        for index in np.argsort(-relevance):
            position = int(index)
            if position not in selected_positions:
                selected_positions.append(position)
            if len(selected_positions) == budget:
                break

    anchor_set = {index for index in segment_anchors if index is not None}
    selected = [
        SelectedFrame(
            candidate=candidates[position],
            score=float(relevance[position]),
            source="qca_segment_anchor" if position in anchor_set else "qca_diverse",
        )
        for position in sorted(selected_positions[:budget])
    ]
    return selected, {
        "qca_quotas": quotas.tolist(),
        "qca_segment_matching": matching.tolist(),
        "qca_segment_deviation": deviation.tolist(),
        "qca_segment_anchors": [
            candidates[index].index if index is not None else None for index in segment_anchors
        ],
    }


def select_multi_event_pack(
    candidates: list[CandidateFrame],
    event_scores: dict[str, list[float]],
    anchor_scores: list[float],
    centers_per_event: int,
    eventlet_radius: int,
    anchor_k: int,
    bridge_k: int,
    final_max_frames: int,
    temporal_nms_seconds: float,
) -> tuple[list[SelectedFrame], dict[str, Any]]:
    selected: dict[int, SelectedFrame] = {}
    priorities: dict[int, int] = {}
    event_centers: dict[str, list[int]] = {}
    primary_centers: list[int] = []
    for label, scores in event_scores.items():
        centers = _rank_with_temporal_nms(
            scores, candidates, centers_per_event, temporal_nms_seconds
        )
        event_centers[label] = [candidates[index].index for index in centers]
        if centers:
            primary_centers.append(centers[0])
        for center in centers:
            _add_eventlet(
                selected,
                priorities,
                candidates,
                center,
                scores[center],
                eventlet_radius,
                label,
                0,
            )
    bridge_positions: list[int] = []
    ordered_centers = sorted(set(primary_centers))
    pair_count = max(len(ordered_centers) - 1, 1)
    per_pair = max(1, bridge_k // pair_count) if bridge_k else 0
    for start, end in zip(ordered_centers, ordered_centers[1:]):
        bridge_positions.extend(_bridge_positions(start, end, per_pair))
    for index in bridge_positions[:bridge_k]:
        _add_frame(selected, candidates, index, anchor_scores[index], "bridge", 2, priorities)
    for index in _uniform_positions(len(candidates), anchor_k):
        _add_frame(selected, candidates, index, anchor_scores[index], "anchor", 3, priorities)
    _fill_uniform_coverage(selected, priorities, candidates, final_max_frames)
    return _finalize_selection(selected, priorities, final_max_frames), {
        "event_centers": event_centers,
        "bridge_frames": min(len(bridge_positions), bridge_k),
        "fill_mode": "uniform_coverage",
    }


def _bridge_positions(start: int, end: int, count: int) -> list[int]:
    if count <= 0 or start == end:
        return []
    low, high = sorted((start, end))
    positions = _uniform_positions(high - low + 1, count + 2)[1:-1]
    return [low + position for position in positions]


def select_temporal_pivot_pack(
    candidates: list[CandidateFrame],
    pivot_scores: list[float],
    target_scores: list[float],
    image_features: np.ndarray,
    program: TemporalProgram,
    pivot_centers: int,
    target_centers: int,
    eventlet_radius: int,
    anchor_k: int,
    bridge_k: int,
    final_max_frames: int,
    temporal_nms_seconds: float,
    fill_mode: str = "semantic_boundary",
) -> tuple[list[SelectedFrame], dict[str, Any]]:
    if program.operator == "GLOBAL":
        return select_eventlet_hybrid(
            candidates,
            target_scores,
            image_features,
            target_centers,
            eventlet_radius,
            anchor_k,
            bridge_k,
            final_max_frames,
            temporal_nms_seconds,
            fill_mode,
        )

    selected: dict[int, SelectedFrame] = {}
    priorities: dict[int, int] = {}
    pivots = _rank_with_temporal_nms(
        pivot_scores,
        candidates,
        pivot_centers,
        temporal_nms_seconds,
    )
    primary_pivot = pivots[0] if pivots else len(candidates) // 2
    for center in pivots:
        _add_eventlet(
            selected,
            priorities,
            candidates,
            center,
            pivot_scores[center],
            eventlet_radius,
            "pivot",
            0,
        )

    allowed: set[int] | None = None
    if program.direction == "forward":
        allowed = set(range(min(primary_pivot + 1, len(candidates)), len(candidates)))
    elif program.direction == "backward":
        allowed = set(range(0, max(primary_pivot, 0)))
    elif program.direction == "earliest":
        relevant = _rank_with_temporal_nms(
            target_scores, candidates, max(target_centers * 4, target_centers), temporal_nms_seconds
        )
        relevant.sort()
        allowed = set(relevant[: max(target_centers * 2, target_centers)])
    elif program.direction == "latest":
        relevant = _rank_with_temporal_nms(
            target_scores, candidates, max(target_centers * 4, target_centers), temporal_nms_seconds
        )
        relevant.sort(reverse=True)
        allowed = set(relevant[: max(target_centers * 2, target_centers)])

    targets = _rank_with_temporal_nms(
        target_scores,
        candidates,
        target_centers,
        temporal_nms_seconds,
        allowed=allowed,
    )
    for center in targets:
        _add_eventlet(
            selected,
            priorities,
            candidates,
            center,
            target_scores[center],
            eventlet_radius,
            "directional_target",
            1,
        )

    bridge_positions: list[int] = []
    for target in targets[:2]:
        bridge_positions.extend(_bridge_positions(primary_pivot, target, bridge_k // 2))
    for idx in bridge_positions[:bridge_k]:
        _add_frame(selected, candidates, idx, target_scores[idx], "bridge", 2, priorities)
    for idx in _uniform_positions(len(candidates), anchor_k):
        _add_frame(selected, candidates, idx, target_scores[idx], "anchor", 3, priorities)
    if fill_mode == "uniform_coverage":
        _fill_uniform_coverage(selected, priorities, candidates, final_max_frames)
    else:
        boundaries = _boundary_scores(image_features)
        _fill_boundaries(selected, priorities, candidates, boundaries, final_max_frames)
    return _finalize_selection(selected, priorities, final_max_frames), {
        "temporal_program": {
            "operator": program.operator,
            "pivot": program.pivot,
            "direction": program.direction,
            "target": program.target,
        },
        "pivot_centers": [candidates[idx].index for idx in pivots],
        "target_centers": [candidates[idx].index for idx in targets],
        "bridge_frames": len(bridge_positions[:bridge_k]),
        "fill_mode": fill_mode,
    }


def build_structured_evidence_prompt(
    row: dict[str, Any], selected_meta: list[dict[str, Any]], program: dict[str, Any] | None
) -> str:
    role_groups: dict[str, list[str]] = {}
    for image_number, item in enumerate(selected_meta, start=1):
        source = str(item.get("source", "evidence"))
        role_groups.setdefault(source, []).append(
            f"{image_number}@{float(item.get('timestamp', 0.0)):.1f}s"
        )
    evidence_lines = [
        f"- {role}: images {', '.join(items)}" for role, items in sorted(role_groups.items())
    ]
    program_line = ""
    if program:
        program_line = (
            f"Temporal operator: {program.get('operator', 'GLOBAL')}; "
            f"direction: {program.get('direction', 'global')}; "
            f"pivot: {program.get('pivot') or 'none'}.\n"
        )
    instruction = (
        "The supplied images are in chronological order. Their evidence roles are:\n"
        + "\n".join(evidence_lines)
        + "\n"
        + program_line
        + "Internally evaluate every option against visible support and contradiction. "
        "For temporal options, verify that the evidence occurs on the required side "
        "of the pivot. Use the role labels only as retrieval hints; trust the images."
    )
    return instruction + "\n\n" + build_longqa_prompt(
        row["question"], row["mcq_options"], prompt_variant="baseline"
    )


def proofpack_fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "schema": PROOFPACK_SCHEMA,
        "strategy": args.strategy,
        "candidate_frames": args.candidate_frames,
        "grounder_model": args.grounder_model,
        "grounder_revision": args.grounder_revision,
        "retrieval_query_mode": "question_options",
        "event_centers": args.event_centers,
        "centers_per_option": args.centers_per_option,
        "pivot_centers": args.pivot_centers,
        "target_centers": args.target_centers,
        "eventlet_radius": args.eventlet_radius,
        "anchor_k": args.anchor_k,
        "boundary_k": args.boundary_k,
        "bridge_k": args.bridge_k,
        "final_max_frames": args.final_max_frames,
        "temporal_nms_seconds": args.temporal_nms_seconds,
        "fill_mode": args.fill_mode,
        "global_uniform_frames": args.global_uniform_frames,
        "multi_event_centers": args.multi_event_centers,
        "max_event_queries": args.max_event_queries,
        "qca_segments": args.qca_segments,
        "qca_alpha": args.qca_alpha,
        "qca_beta": args.qca_beta,
        "qca_temperature": args.qca_temperature,
        "qca_relevance_threshold": args.qca_relevance_threshold,
        "selection_seed": args.selection_seed,
        "adaq_var_scale": args.adaq_var_scale,
        "adaq_p_threshold": args.adaq_p_threshold,
        "focus_arms": args.focus_arms,
        "focus_zoom_ratio": args.focus_zoom_ratio,
        "focus_extra_samples": args.focus_extra_samples,
        "focus_top_ratio": args.focus_top_ratio,
        "focus_temperature": args.focus_temperature,
        "mixed_high_frames": args.mixed_high_frames,
        "mixed_medium_frames": args.mixed_medium_frames,
        "mixed_low_frames": args.mixed_low_frames,
        "mixed_high_pixels": args.mixed_high_pixels,
        "mixed_medium_pixels": args.mixed_medium_pixels,
        "mixed_low_pixels": args.mixed_low_pixels,
        "qframe_temperature": args.qframe_temperature,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def inference_fingerprint(args: argparse.Namespace, proofpack_hash: str) -> str:
    payload = {
        "proofpack": proofpack_hash,
        "structured_evidence": args.structured_evidence,
        "model_type": args.model_type,
        "llm_model": args.llm_model,
        "backend": args.backend,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def parse_args() -> argparse.Namespace:
    from model import MODEL_TYPES

    parser = argparse.ArgumentParser(description="LongQA temporal proof-pack experiments.")
    parser.add_argument(
        "--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--grounding-output", required=True)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-resume-grounding", action="store_true")
    parser.add_argument("--no-resume-predictions", action="store_true")

    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--candidate-frames", type=int, default=128)
    parser.add_argument("--event-centers", type=int, default=8)
    parser.add_argument("--centers-per-option", type=int, default=4)
    parser.add_argument("--pivot-centers", type=int, default=2)
    parser.add_argument("--target-centers", type=int, default=8)
    parser.add_argument("--eventlet-radius", type=int, default=1)
    parser.add_argument("--anchor-k", type=int, default=32)
    parser.add_argument("--boundary-k", type=int, default=8)
    parser.add_argument("--bridge-k", type=int, default=8)
    parser.add_argument("--global-uniform-frames", type=int, default=64)
    parser.add_argument("--multi-event-centers", type=int, default=2)
    parser.add_argument("--max-event-queries", type=int, default=3)
    parser.add_argument("--qca-segments", type=int, default=16)
    parser.add_argument("--qca-alpha", type=float, default=0.5)
    parser.add_argument("--qca-beta", type=float, default=0.5)
    parser.add_argument("--qca-temperature", type=float, default=0.5)
    parser.add_argument("--qca-relevance-threshold", type=float, default=0.7)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--adaq-var-scale", type=float, default=0.5)
    parser.add_argument("--adaq-p-threshold", type=float, default=0.95)
    parser.add_argument("--focus-arms", type=int, default=16)
    parser.add_argument("--focus-zoom-ratio", type=float, default=0.25)
    parser.add_argument("--focus-extra-samples", type=int, default=2)
    parser.add_argument("--focus-top-ratio", type=float, default=0.2)
    parser.add_argument("--focus-temperature", type=float, default=0.06)
    parser.add_argument("--mixed-high-frames", type=int, default=4)
    parser.add_argument("--mixed-medium-frames", type=int, default=8)
    parser.add_argument("--mixed-low-frames", type=int, default=32)
    parser.add_argument("--mixed-high-pixels", type=int, default=451584)
    parser.add_argument("--mixed-medium-pixels", type=int, default=200704)
    parser.add_argument("--mixed-low-pixels", type=int, default=50176)
    parser.add_argument("--qframe-temperature", type=float, default=0.1)
    parser.add_argument("--final-max-frames", type=int, default=64)
    parser.add_argument("--temporal-nms-seconds", type=float, default=10.0)
    parser.add_argument(
        "--fill-mode",
        choices=["semantic_boundary", "uniform_coverage"],
        default="semantic_boundary",
    )
    parser.add_argument("--structured-evidence", action="store_true")

    parser.add_argument(
        "--grounder-model", default="Qwen/Qwen3-VL-Embedding-8B"
    )
    parser.add_argument("--grounder-device", default="cuda")
    parser.add_argument("--grounder-batch-size", type=int, default=4)
    parser.add_argument("--grounder-dtype", default="bfloat16")
    parser.add_argument("--grounder-revision", default=None)
    parser.add_argument("--grounder-cache-dir", default=None)

    parser.add_argument("--model-type", default="qwen", choices=MODEL_TYPES)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--backend", default="vllm", choices=["hf", "vllm"])
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    import gc
    import time

    import torch

    from model import create_model, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    eval_output = _resolve_path(args.eval_output) if args.eval_output else None
    grounding_output = _resolve_path(args.grounding_output)
    cache_dir = _resolve_path(args.grounder_cache_dir) if args.grounder_cache_dir else None
    fingerprint = proofpack_fingerprint(args)
    generation_fingerprint = inference_fingerprint(args, fingerprint)

    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(grounding_output), exist_ok=True)
    print(
        f"Proof-pack config: strategy={args.strategy}, candidates={args.candidate_frames}, "
        f"final_frames={args.final_max_frames}, structured={args.structured_evidence}, "
        f"grounder={args.grounder_model}, cache={cache_dir or 'disabled'}, "
        f"fingerprint={fingerprint}"
    )
    start_time = time.time()

    cached = [] if args.no_resume_grounding else load_jsonl_if_exists(grounding_output)
    records: list[dict[str, Any]] = []
    for idx, meta in enumerate(cached[: len(rows)]):
        if (
            int(meta.get("index", -1)) != idx
            or str(meta.get("video_path", "")) != str(rows[idx].get("video_path", ""))
            or str(meta.get("proofpack_fingerprint", "")) != fingerprint
        ):
            break
        records.append(meta)
    if records:
        print(f"Resuming proof-pack selection from {len(records)}/{len(rows)} rows")

    if len(records) < len(rows):
        grounder = TextImageGrounder(
            args.grounder_model,
            device=args.grounder_device,
            batch_size=args.grounder_batch_size,
            dtype=args.grounder_dtype,
            revision=args.grounder_revision,
        )
        mode = "a" if records else "w"
        if len(records) != len(cached):
            with open(grounding_output, "w") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
        with open(grounding_output, mode) as handle:
            for row_idx, row in enumerate(rows[len(records) :], start=len(records)):
                video_path = os.path.join(video_folder, str(row["video_path"]))
                candidates, image_features, cache_hit = load_or_encode_grounder_features(
                    video_path, args.candidate_frames, grounder, cache_dir
                )
                base_query = _question_options_query(row)
                base_scores = grounder.score_embeddings(base_query, image_features)
                queries = [{"label": "question_options", "hash": query_hash(base_query)}]
                program: TemporalProgram | None = None
                rng = np.random.default_rng(args.selection_seed + row_idx)

                if args.strategy == "adaq":
                    selected, selection_meta = select_adaq_pack(
                        candidates,
                        base_scores,
                        args.final_max_frames,
                        args.adaq_var_scale,
                        args.adaq_p_threshold,
                        rng,
                    )
                elif args.strategy == "focus":
                    selected, selection_meta = select_focus_pack(
                        candidates,
                        base_scores,
                        args.final_max_frames,
                        args.focus_arms,
                        args.focus_zoom_ratio,
                        args.focus_extra_samples,
                        args.focus_top_ratio,
                        args.focus_temperature,
                        rng,
                    )
                elif args.strategy == "mixed_resolution":
                    selected, selection_meta = select_mixed_resolution_pack(
                        candidates,
                        base_scores,
                        args.mixed_high_frames,
                        args.mixed_medium_frames,
                        args.mixed_low_frames,
                        args.mixed_high_pixels,
                        args.mixed_medium_pixels,
                        args.mixed_low_pixels,
                        args.qframe_temperature,
                        rng,
                    )
                elif args.strategy == "eventlet_hybrid":
                    selected, selection_meta = select_eventlet_hybrid(
                        candidates,
                        base_scores,
                        image_features,
                        args.event_centers,
                        args.eventlet_radius,
                        args.anchor_k,
                        args.boundary_k,
                        args.final_max_frames,
                        args.temporal_nms_seconds,
                        args.fill_mode,
                    )
                elif args.strategy == "option_contrastive":
                    option_queries = build_option_hypotheses(row)
                    option_scores = {}
                    queries = []
                    for query in option_queries:
                        option_scores[query.label] = grounder.score_embeddings(
                            query.text, image_features
                        )
                        queries.append({"label": query.label, "hash": query_hash(query.text)})
                    selected, selection_meta = select_option_contrastive_eventlets(
                        candidates,
                        option_scores,
                        image_features,
                        args.centers_per_option,
                        args.eventlet_radius,
                        args.anchor_k,
                        args.final_max_frames,
                        args.temporal_nms_seconds,
                    )
                elif args.strategy == "qca":
                    selected, selection_meta = select_qca_pack(
                        candidates,
                        base_scores,
                        image_features,
                        args.final_max_frames,
                        args.qca_segments,
                        args.qca_alpha,
                        args.qca_beta,
                        args.qca_temperature,
                        args.qca_relevance_threshold,
                    )
                    selection_meta["route"] = "qca"
                elif args.strategy in {"multi_event", "multi_event_router"}:
                    program = compile_temporal_program(row["question"])
                    use_multi_event = args.strategy == "multi_event" or program.operator in {
                        "FIRST",
                        "STATE_CHANGE",
                    }
                    if use_multi_event:
                        event_queries = build_multi_event_queries(
                            row, args.max_event_queries
                        )
                        event_scores = {}
                        queries = []
                        for query in event_queries:
                            event_scores[query.label] = grounder.score_embeddings(
                                query.text, image_features
                            )
                            queries.append(
                                {"label": query.label, "hash": query_hash(query.text)}
                            )
                        selected, selection_meta = select_multi_event_pack(
                            candidates,
                            event_scores,
                            base_scores,
                            args.multi_event_centers,
                            args.eventlet_radius,
                            args.anchor_k,
                            args.bridge_k,
                            args.final_max_frames,
                            args.temporal_nms_seconds,
                        )
                        selection_meta["route"] = "multi_event"
                    else:
                        pivot_query = (
                            "Find the temporal pivot event in the video.\n"
                            f"Pivot event: {program.pivot or row['question']}"
                        )
                        pivot_scores = grounder.score_embeddings(
                            pivot_query, image_features
                        )
                        queries = [
                            {"label": "pivot", "hash": query_hash(pivot_query)},
                            {"label": "target", "hash": query_hash(base_query)},
                        ]
                        selected, selection_meta = select_temporal_pivot_pack(
                            candidates,
                            pivot_scores,
                            base_scores,
                            image_features,
                            program,
                            args.pivot_centers,
                            args.target_centers,
                            args.eventlet_radius,
                            args.anchor_k,
                            args.bridge_k,
                            args.final_max_frames,
                            args.temporal_nms_seconds,
                            args.fill_mode,
                        )
                        selection_meta["route"] = "temporal_pivot"
                    selection_meta["temporal_program"] = {
                        "operator": program.operator,
                        "pivot": program.pivot,
                        "direction": program.direction,
                        "target": program.target,
                    }
                else:
                    program = compile_temporal_program(row["question"])
                    pivot_query = (
                        "Find the temporal pivot event in the video.\n"
                        f"Pivot event: {program.pivot or row['question']}"
                    )
                    pivot_scores = grounder.score_embeddings(pivot_query, image_features)
                    queries = [
                        {"label": "pivot", "hash": query_hash(pivot_query)},
                        {"label": "target", "hash": query_hash(base_query)},
                    ]
                    if args.strategy == "operator_router" and program.operator == "GLOBAL":
                        selected, selection_meta = select_baseline_uniform_frames(
                            video_path, args.global_uniform_frames
                        )
                        selection_meta["temporal_program"] = {
                            "operator": program.operator,
                            "pivot": program.pivot,
                            "direction": program.direction,
                            "target": program.target,
                        }
                        selection_meta["route"] = "uniform_global"
                    elif args.strategy == "qca_router" and program.operator == "GLOBAL":
                        selected, selection_meta = select_qca_pack(
                            candidates,
                            base_scores,
                            image_features,
                            args.final_max_frames,
                            args.qca_segments,
                            args.qca_alpha,
                            args.qca_beta,
                            args.qca_temperature,
                            args.qca_relevance_threshold,
                        )
                        selection_meta["temporal_program"] = {
                            "operator": program.operator,
                            "pivot": program.pivot,
                            "direction": program.direction,
                            "target": program.target,
                        }
                        selection_meta["route"] = "qca_global"
                    else:
                        selected, selection_meta = select_temporal_pivot_pack(
                            candidates,
                            pivot_scores,
                            base_scores,
                            image_features,
                            program,
                            args.pivot_centers,
                            args.target_centers,
                            args.eventlet_radius,
                            args.anchor_k,
                            args.bridge_k,
                            args.final_max_frames,
                            args.temporal_nms_seconds,
                            args.fill_mode,
                        )
                        selection_meta["route"] = "temporal_pivot"

                record = {
                    "index": row_idx,
                    "sample_key": sample_key(row),
                    "video_path": row.get("video_path", ""),
                    "strategy": args.strategy,
                    "candidate_frames": len(candidates),
                    "selected_frames": len(selected),
                    "proofpack_fingerprint": fingerprint,
                    "feature_cache_hit": cache_hit,
                    "grounder_model": args.grounder_model,
                    "queries": queries,
                    "selection_meta": selection_meta,
                    "selected": [
                        {
                            "frame_index": frame.candidate.index,
                            "timestamp": round(frame.candidate.timestamp, 3),
                            "score": round(frame.score, 6) if frame.score is not None else None,
                            "source": frame.source,
                        }
                        for frame in selected
                    ],
                }
                records.append(record)
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(f"  Selection progress: {row_idx + 1}/{len(rows)}")
        del grounder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("Proof-pack cache complete; skipping grounder model load")

    existing = [] if args.no_resume_predictions else load_jsonl_if_exists(output_path)
    pred_start = 0
    for idx, pred in enumerate(existing[: len(rows)]):
        if str(pred.get("video_path", "")) != str(rows[idx].get("video_path", "")):
            break
        if str(pred.get("proofpack_inference_fingerprint", "")) != generation_fingerprint:
            break
        if not str(pred.get("mcq_answer", "")).strip():
            break
        pred_start += 1
    if pred_start:
        print(f"Resuming generation from {pred_start}/{len(rows)} rows")

    model = create_model(
        args.model_type,
        args.llm_model,
        backend=args.backend,
        tp_size=args.tp,
        concurrency=args.concurrency,
        max_frames=args.final_max_frames,
    )
    reset_prompt_token_stats()
    mode = "a" if pred_start else "w"
    prompt_variant = "structured_evidence" if args.structured_evidence else "baseline"
    with model, open(output_path, mode) as handle:
        for row_idx, (row, record) in enumerate(
            zip(rows[pred_start:], records[pred_start:]), start=pred_start
        ):
            video_path = os.path.join(video_folder, str(row["video_path"]))
            selected_meta = record["selected"]
            frame_indices = [int(item["frame_index"]) for item in selected_meta]
            frames = extract_frames_by_indices(video_path, frame_indices)
            frame_pixel_budgets = record.get("selection_meta", {}).get(
                "frame_max_pixels", {}
            )
            if frame_pixel_budgets:
                frames = [
                    resize_frame_to_max_pixels(
                        frame,
                        int(frame_pixel_budgets.get(str(frame_index), args.mixed_high_pixels)),
                    )
                    for frame, frame_index in zip(frames, frame_indices)
                ]
            temporal_program = record.get("selection_meta", {}).get("temporal_program")
            prompt = (
                build_structured_evidence_prompt(row, selected_meta, temporal_program)
                if args.structured_evidence
                else build_longqa_prompt(
                    row["question"], row["mcq_options"], prompt_variant="baseline"
                )
            )
            response = model.generate(
                frames, [{"role": "user", "content": prompt}], max_new_tokens=16
            )
            pred = build_prediction_row(row, response, prompt_variant=prompt_variant)
            pred["proofpack_strategy"] = args.strategy
            pred["proofpack_fingerprint"] = fingerprint
            pred["proofpack_inference_fingerprint"] = generation_fingerprint
            handle.write(json.dumps(pred) + "\n")
            handle.flush()
            print(f"  Generation progress: {row_idx + 1}/{len(rows)}")

    print(f"Predictions written to {output_path}")
    print(f"Proof-pack metadata written to {grounding_output}")
    print(f"Runtime seconds: {time.time() - start_time:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
