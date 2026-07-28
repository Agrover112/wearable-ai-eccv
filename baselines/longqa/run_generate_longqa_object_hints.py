#!/usr/bin/env python3
"""Evaluate object-aware visual hints on cached LongQA proof packs.

The runner is intentionally separate from the established uniform and temporal-
pivot baselines. It reuses a completed proof pack, caches open-vocabulary object
detections, unloads the detector, and then evaluates one of six visual-evidence
policies with Qwen.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from longqa_utils import (
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    index_row_aligned_metadata,
    parse_mcq_options,
    sample_key,
)
from run_generate_longqa_event_ledger import STOPWORDS, _video_metadata
from run_generate_longqa_event_ledger_detector import (
    OpenVocabularyDetector,
    detection_cache_path,
)
from run_generate_longqa_grounded import (
    _run_eval,
    extract_frames_by_indices,
    load_jsonl,
)
from run_generate_longqa_proofpack import (
    baseline_uniform_indices,
    compile_temporal_program,
)


MODES = (
    "object_labels",
    "crop_pairs",
    "evidence_panels",
    "object_tracks",
    "object_ledger",
    "question_router",
)
SCHEMA_VERSION = 1
NON_OBJECT_WORDS = STOPWORDS | {
    "activity",
    "answer",
    "arrive",
    "arrived",
    "became",
    "become",
    "begin",
    "began",
    "bought",
    "bring",
    "brought",
    "carrying",
    "carry",
    "came",
    "change",
    "changed",
    "choose",
    "correct",
    "doing",
    "done",
    "event",
    "find",
    "first",
    "going",
    "happen",
    "happened",
    "holding",
    "last",
    "look",
    "looking",
    "next",
    "option",
    "person",
    "picked",
    "put",
    "returned",
    "second",
    "seen",
    "show",
    "showing",
    "started",
    "stopped",
    "take",
    "took",
    "touch",
    "touched",
    "using",
    "video",
    "wearing",
    "went",
}
SOURCE_PRIORITY = {
    "pivot": 0,
    "directional_target": 0,
    "pivot_context": 1,
    "directional_target_context": 1,
    "bridge": 2,
    "anchor": 3,
    "uniform_global": 3,
    "coverage_fill": 4,
    "semantic_boundary": 5,
}


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def extract_object_concepts(row: dict[str, Any], limit: int = 12) -> list[str]:
    """Extract reproducible open-vocabulary detector prompts without an LLM."""
    options = parse_mcq_options(row.get("mcq_options", ""))
    texts = [*options.values(), str(row.get("question", ""))]
    concepts: list[str] = []
    for text in texts:
        raw_tokens = [
            token.lower() for token in re.findall(r"[A-Za-z][A-Za-z'-]+", text)
        ]
        tokens = [
            token
            for token in raw_tokens
            if len(token) >= 3 and token not in NON_OBJECT_WORDS
        ]
        candidates = [
            f"{left} {right}"
            for left, right in zip(raw_tokens, raw_tokens[1:])
            if len(left) >= 3
            and len(right) >= 3
            and left not in NON_OBJECT_WORDS
            and right not in NON_OBJECT_WORDS
        ] + tokens
        for concept in candidates:
            if concept not in concepts:
                concepts.append(concept)
            if len(concepts) >= limit:
                return concepts
    return concepts


def _selected_priority(item: dict[str, Any]) -> tuple[int, float, float]:
    score = item.get("score")
    return (
        SOURCE_PRIORITY.get(str(item.get("source", "")), 6),
        -float(score) if score is not None else 1e9,
        float(item.get("timestamp", 0.0)),
    )


def choose_detection_indices(
    selected: list[dict[str, Any]],
    total_frames: int,
    proofpack_count: int,
    uniform_count: int,
) -> list[int]:
    focused = sorted(selected, key=_selected_priority)[:proofpack_count]
    indices = [int(item["frame_index"]) for item in focused]
    for frame_index in baseline_uniform_indices(total_frames, uniform_count):
        if frame_index not in indices:
            indices.append(frame_index)
    return sorted(indices)


def _location_name(box: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    x = ((x1 + x2) / 2) / max(width, 1)
    y = ((y1 + y2) / 2) / max(height, 1)
    horizontal = "left" if x < 0.34 else "right" if x > 0.66 else "center"
    vertical = "upper" if y < 0.34 else "lower" if y > 0.66 else "middle"
    return f"{vertical}-{horizontal}"


def _detection_rank(detection: dict[str, Any]) -> tuple[float, float]:
    box = [float(value) for value in detection.get("box", [0, 0, 0, 0])]
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return float(detection.get("score", 0.0)), area


def _top_detections(
    detections: list[dict[str, Any]], count: int = 3
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    labels: set[str] = set()
    for detection in sorted(detections, key=_detection_rank, reverse=True):
        label = str(detection.get("label", "")).strip().lower()
        if not label or label in labels:
            continue
        labels.add(label)
        unique.append(detection)
        if len(unique) >= count:
            break
    return unique


def _load_font(size: int) -> object:
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(10, size))
    except OSError:
        return ImageFont.load_default()


def annotate_frame(
    image: object,
    detections: list[dict[str, Any]],
    timestamp: float,
    max_labels: int = 3,
) -> object:
    from PIL import ImageDraw

    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    font = _load_font(max(16, min(output.width, output.height) // 28))
    for detection in _top_detections(detections, max_labels):
        x1, y1, x2, y2 = [int(round(value)) for value in detection["box"]]
        label = str(detection["label"])
        text = label
        bounds = draw.textbbox((0, 0), text, font=font)
        width = bounds[2] - bounds[0] + 8
        height = bounds[3] - bounds[1] + 6
        text_x = max(0, min(x1, output.width - width))
        text_y = max(0, y1 - height)
        draw.rectangle(
            (text_x, text_y, text_x + width, text_y + height),
            fill=(15, 15, 15),
        )
        draw.text((text_x + 4, text_y + 3), text, fill=(255, 230, 40), font=font)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        draw.line((text_x + width // 2, text_y + height, *center), fill=(255, 60, 60), width=2)
        radius = max(5, min(output.width, output.height) // 100)
        draw.ellipse(
            (
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ),
            fill=(255, 60, 60),
        )
    stamp = f"{timestamp:.1f}s"
    stamp_bounds = draw.textbbox((0, 0), stamp, font=font)
    draw.rectangle(
        (0, 0, stamp_bounds[2] - stamp_bounds[0] + 12, stamp_bounds[3] + 10),
        fill=(15, 15, 15),
    )
    draw.text((6, 5), stamp, fill=(255, 255, 255), font=font)
    return output


def crop_detection(
    image: object,
    detection: dict[str, Any],
    timestamp: float,
    expansion: float = 1.4,
) -> object:
    from PIL import Image, ImageDraw

    image = image.convert("RGB")
    x1, y1, x2, y2 = [float(value) for value in detection["box"]]
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    width = max(32.0, (x2 - x1) * expansion)
    height = max(32.0, (y2 - y1) * expansion)
    left = max(0, int(center_x - width / 2))
    top = max(0, int(center_y - height / 2))
    right = min(image.width, int(center_x + width / 2))
    bottom = min(image.height, int(center_y + height / 2))
    crop = image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
    header = max(36, crop.height // 10)
    canvas = Image.new("RGB", (crop.width, crop.height + header), (15, 15, 15))
    canvas.paste(crop, (0, header))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(14, min(crop.width, crop.height) // 18))
    draw.text(
        (5, max(3, header // 3)),
        f"{timestamp:.1f}s detail: {detection['label']}",
        fill=(255, 230, 40),
        font=font,
    )
    return canvas


def build_evidence_panel(
    image: object,
    detections: list[dict[str, Any]],
    timestamp: float,
    size: int = 672,
) -> object:
    from PIL import Image, ImageDraw, ImageOps

    image = image.convert("RGB")
    top_detections = _top_detections(detections, 2)
    canvas = Image.new("RGB", (size, size), (20, 20, 20))
    full_height = int(size * 0.60)
    canvas.paste(ImageOps.fit(image, (size, full_height)), (0, 0))
    crop_width = size // max(1, len(top_detections))
    for index, detection in enumerate(top_detections):
        crop = crop_detection(image, detection, timestamp)
        crop = ImageOps.fit(crop, (crop_width, size - full_height))
        canvas.paste(crop, (index * crop_width, full_height))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(16, size // 30))
    label = f"{timestamp:.1f}s panel"
    bounds = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, bounds[2] + 12, bounds[3] + 10), fill=(15, 15, 15))
    draw.text((6, 5), label, fill=(255, 255, 255), font=font)
    return canvas


def choose_detail_frames(
    detection_frames: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    ranked = []
    for frame in detection_frames:
        detections = _top_detections(frame.get("detections", []), 2)
        if not detections:
            continue
        ranked.append(
            (
                max(float(item["score"]) for item in detections),
                float(frame["timestamp"]),
                frame,
            )
        )
    chosen = []
    for _, _, frame in sorted(ranked, key=lambda item: (-item[0], item[1])):
        if all(
            abs(float(frame["timestamp"]) - float(other["timestamp"])) >= 2.0
            for other in chosen
        ):
            chosen.append(frame)
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        for _, _, frame in sorted(ranked, key=lambda item: (-item[0], item[1])):
            if frame not in chosen:
                chosen.append(frame)
            if len(chosen) >= count:
                break
    return sorted(chosen, key=lambda item: float(item["timestamp"]))


def choose_base_frames(
    selected: list[dict[str, Any]], count: int, required_indices: set[int]
) -> list[dict[str, Any]]:
    required = [
        item for item in selected if int(item["frame_index"]) in required_indices
    ]
    others = [
        item for item in sorted(selected, key=_selected_priority) if item not in required
    ]
    kept = required + others[: max(0, count - len(required))]
    return sorted(kept[:count], key=lambda item: float(item["timestamp"]))


def build_object_track_indices(
    detection_frames: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    fps: float,
    total_frames: int,
    max_frames: int,
    event_budget: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    by_label: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for frame in detection_frames:
        for detection in _top_detections(frame.get("detections", []), 4):
            by_label[str(detection["label"]).lower()].append((frame, detection))

    events: list[dict[str, Any]] = []
    for label, observations in by_label.items():
        observations.sort(key=lambda item: float(item[0]["timestamp"]))
        peak = max(observations, key=lambda item: float(item[1]["score"]))
        for role, observation in (
            ("first", observations[0]),
            ("peak", peak),
            ("last", observations[-1]),
        ):
            frame, detection = observation
            events.append(
                {
                    "label": label,
                    "role": role,
                    "frame_index": int(frame["frame_index"]),
                    "timestamp": float(frame["timestamp"]),
                    "score": float(detection["score"]),
                }
            )
    events.sort(key=lambda item: (-float(item["score"]), float(item["timestamp"])))
    chosen_events: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    event_center_limit = max(1, (event_budget + 2) // 3)
    for event in events:
        key = (str(event["label"]), int(event["frame_index"]))
        if key in seen:
            continue
        seen.add(key)
        chosen_events.append(event)
        if len(chosen_events) >= event_center_limit:
            break

    event_indices: list[int] = []
    neighbor_offset = max(1, int(round(fps)))
    for event in chosen_events:
        center = int(event["frame_index"])
        for frame_index in (center - neighbor_offset, center, center + neighbor_offset):
            if 0 <= frame_index < total_frames and frame_index not in event_indices:
                event_indices.append(frame_index)
    event_indices = event_indices[:event_budget]
    final = list(event_indices)
    for item in sorted(selected, key=_selected_priority):
        frame_index = int(item["frame_index"])
        if frame_index not in final:
            final.append(frame_index)
        if len(final) >= max_frames:
            break
    return sorted(final[:max_frames]), sorted(
        chosen_events, key=lambda item: float(item["timestamp"])
    )


def build_detection_ledger(
    detection_frames: list[dict[str, Any]], frame_sizes: dict[int, tuple[int, int]]
) -> str:
    lines = []
    for frame in sorted(detection_frames, key=lambda item: float(item["timestamp"])):
        detections = _top_detections(frame.get("detections", []), 3)
        if not detections:
            continue
        width, height = frame_sizes.get(int(frame["frame_index"]), (1, 1))
        descriptions = []
        for detection in detections:
            box = [float(value) for value in detection["box"]]
            area_ratio = (
                max(0.0, box[2] - box[0])
                * max(0.0, box[3] - box[1])
                / max(1, width * height)
            )
            size = "large" if area_ratio > 0.20 else "small" if area_ratio < 0.04 else "medium"
            descriptions.append(
                f"{detection['label']} ({_location_name(box, width, height)}, {size})"
            )
        lines.append(f"- {float(frame['timestamp']):.1f}s: " + "; ".join(descriptions))
    return "\n".join(lines[:16])


def classify_question(row: dict[str, Any]) -> str:
    text = f"{row.get('question', '')} {row.get('mcq_options', '')}".lower()
    program = compile_temporal_program(row.get("question", ""))
    if re.search(r"\b(how many|number of|count|times|changed|change|different)\b", text):
        return "state_or_count"
    if re.search(
        r"\b(color|colour|written|read|text|brand|type of|kind of|wearing|"
        r"material|shape|look like|which item|what item|which object|what object)\b",
        text,
    ):
        return "object_detail"
    if re.search(
        r"\b(where|left|right|inside|outside|behind|beside|next to|location|placed)\b",
        text,
    ):
        return "spatial"
    if program.operator != "GLOBAL":
        return "temporal"
    return "global"


def build_augmented_prompt(
    row: dict[str, Any],
    mode: str,
    ledger: str = "",
    route: str | None = None,
) -> str:
    if mode == "object_labels":
        instruction = (
            "The images are chronological. Automatic labels point to potentially "
            "relevant objects. Verify every label against the pixels because detections "
            "can be wrong."
        )
    elif mode == "crop_pairs":
        instruction = (
            "The images are chronological. Some full frames are immediately followed by "
            "an enlarged detail from the same timestamp. Use the full frame for context "
            "and the detail for object identity or state."
        )
    elif mode == "evidence_panels":
        instruction = (
            "The images are chronological. Some evidence panels combine a full frame "
            "with enlarged details from the same timestamp. Treat each panel as one "
            "moment, not as multiple events."
        )
    elif mode in {"object_tracks", "object_ledger"}:
        instruction = (
            "The images are chronological. The automatically detected object record is "
            "only a navigation aid and may contain mistakes. Confirm the answer from the "
            "visible frames and preserve event order.\n\nDetected object record:\n"
            + (ledger or "- no confident detections")
        )
    else:
        route_instructions = {
            "global": "Use the chronological images as coverage of the complete recording.",
            "temporal": "Identify the referenced events and verify their chronological order.",
            "object_detail": (
                "Use each full frame for context and its enlarged detail for object identity."
            ),
            "spatial": (
                "Use each full frame to verify object location; enlarged details only aid identity."
            ),
            "state_or_count": (
                "Track repeated object occurrences chronologically and count only visibly "
                "distinct events or state changes."
            ),
        }
        instruction = (
            route_instructions.get(route or "global", route_instructions["global"])
            + " Automatic labels, details, and object records may contain errors; "
            "confirm the answer from visible evidence."
        )
        if ledger:
            instruction += "\n\nDetected object record:\n" + ledger
    return instruction + "\n\n" + build_longqa_prompt(
        row["question"], row["mcq_options"], prompt_variant="baseline"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--proofpack", required=True)
    parser.add_argument("--proofpack-reference", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--detections-output", required=True)
    parser.add_argument("--detection-cache-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--detector-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--detector-batch-size", type=int, default=8)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--concept-limit", type=int, default=12)
    parser.add_argument("--proofpack-detection-frames", type=int, default=24)
    parser.add_argument("--uniform-detection-frames", type=int, default=8)
    parser.add_argument("--detail-count", type=int, default=8)
    parser.add_argument("--track-event-budget", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--panel-size", type=int, default=672)
    parser.add_argument("--llm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    import time

    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    eval_output = _resolve_path(args.eval_output)
    detections_output = _resolve_path(args.detections_output)
    detection_cache_dir = _resolve_path(args.detection_cache_dir)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    reference = load_jsonl(_resolve_path(args.proofpack_reference))
    packs = index_row_aligned_metadata(
        load_jsonl(_resolve_path(args.proofpack)), reference, "proof pack"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(detections_output) or ".", exist_ok=True)
    os.makedirs(detection_cache_dir, exist_ok=True)

    config = {
        "schema": SCHEMA_VERSION,
        "mode": args.mode,
        "proofpack": os.path.abspath(_resolve_path(args.proofpack)),
        "detector_model": args.detector_model,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "concept_limit": args.concept_limit,
        "proofpack_detection_frames": args.proofpack_detection_frames,
        "uniform_detection_frames": args.uniform_detection_frames,
        "detail_count": args.detail_count,
        "track_event_budget": args.track_event_budget,
        "max_frames": args.max_frames,
        "panel_size": args.panel_size,
        "llm_model": args.llm_model,
    }
    fingerprint = hashlib.sha1(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:12]
    detection_fingerprint = hashlib.sha1(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "proofpack": config["proofpack"],
                "detector_model": args.detector_model,
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "concept_limit": args.concept_limit,
                "proofpack_detection_frames": args.proofpack_detection_frames,
                "uniform_detection_frames": args.uniform_detection_frames,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    begun = time.time()

    if args.no_resume and os.path.exists(detections_output):
        Path(detections_output).unlink()
    existing_detections = (
        {
            str(item["sample_key"]): item
            for item in load_jsonl(detections_output)
            if item.get("detection_fingerprint") == detection_fingerprint
        }
        if not args.no_resume and os.path.exists(detections_output)
        else {}
    )
    pending = [row for row in rows if sample_key(row) not in existing_detections]
    if pending:
        detector: OpenVocabularyDetector | None = None
        for row_index, row in enumerate(pending):
            key = sample_key(row)
            video_path = os.path.join(video_folder, str(row["video_path"]))
            fps, total_frames = _video_metadata(video_path)
            selected = packs[key]["selected"]
            frame_indices = choose_detection_indices(
                selected,
                total_frames,
                args.proofpack_detection_frames,
                args.uniform_detection_frames,
            )
            concepts = extract_object_concepts(row, args.concept_limit)
            per_frame: dict[int, list[dict[str, Any]]] = {}
            missing_indices = []
            missing_paths: list[Path] = []
            for frame_index in frame_indices:
                cache_path = detection_cache_path(
                    detection_cache_dir,
                    args.detector_model,
                    str(row["video_path"]),
                    frame_index,
                    concepts,
                    args.box_threshold,
                    args.text_threshold,
                )
                if cache_path.exists():
                    per_frame[frame_index] = json.loads(cache_path.read_text())["detections"]
                else:
                    missing_indices.append(frame_index)
                    missing_paths.append(cache_path)
            missing_frames = (
                extract_frames_by_indices(video_path, missing_indices)
                if missing_indices
                else []
            )
            if len(missing_frames) != len(missing_indices):
                raise RuntimeError(
                    f"Decoded {len(missing_frames)}/{len(missing_indices)} detection "
                    f"frames for {row['video_path']}"
                )
            for start in range(0, len(missing_frames), args.detector_batch_size):
                if detector is None:
                    detector = OpenVocabularyDetector(args.detector_model)
                batch_frames = missing_frames[start : start + args.detector_batch_size]
                batch_results = detector.detect_batch(
                    batch_frames, concepts, args.box_threshold, args.text_threshold
                )
                for frame_index, cache_path, detections in zip(
                    missing_indices[start : start + args.detector_batch_size],
                    missing_paths[start : start + args.detector_batch_size],
                    batch_results,
                ):
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
                    temporary.write_text(json.dumps({"detections": detections}) + "\n")
                    os.replace(temporary, cache_path)
                    per_frame[frame_index] = detections
            record = {
                "sample_key": key,
                "video_path": row.get("video_path", ""),
                "concepts": concepts,
                "detector_model": args.detector_model,
                "detection_schema": SCHEMA_VERSION,
                "detection_fingerprint": detection_fingerprint,
                "frames": [
                    {
                        "frame_index": frame_index,
                        "timestamp": round(frame_index / max(fps, 1e-6), 3),
                        "detections": per_frame.get(frame_index, []),
                    }
                    for frame_index in frame_indices
                ],
            }
            with open(detections_output, "a") as handle:
                handle.write(json.dumps(record) + "\n")
            existing_detections[key] = record
            print(f"  Detection progress: {row_index + 1}/{len(pending)}")
        if detector is not None:
            del detector
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    else:
        print("Detection cache complete; skipping detector model load")

    existing = (
        load_jsonl(output_path)
        if not args.no_resume and os.path.exists(output_path)
        else []
    )
    start = 0
    for index, prediction in enumerate(existing[: len(rows)]):
        if (
            sample_key(prediction) != sample_key(rows[index])
            or prediction.get("object_hint_fingerprint") != fingerprint
            or not str(prediction.get("mcq_answer", "")).strip()
        ):
            break
        start += 1
    if len(existing) != start:
        _write_jsonl(output_path, existing[:start])
    if start:
        print(f"Resuming generation from {start}/{len(rows)}")

    model = VLLMModel(
        args.llm_model,
        tp_size=1,
        concurrency=args.concurrency,
        max_frames=args.max_frames,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    mode = "a" if start else "w"
    with model, open(output_path, mode) as handle:
        for row_index, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            video_path = os.path.join(video_folder, str(row["video_path"]))
            fps, total_frames = _video_metadata(video_path)
            selected = packs[key]["selected"]
            detection_frames = existing_detections[key]["frames"]
            detection_map = {
                int(item["frame_index"]): item.get("detections", [])
                for item in detection_frames
            }
            route = None
            effective_mode = args.mode
            if args.mode == "question_router":
                route = classify_question(row)
                if route in {"object_detail", "spatial"}:
                    effective_mode = "crop_pairs"
                elif route == "state_or_count":
                    effective_mode = "object_tracks"
                elif route == "global":
                    effective_mode = "global_uniform"
                else:
                    effective_mode = "temporal_pivot"

            selected_indices = {
                int(item["frame_index"]) for item in selected
            }
            detail_frames = choose_detail_frames(
                [
                    item
                    for item in detection_frames
                    if int(item["frame_index"]) in selected_indices
                ],
                args.detail_count,
            )
            detail_indices = {int(item["frame_index"]) for item in detail_frames}
            frame_sizes: dict[int, tuple[int, int]] = {}
            object_events: list[dict[str, Any]] = []

            if effective_mode in {"crop_pairs", "evidence_panels"}:
                base_count = max(1, args.max_frames - len(detail_frames))
                base_meta = choose_base_frames(selected, base_count, detail_indices)
                indices = sorted(
                    {int(item["frame_index"]) for item in base_meta} | detail_indices
                )
            elif effective_mode == "object_tracks":
                indices, object_events = build_object_track_indices(
                    detection_frames,
                    selected,
                    fps,
                    total_frames,
                    args.max_frames,
                    args.track_event_budget,
                )
                base_meta = [
                    {
                        "frame_index": index,
                        "timestamp": index / max(fps, 1e-6),
                        "source": "object_track",
                    }
                    for index in indices
                ]
            elif effective_mode == "global_uniform":
                indices = baseline_uniform_indices(total_frames, args.max_frames)
                base_meta = [
                    {
                        "frame_index": index,
                        "timestamp": index / max(fps, 1e-6),
                        "source": "uniform_global",
                    }
                    for index in indices
                ]
            else:
                base_meta = selected[: args.max_frames]
                indices = [int(item["frame_index"]) for item in base_meta]

            extracted = extract_frames_by_indices(video_path, indices)
            image_by_index = dict(zip(indices, extracted))
            frame_sizes.update(
                {
                    index: image.size
                    for index, image in image_by_index.items()
                    if hasattr(image, "size")
                }
            )
            default_size = next(iter(frame_sizes.values()), (1, 1))
            for item in detection_frames:
                frame_sizes.setdefault(int(item["frame_index"]), default_size)
            ledger = build_detection_ledger(detection_frames, frame_sizes)
            evidence: list[tuple[float, int, object]] = []
            for item in base_meta:
                frame_index = int(item["frame_index"])
                image = image_by_index.get(frame_index)
                if image is None:
                    continue
                timestamp = float(item.get("timestamp", frame_index / max(fps, 1e-6)))
                if effective_mode == "object_labels":
                    image = annotate_frame(
                        image, detection_map.get(frame_index, []), timestamp
                    )
                evidence.append((timestamp, 0, image))

            if effective_mode in {"crop_pairs", "evidence_panels"}:
                for item in detail_frames:
                    frame_index = int(item["frame_index"])
                    image = image_by_index.get(frame_index)
                    detections = _top_detections(
                        detection_map.get(frame_index, []), 2
                    )
                    if image is None or not detections:
                        continue
                    timestamp = float(item["timestamp"])
                    if effective_mode == "crop_pairs":
                        detail = crop_detection(image, detections[0], timestamp)
                    else:
                        detail = build_evidence_panel(
                            image, detections, timestamp, args.panel_size
                        )
                    evidence.append((timestamp, 1, detail))
            evidence.sort(key=lambda item: (item[0], item[1]))
            frames = [item[2] for item in evidence[: args.max_frames]]
            prompt = build_augmented_prompt(
                row,
                effective_mode
                if args.mode != "question_router"
                else "question_router",
                ledger=ledger if effective_mode in {"object_tracks", "object_ledger"} else "",
                route=route,
            )
            response = model.generate(
                frames, [{"role": "user", "content": prompt}], max_new_tokens=16
            )
            prediction = build_prediction_row(
                row, response, prompt_variant=f"object_hints_{args.mode}"
            )
            prediction.update(
                {
                    "object_hint_mode": args.mode,
                    "object_hint_effective_mode": effective_mode,
                    "object_hint_route": route,
                    "object_hint_fingerprint": fingerprint,
                    "object_hint_concepts": existing_detections[key]["concepts"],
                    "object_hint_detection_frames": len(detection_frames),
                    "object_hint_detail_frames": len(detail_frames)
                    if effective_mode in {"crop_pairs", "evidence_panels"}
                    else 0,
                    "object_hint_final_frames": len(frames),
                    "object_hint_events": object_events,
                }
            )
            handle.write(json.dumps(prediction) + "\n")
            handle.flush()
            print(f"  Generation progress: {row_index + 1}/{len(rows)}")

    print(f"Predictions written to {output_path}")
    print(f"Detections written to {detections_output}")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
