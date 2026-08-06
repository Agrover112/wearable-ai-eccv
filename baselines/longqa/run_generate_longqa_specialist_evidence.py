#!/usr/bin/env python3
"""Run gated Qwen OCR or object re-identification evidence branches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from longqa_utils import (
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    compute_diagnostics,
    index_row_aligned_metadata,
    load_jsonl,
    sample_key,
)
from run_generate_longqa_grounded import (
    CandidateFrame,
    create_text_image_grounder,
    extract_frames_by_indices,
)
from run_generate_longqa_object_hints import (
    _top_detections,
    choose_base_frames,
    choose_detail_frames,
    crop_detection,
)
from run_generate_longqa_uncertainty import _index_jsonl, _video_metadata


MODES = ("qwen_ocr", "object_reid")
SCHEMA_VERSION = 1


def _resolve(path: str | None) -> str | None:
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def raw_crop(image: object, detection: dict[str, Any], expansion: float = 1.35) -> object:
    image = image.convert("RGB")
    x1, y1, x2, y2 = [float(value) for value in detection["box"]]
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    width = max(32.0, (x2 - x1) * expansion)
    height = max(32.0, (y2 - y1) * expansion)
    left = max(0, int(center_x - width / 2))
    top = max(0, int(center_y - height / 2))
    right = min(image.width, max(left + 1, int(center_x + width / 2)))
    bottom = min(image.height, max(top + 1, int(center_y + height / 2)))
    return image.crop((left, top, right, bottom))


def normalize_label(label: object) -> str:
    return " ".join(str(label).lower().replace("_", " ").split())


def cluster_reid_observations(
    observations: list[dict[str, Any]],
    features: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    """Cluster same-concept detections into appearance-consistent track identities."""
    clusters: list[dict[str, Any]] = []
    for observation, feature in sorted(
        zip(observations, features), key=lambda item: float(item[0]["timestamp"])
    ):
        label = normalize_label(observation["label"])
        candidates = [cluster for cluster in clusters if cluster["label"] == label]
        best = None
        best_similarity = -1.0
        for cluster in candidates:
            similarity = float(feature @ cluster["centroid"])
            if similarity > best_similarity:
                best = cluster
                best_similarity = similarity
        if best is None or best_similarity < threshold:
            clusters.append(
                {
                    "label": label,
                    "centroid": feature.copy(),
                    "observations": [dict(observation, reid_similarity=1.0)],
                }
            )
            continue
        best["observations"].append(
            dict(observation, reid_similarity=best_similarity)
        )
        centroid = np.mean(
            [features_item for features_item in [best["centroid"], feature]], axis=0
        )
        norm = np.linalg.norm(centroid)
        best["centroid"] = centroid / max(float(norm), 1e-12)
    return clusters


def summarize_tracks(
    clusters: list[dict[str, Any]], max_tracks: int
) -> tuple[list[int], list[dict[str, Any]]]:
    ranked = sorted(
        clusters,
        key=lambda cluster: (
            -len(cluster["observations"]),
            -max(float(item["score"]) for item in cluster["observations"]),
            cluster["label"],
        ),
    )[:max_tracks]
    events: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    for track_number, cluster in enumerate(ranked, start=1):
        observations = sorted(
            cluster["observations"], key=lambda item: float(item["timestamp"])
        )
        peak = max(observations, key=lambda item: float(item["score"]))
        chosen = [("first", observations[0]), ("peak", peak), ("last", observations[-1])]
        seen = set()
        for role, item in chosen:
            frame_index = int(item["frame_index"])
            if (role, frame_index) in seen:
                continue
            seen.add((role, frame_index))
            selected_indices.append(frame_index)
            events.append(
                {
                    "track_id": f"T{track_number:02d}",
                    "label": cluster["label"],
                    "role": role,
                    "frame_index": frame_index,
                    "timestamp": float(item["timestamp"]),
                    "score": float(item["score"]),
                    "reid_similarity": float(item.get("reid_similarity", 1.0)),
                    "observations": len(observations),
                }
            )
    return sorted(set(selected_indices)), sorted(events, key=lambda item: item["timestamp"])


def prepare_reid_record(
    grounder: Any,
    row: dict[str, Any],
    video_path: str,
    detection_record: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    frames = []
    for frame in detection_record.get("frames", []):
        for detection in _top_detections(frame.get("detections", []), args.detections_per_frame):
            frames.append((frame, detection))
    frames = sorted(
        frames,
        key=lambda item: (-float(item[1].get("score", 0.0)), float(item[0]["timestamp"])),
    )[: args.max_reid_detections]
    frames.sort(key=lambda item: float(item[0]["timestamp"]))
    indices = sorted({int(frame["frame_index"]) for frame, _ in frames})
    images = extract_frames_by_indices(video_path, indices)
    by_index = dict(zip(indices, images))
    observations = []
    candidates = []
    for frame, detection in frames:
        frame_index = int(frame["frame_index"])
        image = by_index.get(frame_index)
        if image is None:
            continue
        crop = raw_crop(image, detection)
        timestamp = float(frame["timestamp"])
        observations.append(
            {
                "label": detection["label"],
                "frame_index": frame_index,
                "timestamp": timestamp,
                "score": float(detection["score"]),
                "box": detection["box"],
            }
        )
        candidates.append(CandidateFrame(frame_index, timestamp, crop))
    if not candidates:
        return {"sample_key": sample_key(row), "selected_indices": [], "events": []}
    features = grounder.encode_images(candidates)
    clusters = cluster_reid_observations(observations, features, args.reid_threshold)
    selected, events = summarize_tracks(clusters, args.max_tracks)
    return {
        "sample_key": sample_key(row),
        "video_path": row["video_path"],
        "selected_indices": selected,
        "events": events,
        "detections_encoded": len(candidates),
        "tracks": len(clusters),
    }


def build_reid_ledger(events: list[dict[str, Any]]) -> str:
    if not events:
        return "- No repeated target object could be linked confidently."
    return "\n".join(
        f"- {event['timestamp']:.1f}s: {event['track_id']} ({event['label']}), "
        f"{event['role']} observation"
        for event in events
    )


def build_final_indices(
    selected: list[dict[str, Any]],
    priority: list[int],
    fps: float,
    total_frames: int,
    max_frames: int,
) -> list[int]:
    indices = []
    offset = max(1, int(round(fps)))
    for center in priority:
        for frame_index in (center - offset, center, center + offset):
            if 0 <= frame_index < total_frames and frame_index not in indices:
                indices.append(frame_index)
    for item in selected:
        frame_index = int(item["frame_index"])
        if frame_index not in indices:
            indices.append(frame_index)
        if len(indices) >= max_frames:
            break
    return sorted(indices[:max_frames])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl")
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--proofpack", required=True)
    parser.add_argument("--proofpack-reference", required=True)
    parser.add_argument("--detections-input", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--detail-count", type=int, default=12)
    parser.add_argument("--reid-model", default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--reid-batch-size", type=int, default=16)
    parser.add_argument("--reid-threshold", type=float, default=0.72)
    parser.add_argument("--detections-per-frame", type=int, default=2)
    parser.add_argument("--max-reid-detections", type=int, default=48)
    parser.add_argument("--max-tracks", type=int, default=6)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def run_fingerprint(args: argparse.Namespace) -> str:
    ignored = {"output", "evidence_output", "eval_output", "no_resume"}
    payload = {key: value for key, value in vars(args).items() if key not in ignored}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    import time
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve(args.input)
    video_folder = _resolve(args.video_folder)
    output_path = _resolve(args.output)
    evidence_path = _resolve(args.evidence_output)
    eval_path = _resolve(args.eval_output)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    proofpacks = index_row_aligned_metadata(
        load_jsonl(_resolve(args.proofpack)),
        load_jsonl(_resolve(args.proofpack_reference)),
        "specialist proof pack",
    )
    detections = _index_jsonl(_resolve(args.detections_input))
    required = {sample_key(row) for row in rows}
    for label, indexed in (("proofpack", proofpacks), ("detections", detections)):
        missing = required - set(indexed)
        if missing:
            raise RuntimeError(f"{label} is missing {len(missing)} specialist rows")
    fingerprint = run_fingerprint(args)
    evidence_records = (
        _index_jsonl(evidence_path)
        if args.mode == "object_reid" and not args.no_resume and os.path.exists(evidence_path)
        else {}
    )
    evidence_records = {
        key: record
        for key, record in evidence_records.items()
        if record.get("specialist_fingerprint") == fingerprint
    }
    if args.mode == "object_reid":
        pending = [row for row in rows if sample_key(row) not in evidence_records]
        if pending:
            grounder = create_text_image_grounder(
                args.reid_model,
                device="cuda",
                batch_size=args.reid_batch_size,
                dtype="bfloat16",
            )
            Path(evidence_path).parent.mkdir(parents=True, exist_ok=True)
            with open(evidence_path, "a") as handle:
                for index, row in enumerate(pending):
                    key = sample_key(row)
                    video_path = os.path.join(video_folder, str(row["video_path"]))
                    record = prepare_reid_record(
                        grounder, row, video_path, detections[key], args
                    )
                    record["specialist_fingerprint"] = fingerprint
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    evidence_records[key] = record
                    print(f"  ReID preparation: {index + 1}/{len(pending)}")
            del grounder
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
    else:
        write_jsonl(evidence_path, [])

    existing = [] if args.no_resume or not os.path.exists(output_path) else load_jsonl(output_path)
    start = 0
    for index, prediction in enumerate(existing[: len(rows)]):
        if (
            sample_key(prediction) != sample_key(rows[index])
            or prediction.get("specialist_fingerprint") != fingerprint
        ):
            break
        start += 1
    write_jsonl(output_path, existing[:start])
    model = VLLMModel(
        args.llm_model,
        tp_size=1,
        concurrency=args.concurrency,
        max_frames=args.max_frames,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    begun = time.time()
    with model, open(output_path, "a") as handle:
        for index, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            video_path = os.path.join(video_folder, str(row["video_path"]))
            fps, total_frames = _video_metadata(video_path)
            selected = proofpacks[key]["selected"]
            detection_frames = detections[key].get("frames", [])
            if args.mode == "qwen_ocr":
                details = choose_detail_frames(detection_frames, args.detail_count)
                detail_indices = [int(item["frame_index"]) for item in details]
                source_images = extract_frames_by_indices(video_path, detail_indices)
                detection_map = {
                    int(item["frame_index"]): item.get("detections", [])
                    for item in detection_frames
                }
                crops = []
                valid_details = []
                for item, image in zip(details, source_images):
                    top = _top_detections(detection_map.get(int(item["frame_index"]), []), 1)
                    if top:
                        crops.append(crop_detection(image, top[0], float(item["timestamp"])))
                        valid_details.append(item)
                ocr_prompt = (
                    "Transcribe only visible text that may help answer this question. "
                    "Preserve names, prices, numbers, capitalization, and timestamps. "
                    "Do not answer the multiple-choice question and do not infer unreadable text.\n\n"
                    f"Question: {row['question']}\n\nOptions:\n{row['mcq_options']}"
                )
                ledger = str(
                    model.generate(
                        crops,
                        [{"role": "user", "content": ocr_prompt}],
                        max_new_tokens=256,
                    )
                )
                base_count = max(1, args.max_frames - len(crops))
                valid_detail_indices = [int(item["frame_index"]) for item in valid_details]
                base = choose_base_frames(selected, base_count, set(valid_detail_indices))
                base_indices = [int(item["frame_index"]) for item in base]
                base_images = extract_frames_by_indices(video_path, base_indices)
                evidence = [(float(item["timestamp"]), 0, image) for item, image in zip(base, base_images)]
                evidence.extend(
                    (float(item["timestamp"]), 1, crop)
                    for item, crop in zip(valid_details, crops)
                )
                evidence.sort(key=lambda item: (item[0], item[1]))
                frames = [item[2] for item in evidence[: args.max_frames]]
                prompt = (
                    "The following is an automatic transcription of selected visual details. "
                    "It may omit or misread text, so verify it against the images.\n\n"
                    f"Transcription:\n{ledger}\n\n"
                    + build_longqa_prompt(row["question"], row["mcq_options"])
                )
                evidence_meta = {
                    "detail_indices": valid_detail_indices,
                    "ocr_ledger": ledger,
                }
            else:
                record = evidence_records[key]
                final_indices = build_final_indices(
                    selected,
                    record.get("selected_indices", []),
                    fps,
                    total_frames,
                    args.max_frames,
                )
                frames = extract_frames_by_indices(video_path, final_indices)
                ledger = build_reid_ledger(record.get("events", []))
                prompt = (
                    "The object track IDs below link visually similar detections across time. "
                    "They are retrieval aids and may be wrong; confirm identity and state from "
                    "the chronological images.\n\n"
                    f"Object tracks:\n{ledger}\n\n"
                    + build_longqa_prompt(row["question"], row["mcq_options"])
                )
                evidence_meta = {
                    "final_indices": final_indices,
                    "events": record.get("events", []),
                }
            response = model.generate(
                frames,
                [{"role": "user", "content": prompt}],
                max_new_tokens=16,
            )
            prediction = build_prediction_row(row, response, prompt_variant=f"specialist_{args.mode}")
            prediction.update(
                {
                    "specialist_schema": SCHEMA_VERSION,
                    "specialist_fingerprint": fingerprint,
                    "specialist_mode": args.mode,
                    "specialist_final_frames": len(frames),
                    "specialist_evidence": evidence_meta,
                }
            )
            handle.write(json.dumps(prediction) + "\n")
            handle.flush()
            print(f"  Specialist progress: {index + 1}/{len(rows)}")
    predictions = load_jsonl(output_path)
    result = compute_diagnostics(rows, predictions, run_id=f"specialist_{args.mode}")
    Path(eval_path).write_text(json.dumps(result, indent=2) + "\n")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    print(f"accuracy={result['accuracy_raw']:.4f} correct={result['correct']}/{result['total']}")


if __name__ == "__main__":
    main()
