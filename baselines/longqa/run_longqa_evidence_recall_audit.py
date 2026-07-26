#!/usr/bin/env python3
"""Prepare and score human evidence-recall audits for LongQA frame packs.

The runner deliberately does not infer decisive events.  Without an annotation
JSONL it writes a deterministic manifest containing the question, prediction
outcome, video duration, and every candidate/final-pack timestamp.  A human
can then add decisive event intervals to a separate annotation JSONL.  With
that JSONL present, the same manifest is scored offline.

Annotation JSONL schema (one record per ``sample_key``)::

    {
      "sample_key": "...",
      "decisive_events": [
        {
          "event_id": "pivot",
          "start_seconds": 41.0,
          "end_seconds": 44.0,
          "role": "pivot",
          "required": true
        },
        {
          "event_id": "target",
          "start_seconds": 47.0,
          "end_seconds": 50.0,
          "role": "target",
          "required": true
        }
      ],
      "temporal_order": ["pivot", "target"],
      "notes": ""
    }

``temporal_order`` is optional.  When supplied, the audit checks whether a
single increasing sequence of sampled timestamps can cover those events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


AUDIT_SCHEMA = 1
MANIFEST_SCHEMA = 1
RESULTS_SCHEMA = 1

ANNOTATION_SCHEMA: dict[str, Any] = {
    "schema": AUDIT_SCHEMA,
    "description": (
        "Manual labels for deciding whether a candidate grid or final frame pack "
        "contains all visual evidence needed for a LongQA question. Do not infer "
        "event times from model output."
    ),
    "record": {
        "sample_key": "Stable key copied from annotation_manifest.jsonl.",
        "decisive_events": [
            {
                "event_id": "Unique event label within the sample.",
                "start_seconds": "Inclusive start of the decisive evidence interval.",
                "end_seconds": "Inclusive end of the decisive evidence interval.",
                "role": "Optional label such as pivot, target, state_before, or state_after.",
                "required": "Boolean; defaults to true for scoring.",
            }
        ],
        "temporal_order": (
            "Optional ordered list of event_id values. Its order is scored only "
            "when it contains at least two event IDs."
        ),
        "notes": "Optional free-text annotation rationale.",
    },
}


def sample_key(row: dict[str, Any]) -> str:
    return str(row.get("id") or f"{row.get('video_path', '')}||{row.get('question', '')}")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: str, value: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def file_sha1(path: str | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def _subset_keys(path: str | None) -> set[str] | None:
    if path is None:
        return None
    with open(path, "r") as handle:
        raw = json.load(handle)
    values = raw.get("samples", raw) if isinstance(raw, dict) else raw
    keys: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            keys.add(str(value.get("key") or value.get("sample_key") or sample_key(value)))
        else:
            keys.add(str(value))
    return keys


def apply_subset(rows: list[dict[str, Any]], subset_file: str | None) -> list[dict[str, Any]]:
    keys = _subset_keys(subset_file)
    return rows if keys is None else [row for row in rows if sample_key(row) in keys]


def index_by_sample_key(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("sample_key") or sample_key(row))
        if not key or key == "||":
            raise RuntimeError(f"{label} row is missing sample_key")
        if key in indexed:
            raise RuntimeError(f"{label} has duplicate sample_key: {key}")
        indexed[key] = row
    return indexed


def normalize_answer(raw: object) -> str:
    text = str(raw or "").strip().upper()
    if len(text) == 1 and text in "ABCD":
        return text
    matches = re.findall(r"(?:ANSWER|OPTION|CHOICE)\s*[:.]?\s*\(?([A-D])\)?", text)
    if matches:
        return matches[-1]
    standalone = re.findall(r"\b([A-D])\b", text)
    return standalone[-1] if standalone else ""


def prediction_answer(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))


def prediction_summary(
    prediction: dict[str, Any] | None, gold_answer: str
) -> dict[str, Any] | None:
    if prediction is None:
        return None
    answer = prediction_answer(prediction)
    return {"answer": answer, "correct": answer == gold_answer}


def selection_reasons(
    row: dict[str, Any],
    primary: dict[str, Any] | None,
    secondary: dict[str, Any] | None,
) -> list[str]:
    gold = normalize_answer(row.get("mcq_answer"))
    primary_answer = prediction_answer(primary)
    secondary_answer = prediction_answer(secondary)
    reasons: list[str] = []
    if primary is not None and primary_answer != gold:
        reasons.append("primary_error")
    if secondary is not None and secondary_answer != gold:
        reasons.append("secondary_error")
    if primary is not None and secondary is not None and primary_answer != secondary_answer:
        reasons.append("prediction_disagreement")
    if primary is None and secondary is None:
        reasons.append("no_predictions_available")
    return reasons


def select_rows_for_annotation(
    rows: list[dict[str, Any]],
    primary_predictions: dict[str, dict[str, Any]],
    secondary_predictions: dict[str, dict[str, Any]],
    selection_mode: str,
    max_samples: int | None = None,
) -> list[tuple[int, dict[str, Any], list[str]]]:
    """Choose input-order-stable annotation targets from prediction outcomes."""
    selected: list[tuple[int, dict[str, Any], list[str]]] = []
    for input_index, row in enumerate(rows):
        key = sample_key(row)
        reasons = selection_reasons(
            row, primary_predictions.get(key), secondary_predictions.get(key)
        )
        has_disagreement = "prediction_disagreement" in reasons
        has_error = any(reason.endswith("_error") for reason in reasons)
        # When neither prediction file is supplied, audit the requested input split.
        no_predictions = not primary_predictions and not secondary_predictions
        include = {
            "all": True,
            "disagreements": has_disagreement,
            "errors": has_error,
            "disagreements_or_errors": has_disagreement or has_error or no_predictions,
        }[selection_mode]
        if include:
            selected.append((input_index, row, reasons))
        if max_samples is not None and len(selected) >= max_samples:
            break
    return selected


def temporal_operator(question: object) -> str:
    text = " ".join(str(question).split())
    if re.match(r"(?i)^(after|before)\s+.+?,", text):
        return text.split(None, 1)[0].upper()
    if re.search(r"(?i)\bafter\b", text):
        return "AFTER"
    if re.search(r"(?i)\bbefore\b", text):
        return "BEFORE"
    if re.search(r"(?i)\b(first|earliest|at the start|beginning)\b", text):
        return "FIRST"
    if re.search(r"(?i)\b(last|latest|finally|at the end|end of)\b", text):
        return "LAST"
    if re.search(r"(?i)\b(chang(?:e|ed)|different|compared|between)\b", text):
        return "STATE_CHANGE"
    return "GLOBAL"


def _metadata_fps(record: dict[str, Any]) -> float | None:
    for name in ("fps", "video_fps"):
        value = record.get(name)
        if value is not None and float(value) > 0:
            return float(value)
    return None


def _frame_item(item: object, fps: float | None, numeric_is_index: bool) -> dict[str, Any]:
    if isinstance(item, dict):
        timestamp = item.get("timestamp", item.get("time_seconds", item.get("time")))
        frame_index = item.get("frame_index", item.get("index"))
        if timestamp is None and frame_index is not None:
            if fps is None:
                raise RuntimeError("frame metadata has frame_index but no video FPS")
            timestamp = float(frame_index) / fps
        if timestamp is None:
            raise RuntimeError("frame metadata item is missing timestamp")
        frame = {"timestamp": round(float(timestamp), 6)}
        if frame_index is not None:
            frame["frame_index"] = int(frame_index)
        if item.get("source") is not None:
            frame["source"] = str(item["source"])
        return frame
    if numeric_is_index:
        if fps is None:
            raise RuntimeError("frame-index metadata has no video FPS")
        return {"timestamp": round(float(item) / fps, 6), "frame_index": int(item)}
    return {"timestamp": round(float(item), 6)}


def _frames_from_field(
    record: dict[str, Any], field: str, numeric_is_index: bool
) -> list[dict[str, Any]] | None:
    values = record.get(field)
    if not isinstance(values, list):
        return None
    fps = _metadata_fps(record)
    frames = [_frame_item(value, fps, numeric_is_index) for value in values]
    return sorted(frames, key=lambda frame: (frame["timestamp"], frame.get("frame_index", -1)))


def uniform_candidate_indices(total_frames: int, candidate_frames: int) -> list[int]:
    """Match the grounded proof-pack runner's deterministic candidate grid."""
    if total_frames <= 0 or candidate_frames <= 0:
        return []
    count = min(candidate_frames, total_frames)
    if count == 1:
        return [total_frames // 2]
    positions = [round(index * (total_frames - 1) / (count - 1)) for index in range(count)]
    return list(dict.fromkeys(max(0, min(int(index), total_frames - 1)) for index in positions))


def needs_candidate_grid_reconstruction(record: dict[str, Any]) -> bool:
    """Return true only for proof packs that retained count but not grid timestamps."""
    if not isinstance(record.get("candidate_frames"), int):
        return False
    explicit_fields = (
        "candidate_timestamps",
        "candidate_frame_timestamps",
        "candidate_frames_timestamps",
        "candidate_frame_indices",
        "candidate_indices",
        "candidates",
    )
    return not any(isinstance(record.get(field), list) for field in explicit_fields)


def frame_pack_frames(
    record: dict[str, Any],
    pack: str,
    video_fps: float | None = None,
    total_video_frames: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Read either generic candidate grids or established proof-pack metadata."""
    if pack == "candidate":
        fields = (
            ("candidate_timestamps", False),
            ("candidate_frame_timestamps", False),
            ("candidate_frames_timestamps", False),
            ("candidate_frame_indices", True),
            ("candidate_indices", True),
            ("candidates", False),
        )
    else:
        fields = (
            ("final_pack_timestamps", False),
            ("final_timestamps", False),
            ("selected", False),
            ("frame_indices", True),
        )
    for field, numeric_is_index in fields:
        frames = _frames_from_field(record, field, numeric_is_index)
        if frames is not None:
            return frames, field
    if pack == "candidate" and needs_candidate_grid_reconstruction(record):
        if video_fps is None or total_video_frames is None:
            raise RuntimeError("candidate-grid reconstruction requires video FPS and total frames")
        indices = uniform_candidate_indices(total_video_frames, int(record["candidate_frames"]))
        return (
            [
                {"frame_index": index, "timestamp": round(index / video_fps, 6)}
                for index in indices
            ],
            "reconstructed_uniform_candidate_grid",
        )
    if pack == "candidate":
        # Existing proof packs only persist their final selected frames.
        frames = _frames_from_field(record, "selected", False)
        if frames is not None:
            return frames, "selected_fallback"
    raise RuntimeError(f"{pack} metadata has no supported frame timestamp field")


def candidate_grid_video_metadata(
    row: dict[str, Any],
    candidate_record: dict[str, Any],
    final_record: dict[str, Any],
    video_folder: str | None,
) -> tuple[float, int]:
    """Find video geometry needed to recreate a stored uniform candidate count."""
    for record in (candidate_record, final_record):
        fps = _metadata_fps(record)
        total_frames = record.get("total_video_frames", record.get("total_frames"))
        if fps is not None and total_frames is not None:
            return fps, int(total_frames)
        duration = next(
            (
                float(record[field])
                for field in ("video_duration_seconds", "duration_seconds", "duration")
                if record.get(field) is not None
            ),
            None,
        )
        if fps is not None and duration is not None:
            return fps, round(fps * duration)
    if video_folder is None:
        raise RuntimeError(
            "proof-pack candidate_frames is a count; pass --video-folder or metadata with FPS and total frames"
        )
    import cv2

    video_path = str(row["video_path"])
    if not os.path.isabs(video_path):
        video_path = os.path.join(video_folder, video_path)
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return fps, total_frames


def video_duration_seconds(
    row: dict[str, Any],
    candidate_record: dict[str, Any],
    final_record: dict[str, Any],
    video_folder: str | None,
) -> float:
    for record in (final_record, candidate_record):
        for field in ("video_duration_seconds", "duration_seconds", "duration"):
            if record.get(field) is not None:
                return round(float(record[field]), 6)
        fps = _metadata_fps(record)
        total_frames = record.get("total_video_frames", record.get("total_frames"))
        if fps is not None and total_frames is not None:
            return round(float(total_frames) / fps, 6)
    if video_folder is None:
        raise RuntimeError("video duration is absent from metadata; pass --video-folder")
    import cv2

    video_path = str(row["video_path"])
    if not os.path.isabs(video_path):
        video_path = os.path.join(video_folder, video_path)
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return round(total_frames / fps, 6)


def manifest_fingerprint(
    input_annotations: str,
    candidate_metadata: str,
    final_pack_metadata: str | None,
    subset_file: str | None,
    primary_predictions: str | None,
    secondary_predictions: str | None,
    selection_mode: str,
    max_samples: int | None,
) -> str:
    return fingerprint(
        {
            "schema": MANIFEST_SCHEMA,
            "input_annotations_sha1": file_sha1(input_annotations),
            "candidate_metadata_sha1": file_sha1(candidate_metadata),
            "final_pack_metadata_sha1": file_sha1(final_pack_metadata),
            "subset_file_sha1": file_sha1(subset_file),
            "primary_predictions_sha1": file_sha1(primary_predictions),
            "secondary_predictions_sha1": file_sha1(secondary_predictions),
            "selection_mode": selection_mode,
            "max_samples": max_samples,
        }
    )


def build_manifest_records(
    selected_rows: list[tuple[int, dict[str, Any], list[str]]],
    candidate_metadata: dict[str, dict[str, Any]],
    final_pack_metadata: dict[str, dict[str, Any]],
    primary_predictions: dict[str, dict[str, Any]],
    secondary_predictions: dict[str, dict[str, Any]],
    audit_fingerprint: str,
    video_folder: str | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for input_index, row, reasons in selected_rows:
        key = sample_key(row)
        if key not in candidate_metadata:
            raise RuntimeError(f"candidate metadata is missing sample_key: {key}")
        if key not in final_pack_metadata:
            raise RuntimeError(f"final-pack metadata is missing sample_key: {key}")
        candidate_record = candidate_metadata[key]
        final_record = final_pack_metadata[key]
        video_fps = None
        total_video_frames = None
        if needs_candidate_grid_reconstruction(candidate_record):
            video_fps, total_video_frames = candidate_grid_video_metadata(
                row, candidate_record, final_record, video_folder
            )
        candidate_frames, candidate_source = frame_pack_frames(
            candidate_record, "candidate", video_fps, total_video_frames
        )
        final_frames, final_source = frame_pack_frames(final_record, "final")
        gold_answer = normalize_answer(row.get("mcq_answer"))
        records.append(
            {
                "schema": MANIFEST_SCHEMA,
                "audit_fingerprint": audit_fingerprint,
                "sample_key": key,
                "input_index": input_index,
                "video_path": str(row.get("video_path", "")),
                "video_duration_seconds": video_duration_seconds(
                    row, candidate_record, final_record, video_folder
                ),
                "question": str(row.get("question", "")),
                "mcq_options": str(row.get("mcq_options", "")),
                "gold_answer": gold_answer,
                "operator": temporal_operator(row.get("question", "")),
                "selection_reasons": reasons,
                "primary_prediction": prediction_summary(
                    primary_predictions.get(key), gold_answer
                ),
                "secondary_prediction": prediction_summary(
                    secondary_predictions.get(key), gold_answer
                ),
                "candidate_frame_source": candidate_source,
                "candidate_frames": candidate_frames,
                "final_pack_frame_source": final_source,
                "final_pack_frames": final_frames,
                "annotation_template": {
                    "sample_key": key,
                    "decisive_events": [],
                    "temporal_order": [],
                    "notes": "",
                },
            }
        )
    return records


def manifest_matches(
    existing: list[dict[str, Any]], expected: list[dict[str, Any]], audit_fingerprint: str
) -> bool:
    if len(existing) != len(expected):
        return False
    return all(
        actual.get("audit_fingerprint") == audit_fingerprint
        and actual.get("sample_key") == wanted.get("sample_key")
        and actual.get("input_index") == wanted.get("input_index")
        for actual, wanted in zip(existing, expected)
    )


def timestamp_covers_interval(
    timestamp: float, start_seconds: float, end_seconds: float, tolerance_seconds: float
) -> bool:
    return start_seconds - tolerance_seconds <= timestamp <= end_seconds + tolerance_seconds


def normalize_decisive_events(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    events = annotation.get("decisive_events", [])
    if not isinstance(events, list):
        raise RuntimeError("decisive_events must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        event_id = str(event["event_id"])
        start = float(event["start_seconds"])
        end = float(event["end_seconds"])
        if start > end:
            raise RuntimeError(f"event {event_id} has start_seconds after end_seconds")
        if event_id in seen:
            raise RuntimeError(f"duplicate decisive event_id: {event_id}")
        seen.add(event_id)
        normalized.append(
            {
                "event_id": event_id,
                "start_seconds": start,
                "end_seconds": end,
                "required": bool(event.get("required", True)),
            }
        )
    return normalized


def _ordered_events_covered(
    frame_timestamps: list[float],
    events_by_id: dict[str, dict[str, Any]],
    event_order: list[str],
    tolerance_seconds: float,
) -> bool:
    previous = float("-inf")
    for event_id in event_order:
        event = events_by_id[event_id]
        candidates = [
            timestamp
            for timestamp in frame_timestamps
            if timestamp > previous
            and timestamp_covers_interval(
                timestamp,
                event["start_seconds"],
                event["end_seconds"],
                tolerance_seconds,
            )
        ]
        if not candidates:
            return False
        previous = min(candidates)
    return True


def score_frame_timestamps(
    frame_timestamps: list[float],
    events: list[dict[str, Any]],
    temporal_order: list[str],
    tolerance_seconds: float,
) -> dict[str, Any]:
    required_events = [event for event in events if event["required"]]
    covered_event_ids = [
        event["event_id"]
        for event in required_events
        if any(
            timestamp_covers_interval(
                timestamp,
                event["start_seconds"],
                event["end_seconds"],
                tolerance_seconds,
            )
            for timestamp in frame_timestamps
        )
    ]
    event_ids = {event["event_id"] for event in events}
    if any(event_id not in event_ids for event_id in temporal_order):
        raise RuntimeError("temporal_order references an unknown event_id")
    order_required = len(temporal_order) >= 2
    return {
        "required_events": len(required_events),
        "covered_events": len(covered_event_ids),
        "covered_event_ids": covered_event_ids,
        "all_events_covered": len(covered_event_ids) == len(required_events),
        "temporal_order_required": order_required,
        "temporal_order_covered": (
            _ordered_events_covered(
                frame_timestamps,
                {event["event_id"]: event for event in events},
                temporal_order,
                tolerance_seconds,
            )
            if order_required
            else None
        ),
    }


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_events = sum(row["candidate"]["required_events"] for row in rows)
    candidate_covered = sum(row["candidate"]["covered_events"] for row in rows)
    final_covered = sum(row["final_pack"]["covered_events"] for row in rows)
    order_rows = [row for row in rows if row["candidate"]["temporal_order_required"]]
    return {
        "annotated_samples": len(rows),
        "required_decisive_events": total_events,
        "candidate_events_covered": candidate_covered,
        "final_pack_events_covered": final_covered,
        "candidate_recall": candidate_covered / total_events if total_events else 0.0,
        "final_pack_recall": final_covered / total_events if total_events else 0.0,
        "candidate_all_events_covered_rate": (
            sum(row["candidate"]["all_events_covered"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "final_pack_all_events_covered_rate": (
            sum(row["final_pack"]["all_events_covered"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "temporal_order_samples": len(order_rows),
        "candidate_temporal_order_covered_rate": (
            sum(row["candidate"]["temporal_order_covered"] for row in order_rows)
            / len(order_rows)
            if order_rows
            else 0.0
        ),
        "final_pack_temporal_order_covered_rate": (
            sum(row["final_pack"]["temporal_order_covered"] for row in order_rows)
            / len(order_rows)
            if order_rows
            else 0.0
        ),
    }


def score_annotations(
    manifest: list[dict[str, Any]], annotations: dict[str, dict[str, Any]], tolerance_seconds: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_by_key = index_by_sample_key(manifest, "annotation manifest")
    unknown = sorted(set(annotations) - set(manifest_by_key))
    if unknown:
        raise RuntimeError(f"annotations contain sample_key outside manifest: {unknown[0]}")
    per_sample: list[dict[str, Any]] = []
    for manifest_record in manifest:
        key = str(manifest_record["sample_key"])
        annotation = annotations.get(key)
        if annotation is None:
            continue
        events = normalize_decisive_events(annotation)
        if not events:
            continue
        order = [str(event_id) for event_id in annotation.get("temporal_order", [])]
        candidate = score_frame_timestamps(
            [float(frame["timestamp"]) for frame in manifest_record["candidate_frames"]],
            events,
            order,
            tolerance_seconds,
        )
        final_pack = score_frame_timestamps(
            [float(frame["timestamp"]) for frame in manifest_record["final_pack_frames"]],
            events,
            order,
            tolerance_seconds,
        )
        per_sample.append(
            {
                "sample_key": key,
                "operator": manifest_record["operator"],
                "candidate": candidate,
                "final_pack": final_pack,
            }
        )
    by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_sample:
        by_operator[row["operator"]].append(row)
    return per_sample, {
        "overall": summarize_scores(per_sample),
        "by_operator": {
            operator: summarize_scores(rows) for operator, rows in sorted(by_operator.items())
        },
    }


def _safe_artifact_name(sample_key_value: str) -> str:
    return hashlib.sha1(sample_key_value.encode("utf-8")).hexdigest()[:16]


def render_previews(
    manifest: list[dict[str, Any]],
    video_folder: str,
    artifact_dir: str,
    mode: str,
    preview_fps: float,
    preview_max_frames: int,
    thumbnail_width: int,
    contact_sheet_columns: int,
) -> None:
    """Optionally decode 1-FPS thumbnails/contact sheets for human labeling."""
    if mode == "none":
        return
    import cv2
    import numpy as np

    def write_contact_sheet(images: list[Any], output_path: Path) -> None:
        max_height = max(image.shape[0] for image in images)
        padded = [
            cv2.copyMakeBorder(
                image,
                0,
                max_height - image.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            for image in images
        ]
        rows = [
            np.hstack(padded[offset : offset + contact_sheet_columns])
            for offset in range(0, len(padded), contact_sheet_columns)
        ]
        cv2.imwrite(str(output_path), np.vstack(rows))

    for record in manifest:
        video_path = record["video_path"]
        if not os.path.isabs(video_path):
            video_path = os.path.join(video_folder, video_path)
        sample_dir = Path(artifact_dir) / "previews" / _safe_artifact_name(record["sample_key"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, round(fps / preview_fps))
            indices = list(range(0, total_frames, step))
            if preview_max_frames > 0:
                indices = indices[:preview_max_frames]
            contact_images: list[Any] = []
            page_index = 0
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read()
                if not ok:
                    continue
                height = max(1, round(image.shape[0] * thumbnail_width / image.shape[1]))
                thumb = cv2.resize(image, (thumbnail_width, height))
                cv2.putText(
                    thumb,
                    f"{index / fps:.1f}s",
                    (6, min(height - 6, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    thumb,
                    f"{index / fps:.1f}s",
                    (6, min(height - 6, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                if mode == "thumbnails":
                    cv2.imwrite(str(sample_dir / f"thumbnail_{index:08d}.jpg"), thumb)
                    continue
                contact_images.append(thumb)
                if len(contact_images) == contact_sheet_columns * contact_sheet_columns:
                    write_contact_sheet(
                        contact_images, sample_dir / f"contact_sheet_{page_index:03d}.jpg"
                    )
                    page_index += 1
                    contact_images = []
            if mode == "contact_sheets" and contact_images:
                blank = np.zeros((max(image.shape[0] for image in contact_images), thumbnail_width, 3), dtype=np.uint8)
                while len(contact_images) % contact_sheet_columns:
                    contact_images.append(blank)
                write_contact_sheet(
                    contact_images, sample_dir / f"contact_sheet_{page_index:03d}.jpg"
                )
        finally:
            cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--input-annotations",
        dest="input_annotations",
        required=True,
        help="LongQA evaluation JSONL with questions and gold answers.",
    )
    parser.add_argument(
        "--candidate-metadata",
        "--proofpack",
        dest="candidate_metadata",
        required=True,
        help="Candidate-grid or proof-pack metadata JSONL.",
    )
    parser.add_argument(
        "--final-pack-metadata",
        default=None,
        help="Optional final-pack JSONL; defaults to --candidate-metadata.",
    )
    parser.add_argument("--annotations", default=None, help="Manual decisive-event JSONL.")
    parser.add_argument("--primary-predictions", default=None)
    parser.add_argument("--secondary-predictions", default=None)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--video-folder", default=None)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--results-output", default=None)
    parser.add_argument(
        "--selection-mode",
        choices=["all", "disagreements", "errors", "disagreements_or_errors"],
        default="disagreements_or_errors",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tolerance-seconds", type=float, default=1.0)
    parser.add_argument(
        "--preview-mode", choices=["none", "thumbnails", "contact_sheets"], default="none"
    )
    parser.add_argument("--preview-fps", type=float, default=1.0)
    parser.add_argument("--preview-max-frames", type=int, default=0)
    parser.add_argument("--thumbnail-width", type=int, default=240)
    parser.add_argument("--contact-sheet-columns", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tolerance_seconds < 0:
        raise ValueError("--tolerance-seconds must be non-negative")
    if args.preview_fps <= 0:
        raise ValueError("--preview-fps must be positive")

    artifact_dir = os.path.abspath(args.artifact_dir)
    manifest_path = args.manifest_output or os.path.join(artifact_dir, "annotation_manifest.jsonl")
    results_path = args.results_output or os.path.join(artifact_dir, "recall_results.json")
    schema_path = os.path.join(artifact_dir, "annotation_schema.json")
    input_rows = apply_subset(load_jsonl(args.input_annotations), args.subset_file)
    candidate_rows = index_by_sample_key(load_jsonl(args.candidate_metadata), "candidate metadata")
    final_metadata_path = args.final_pack_metadata or args.candidate_metadata
    final_rows = index_by_sample_key(load_jsonl(final_metadata_path), "final-pack metadata")
    primary_rows = (
        index_by_sample_key(load_jsonl(args.primary_predictions), "primary predictions")
        if args.primary_predictions
        else {}
    )
    secondary_rows = (
        index_by_sample_key(load_jsonl(args.secondary_predictions), "secondary predictions")
        if args.secondary_predictions
        else {}
    )
    audit_id = manifest_fingerprint(
        args.input_annotations,
        args.candidate_metadata,
        final_metadata_path,
        args.subset_file,
        args.primary_predictions,
        args.secondary_predictions,
        args.selection_mode,
        args.max_samples,
    )
    selected = select_rows_for_annotation(
        input_rows,
        primary_rows,
        secondary_rows,
        args.selection_mode,
        args.max_samples,
    )
    expected_manifest = build_manifest_records(
        selected,
        candidate_rows,
        final_rows,
        primary_rows,
        secondary_rows,
        audit_id,
        args.video_folder,
    )
    if os.path.exists(manifest_path):
        manifest = load_jsonl(manifest_path)
        if not manifest_matches(manifest, expected_manifest, audit_id):
            raise RuntimeError(
                "existing manifest does not match this audit fingerprint; use a new "
                "artifact directory to avoid mixing manual labels"
            )
        print(f"Reusing matching annotation manifest: {manifest_path}")
    else:
        manifest = expected_manifest
        write_jsonl(manifest_path, manifest)
        print(f"Annotation manifest written to {manifest_path}")
    write_json(schema_path, ANNOTATION_SCHEMA)
    if args.preview_mode != "none":
        if args.video_folder is None:
            raise RuntimeError("--video-folder is required for preview rendering")
        render_previews(
            manifest,
            args.video_folder,
            artifact_dir,
            args.preview_mode,
            args.preview_fps,
            args.preview_max_frames,
            args.thumbnail_width,
            args.contact_sheet_columns,
        )

    if args.annotations is None:
        print(
            f"Selected {len(manifest)} samples for manual annotation; no event labels were created. "
            f"Schema: {schema_path}"
        )
        return

    annotations = index_by_sample_key(load_jsonl(args.annotations), "manual annotations")
    per_sample, scores = score_annotations(manifest, annotations, args.tolerance_seconds)
    results_fingerprint = fingerprint(
        {
            "schema": RESULTS_SCHEMA,
            "audit_fingerprint": audit_id,
            "annotations_sha1": file_sha1(args.annotations),
            "tolerance_seconds": args.tolerance_seconds,
        }
    )
    results = {
        "schema": RESULTS_SCHEMA,
        "results_fingerprint": results_fingerprint,
        "audit_fingerprint": audit_id,
        "tolerance_seconds": args.tolerance_seconds,
        "annotation_schema": ANNOTATION_SCHEMA,
        **scores,
        "per_sample": per_sample,
    }
    if os.path.exists(results_path):
        existing = json.loads(Path(results_path).read_text())
        if existing.get("results_fingerprint") == results_fingerprint:
            print(f"Reusing matching recall results: {results_path}")
            return
    write_json(results_path, results)
    print(json.dumps(scores["overall"], indent=2, sort_keys=True))
    print(f"Recall results written to {results_path}")


if __name__ == "__main__":
    main()
