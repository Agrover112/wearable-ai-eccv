#!/usr/bin/env python3
"""Build a cached semantic event ledger and use it as LongQA retrieval evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from longqa_utils import (
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    index_row_aligned_metadata,
    parse_mcq_options,
    sample_key,
)
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices, load_jsonl
from run_generate_longqa_narrative_gate import build_final_indices
from run_generate_longqa_openqa import MiniLMTextEncoder
from run_generate_longqa_proofpack import (
    baseline_uniform_indices,
    resize_frame_to_max_pixels,
)


EVENT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "scene": {"type": "string", "maxLength": 120},
        "entities": {
            "type": "array",
            "items": {"type": "string", "maxLength": 80},
            "maxItems": 6,
        },
        "actions": {
            "type": "array",
            "items": {"type": "string", "maxLength": 80},
            "maxItems": 6,
        },
        "attributes": {
            "type": "array",
            "items": {"type": "string", "maxLength": 80},
            "maxItems": 6,
        },
        "relations": {
            "type": "array",
            "items": {"type": "string", "maxLength": 80},
            "maxItems": 6,
        },
        "ocr": {
            "type": "array",
            "items": {"type": "string", "maxLength": 80},
            "maxItems": 6,
        },
        "state_changes": {
            "type": "array",
            "items": {"type": "string", "maxLength": 80},
            "maxItems": 3,
        },
    },
    "required": [
        "scene",
        "entities",
        "actions",
        "attributes",
        "relations",
        "ocr",
        "state_changes",
    ],
    "additionalProperties": False,
}

EVENT_PROMPT = (
    "Describe only what is visibly supported in this single egocentric video frame. "
    "Create a compact semantic event record. Use at most six short phrases per field. "
    "Prefer only the most salient evidence. Entities should name "
    "visible people, objects, and places; actions should describe visible interactions; "
    "attributes should include colors, materials, quantities, or object states; relations "
    "should describe spatial or interaction relations; OCR should contain readable text; "
    "state_changes should be empty unless a change is directly inferable. Do not guess "
    "events outside this frame and do not answer any question."
)

STOPWORDS = {
    "a", "an", "and", "after", "before", "did", "do", "does", "during", "for",
    "from", "had", "has", "have", "how", "i", "in", "into", "is", "it", "later",
    "me", "my", "of", "on", "or", "that", "the", "then", "there", "this", "to",
    "was", "were", "what", "when", "where", "which", "while", "with",
}


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _video_metadata(video_path: str) -> tuple[float, int]:
    import cv2

    capture = cv2.VideoCapture(video_path)
    try:
        return float(capture.get(cv2.CAP_PROP_FPS)), int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
    finally:
        capture.release()


def normalize_event_record(raw: dict[str, object]) -> dict[str, object]:
    record: dict[str, object] = {"scene": str(raw.get("scene", "")).strip()}
    for field in (
        "entities",
        "actions",
        "attributes",
        "relations",
        "ocr",
        "state_changes",
    ):
        value = raw.get(field, [])
        if not isinstance(value, list):
            value = [value]
        record[field] = [str(item).strip() for item in value if str(item).strip()]
    return record


def event_record_text(record: dict[str, object]) -> str:
    parts = [str(record.get("scene", ""))]
    for field in ("entities", "actions", "attributes", "relations", "ocr", "state_changes"):
        values = record.get(field, [])
        if isinstance(values, list) and values:
            parts.append(f"{field}: " + "; ".join(str(item) for item in values))
    return " | ".join(part for part in parts if part)


def extract_concept_phrases(row: dict[str, Any], limit: int = 16) -> list[str]:
    options = parse_mcq_options(row.get("mcq_options", ""))
    phrases: list[str] = []
    for text in [str(row.get("question", "")), *options.values()]:
        clean = re.sub(r"[^A-Za-z0-9' -]+", " ", text).strip().lower()
        for clause in re.split(r"\b(?:and|or|with|after|before|near|beside|at|in|on)\b", clean):
            words = [word for word in clause.split() if word not in STOPWORDS]
            if 1 <= len(words) <= 6:
                phrases.append(" ".join(words))
            for word in words:
                if len(word) >= 4:
                    phrases.append(word)
    unique: list[str] = []
    for phrase in phrases:
        if phrase and phrase not in unique:
            unique.append(phrase)
        if len(unique) == limit:
            break
    return unique


def _cache_path(cache_dir: str, fingerprint: str, video_path: str, frame_index: int) -> Path:
    video_key = hashlib.sha1(video_path.encode()).hexdigest()[:16]
    return Path(cache_dir) / fingerprint / video_key / f"{frame_index}.json"


def select_ledger_entries(
    row: dict[str, Any],
    entries: list[dict[str, Any]],
    encoder: MiniLMTextEncoder,
    count: int,
) -> tuple[list[dict[str, Any]], list[float]]:
    query = f"{row['question']}\n{row['mcq_options']}"
    texts = [event_record_text(entry["record"]) for entry in entries]
    embeddings = encoder.encode([query] + texts)
    scores = embeddings[1:] @ embeddings[0]
    order = np.argsort(-scores)[: min(count, len(entries))]
    return [entries[int(index)] for index in order], [float(scores[index]) for index in order]


def build_ledger_answer_prompt(
    row: dict[str, Any], selected: list[dict[str, Any]], detections: dict[int, list[str]] | None = None
) -> str:
    notes = []
    detections = detections or {}
    for entry in sorted(selected, key=lambda item: float(item["timestamp"])):
        detected = detections.get(int(entry["frame_index"]), [])
        suffix = f" | detected: {', '.join(detected)}" if detected else ""
        notes.append(
            f"- {float(entry['timestamp']):.1f}s: {event_record_text(entry['record'])}{suffix}"
        )
    return (
        "The images are chronological. The semantic event ledger below is a retrieval "
        "aid generated independently of the question. Trust visible image evidence over "
        "a ledger entry if they conflict. Internally identify the relevant event and "
        "temporal neighborhood, then answer the multiple-choice question.\n\n"
        "Event ledger:\n"
        + "\n".join(notes)
        + "\n\n"
        + build_longqa_prompt(row["question"], row["mcq_options"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl")
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--proofpack", required=True)
    parser.add_argument("--proofpack-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ledger-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--ledger-cache-dir", required=True)
    parser.add_argument("--ledger-frames", type=int, default=16)
    parser.add_argument("--ledger-retrieval", type=int, default=8)
    parser.add_argument("--visual-quota", type=int, default=44)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--ledger-max-pixels", type=int, default=200704)
    parser.add_argument("--text-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    import time

    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    ledger_output = _resolve_path(args.ledger_output)
    eval_output = _resolve_path(args.eval_output)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    reference_rows = load_jsonl(_resolve_path(args.proofpack_reference))
    packs = index_row_aligned_metadata(
        load_jsonl(_resolve_path(args.proofpack)), reference_rows, "proof pack"
    )
    payload = {
        "schema": EVENT_SCHEMA,
        "ledger_frames": args.ledger_frames,
        "ledger_retrieval": args.ledger_retrieval,
        "visual_quota": args.visual_quota,
        "max_frames": args.max_frames,
        "ledger_max_pixels": args.ledger_max_pixels,
        "model": args.llm_model,
    }
    fingerprint = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    cache_fingerprint = hashlib.sha1(
        json.dumps(
            {"schema": EVENT_SCHEMA, "model": args.llm_model, "pixels": args.ledger_max_pixels},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(ledger_output) or ".", exist_ok=True)
    existing = [] if args.no_resume or not os.path.exists(output_path) else load_jsonl(output_path)
    ledger_existing = (
        [] if args.no_resume or not os.path.exists(ledger_output) else load_jsonl(ledger_output)
    )
    start = 0
    for index, (prediction, ledger_row) in enumerate(
        zip(existing[: len(rows)], ledger_existing[: len(rows)])
    ):
        if (
            sample_key(prediction) != sample_key(rows[index])
            or prediction.get("event_ledger_fingerprint") != fingerprint
            or str(ledger_row.get("sample_key", "")) != sample_key(rows[index])
            or ledger_row.get("event_ledger_fingerprint") != fingerprint
        ):
            break
        start += 1
    if len(existing) != start or len(ledger_existing) != start:
        with open(output_path, "w") as handle:
            for item in existing[:start]:
                handle.write(json.dumps(item) + "\n")
        with open(ledger_output, "w") as handle:
            for item in ledger_existing[:start]:
                handle.write(json.dumps(item) + "\n")

    encoder = MiniLMTextEncoder(args.text_encoder, device="cpu", batch_size=64)
    model = VLLMModel(
        args.llm_model,
        tp_size=1,
        concurrency=args.concurrency,
        max_frames=args.max_frames,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    begun = time.time()
    mode = "a" if start else "w"
    with model, open(output_path, mode) as prediction_handle, open(
        ledger_output, mode
    ) as ledger_handle:
        for row_index, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            if key not in packs:
                raise RuntimeError(f"Proof pack is missing sample: {key}")
            video_path = os.path.join(video_folder, str(row["video_path"]))
            fps, total_frames = _video_metadata(video_path)
            frame_indices = baseline_uniform_indices(total_frames, args.ledger_frames)
            entries: list[dict[str, Any] | None] = [None] * len(frame_indices)
            missing_positions: list[int] = []
            missing_frames: list[list[object]] = []
            missing_messages: list[list[dict[str, str]]] = []
            for position, frame_index in enumerate(frame_indices):
                cache_path = _cache_path(
                    args.ledger_cache_dir,
                    cache_fingerprint,
                    str(row["video_path"]),
                    frame_index,
                )
                if cache_path.exists():
                    cached = json.loads(cache_path.read_text())
                    entries[position] = cached
                    continue
                frame = extract_frames_by_indices(video_path, [frame_index])[0]
                frame = resize_frame_to_max_pixels(frame, args.ledger_max_pixels)
                missing_positions.append(position)
                missing_frames.append([frame])
                missing_messages.append([{"role": "user", "content": EVENT_PROMPT}])
            if missing_frames:
                generated = model.generate_json_batch(
                    missing_frames,
                    missing_messages,
                    EVENT_SCHEMA,
                    schema_name="semantic_event_record",
                    max_new_tokens=256,
                )
                for position, raw in zip(missing_positions, generated):
                    frame_index = frame_indices[position]
                    entry = {
                        "frame_index": frame_index,
                        "timestamp": frame_index / max(fps, 1e-6),
                        "record": normalize_event_record(raw),
                    }
                    cache_path = _cache_path(
                        args.ledger_cache_dir,
                        cache_fingerprint,
                        str(row["video_path"]),
                        frame_index,
                    )
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
                    temporary.write_text(json.dumps(entry) + "\n")
                    os.replace(temporary, cache_path)
                    entries[position] = entry
            complete_entries = [entry for entry in entries if entry is not None]
            if len(complete_entries) != len(frame_indices):
                raise RuntimeError("Event ledger did not produce one record per frame")
            selected, selected_scores = select_ledger_entries(
                row, complete_entries, encoder, args.ledger_retrieval
            )
            selected_indices = [int(entry["frame_index"]) for entry in selected]
            final_indices = build_final_indices(
                packs[key], selected_indices, total_frames, args.max_frames, args.visual_quota
            )
            final_frames = extract_frames_by_indices(video_path, final_indices)
            answer_prompt = build_ledger_answer_prompt(row, selected)
            response = model.generate(
                final_frames,
                [{"role": "user", "content": answer_prompt}],
                max_new_tokens=16,
            )
            prediction = build_prediction_row(row, response, prompt_variant="event_ledger")
            prediction.update(
                {
                    "event_ledger_fingerprint": fingerprint,
                    "event_ledger_frames": len(complete_entries),
                    "event_ledger_selected_indices": selected_indices,
                    "event_ledger_final_frames": len(final_frames),
                }
            )
            ledger_row = {
                "sample_key": key,
                "video_path": row.get("video_path", ""),
                "event_ledger_fingerprint": fingerprint,
                "cache_fingerprint": cache_fingerprint,
                "schema_complete": len(complete_entries) == args.ledger_frames,
                "concepts": extract_concept_phrases(row),
                "entries": complete_entries,
                "selected_indices": selected_indices,
                "selected_scores": selected_scores,
                "final_indices": final_indices,
            }
            prediction_handle.write(json.dumps(prediction) + "\n")
            ledger_handle.write(json.dumps(ledger_row) + "\n")
            prediction_handle.flush()
            ledger_handle.flush()
            print(f"  Event-ledger progress: {row_index + 1}/{len(rows)}")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
