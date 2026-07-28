#!/usr/bin/env python3
"""Augment a cached LongQA event ledger with concept-conditioned detections."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from longqa_utils import apply_subset, build_prediction_row, index_row_aligned_metadata, sample_key
from run_generate_longqa_event_ledger import (
    build_ledger_answer_prompt,
    select_ledger_entries,
)
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices, load_jsonl
from run_generate_longqa_narrative_gate import build_final_indices
from run_generate_longqa_openqa import MiniLMTextEncoder


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def normalize_detector_result(result: dict[str, object]) -> list[dict[str, object]]:
    labels = result.get("text_labels", result.get("labels", []))
    scores = result.get("scores", [])
    boxes = result.get("boxes", [])
    normalized = []
    for label, score, box in zip(labels, scores, boxes):
        label_text = str(label).strip()
        if label_text:
            normalized.append(
                {
                    "label": label_text,
                    "score": round(float(score), 6),
                    "box": [round(float(value), 2) for value in box],
                }
            )
    return normalized


class OpenVocabularyDetector:
    def __init__(self, model_id: str, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Grounding DINO's text-enhancer path promotes some activations to FP32;
        # loading the small detector in BF16 then causes mixed-dtype linear ops.
        dtype = torch.float32
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, dtype=dtype
        ).to(self.device).eval()
        self.dtype = dtype

    def detect(
        self,
        image: object,
        concepts: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict[str, object]]:
        return self.detect_batch(
            [image], concepts, box_threshold, text_threshold
        )[0]

    def detect_batch(
        self,
        images: list[object],
        concepts: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[list[dict[str, object]]]:
        if not images:
            return []
        if not concepts:
            return [[] for _ in images]
        prompt = ". ".join(concepts) + "."
        inputs = self.processor(
            images=images,
            text=[prompt] * len(images),
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: (
                value.to(device=self.device, dtype=self.dtype)
                if value.is_floating_point()
                else value.to(self.device)
            )
            for key, value in inputs.items()
        }
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        kwargs = {
            "threshold": box_threshold,
            "text_threshold": text_threshold,
            "target_sizes": [image.size[::-1] for image in images],
        }
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.get("input_ids"), **kwargs
            )
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs, **kwargs
            )
        normalized = []
        for result in results:
            cpu_result = {
                key: (
                    value.detach().float().cpu().tolist()
                    if hasattr(value, "detach")
                    else value
                )
                for key, value in result.items()
            }
            normalized.append(normalize_detector_result(cpu_result))
        return normalized


def detection_cache_path(
    cache_dir: str,
    model_id: str,
    video_path: str,
    frame_index: int,
    concepts: list[str],
    box_threshold: float,
    text_threshold: float,
) -> Path:
    payload = json.dumps(
        {
            "model": model_id,
            "video": video_path,
            "frame": frame_index,
            "concepts": concepts,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
        },
        sort_keys=True,
    )
    return Path(cache_dir) / hashlib.sha1(payload.encode()).hexdigest()[:2] / (
        hashlib.sha1(payload.encode()).hexdigest() + ".json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl")
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--ledger-input", required=True)
    parser.add_argument("--proofpack", required=True)
    parser.add_argument("--proofpack-reference", required=True)
    parser.add_argument("--detection-cache-dir", required=True)
    parser.add_argument("--detections-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--detector-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--ledger-retrieval", type=int, default=8)
    parser.add_argument("--visual-quota", type=int, default=44)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--text-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    import time

    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary
    from run_generate_longqa_event_ledger import _video_metadata

    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    detections_output = _resolve_path(args.detections_output)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    row_by_key = {sample_key(row): row for row in rows}
    ledgers = {str(item["sample_key"]): item for item in load_jsonl(_resolve_path(args.ledger_input))}
    missing = sorted(set(row_by_key) - set(ledgers))
    if missing:
        raise RuntimeError(f"Ledger input is missing {len(missing)} subset samples")
    reference = load_jsonl(_resolve_path(args.proofpack_reference))
    packs = index_row_aligned_metadata(
        load_jsonl(_resolve_path(args.proofpack)), reference, "proof pack"
    )
    os.makedirs(os.path.dirname(detections_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fingerprint_payload = {
        "ledger": os.path.abspath(_resolve_path(args.ledger_input)),
        "proofpack": os.path.abspath(_resolve_path(args.proofpack)),
        "detector_model": args.detector_model,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "ledger_retrieval": args.ledger_retrieval,
        "visual_quota": args.visual_quota,
        "max_frames": args.max_frames,
        "llm_model": args.llm_model,
    }
    fingerprint = hashlib.sha1(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()[:12]

    existing_detection_rows = (
        {str(item["sample_key"]): item for item in load_jsonl(detections_output)}
        if os.path.exists(detections_output)
        else {}
    )
    begun = time.time()
    pending_detection_rows = [
        row for row in rows if sample_key(row) not in existing_detection_rows
    ]
    if pending_detection_rows:
        detector = OpenVocabularyDetector(args.detector_model)
        for row_index, row in enumerate(pending_detection_rows):
            key = sample_key(row)
            ledger = ledgers[key]
            concepts = [str(item) for item in ledger.get("concepts", [])]
            video_path = os.path.join(video_folder, str(row["video_path"]))
            frame_indices = [int(entry["frame_index"]) for entry in ledger["entries"]]
            frames = extract_frames_by_indices(video_path, frame_indices)
            per_frame = []
            for frame_index, frame in zip(frame_indices, frames):
                cache_path = detection_cache_path(
                    args.detection_cache_dir,
                    args.detector_model,
                    str(row["video_path"]),
                    frame_index,
                    concepts,
                    args.box_threshold,
                    args.text_threshold,
                )
                if cache_path.exists():
                    detections = json.loads(cache_path.read_text())["detections"]
                else:
                    detections = detector.detect(
                        frame, concepts, args.box_threshold, args.text_threshold
                    )
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
                    temporary.write_text(json.dumps({"detections": detections}) + "\n")
                    os.replace(temporary, cache_path)
                per_frame.append({"frame_index": frame_index, "detections": detections})
            record = {
                "sample_key": key,
                "video_path": row.get("video_path", ""),
                "concepts": concepts,
                "detector_model": args.detector_model,
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "frames": per_frame,
            }
            with open(detections_output, "a") as handle:
                handle.write(json.dumps(record) + "\n")
            existing_detection_rows[key] = record
            print(f"  Detection progress: {row_index + 1}/{len(pending_detection_rows)}")
        del detector
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    existing_predictions = load_jsonl(output_path) if os.path.exists(output_path) else []
    start = 0
    for index, prediction in enumerate(existing_predictions[: len(rows)]):
        if sample_key(prediction) != sample_key(rows[index]):
            break
        if prediction.get("event_ledger_detector_fingerprint") != fingerprint:
            break
        start += 1
    if len(existing_predictions) != start:
        with open(output_path, "w") as handle:
            for item in existing_predictions[:start]:
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
    with model, open(output_path, "a" if start else "w") as handle:
        for row_index, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            ledger = ledgers[key]
            selected, selected_scores = select_ledger_entries(
                row, ledger["entries"], encoder, args.ledger_retrieval
            )
            selected_indices = [int(entry["frame_index"]) for entry in selected]
            detector_row = existing_detection_rows[key]
            label_map = {
                int(item["frame_index"]): list(dict.fromkeys(
                    str(detection["label"]) for detection in item["detections"]
                ))
                for item in detector_row["frames"]
            }
            video_path = os.path.join(video_folder, str(row["video_path"]))
            _, total_frames = _video_metadata(video_path)
            final_indices = build_final_indices(
                packs[key], selected_indices, total_frames, args.max_frames, args.visual_quota
            )
            frames = extract_frames_by_indices(video_path, final_indices)
            response = model.generate(
                frames,
                [{"role": "user", "content": build_ledger_answer_prompt(row, selected, label_map)}],
                max_new_tokens=16,
            )
            prediction = build_prediction_row(
                row, response, prompt_variant="event_ledger_open_vocab_detection"
            )
            prediction.update(
                {
                    "event_ledger_selected_indices": selected_indices,
                    "event_ledger_selected_scores": selected_scores,
                    "event_ledger_final_frames": len(frames),
                    "detector_model": args.detector_model,
                    "event_ledger_detector_fingerprint": fingerprint,
                }
            )
            handle.write(json.dumps(prediction) + "\n")
            handle.flush()
            print(f"  Answer progress: {row_index + 1}/{len(rows)}")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    if not args.no_eval:
        _run_eval(input_path, output_path, _resolve_path(args.eval_output))


if __name__ == "__main__":
    main()
