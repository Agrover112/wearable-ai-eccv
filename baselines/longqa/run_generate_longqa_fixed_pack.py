#!/usr/bin/env python3
"""Run Qwen on either baseline-uniform or saved proof-pack frame indices.

This runner deliberately does no retrieval.  It is for clean comparisons where
the only changed variable is the final VLM (or the already-saved frame pack).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from typing import Any

from longqa_utils import (
    PROMPT_VARIANTS,
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    sample_key,
)
from run_generate_longqa_grounded import _run_eval, extract_frames_by_indices

logger = logging.getLogger(__name__)

FRAME_PACK_SCHEMA = 1
DEFAULT_FRAME_COUNT = 64
DEFAULT_MODEL_TYPE = "qwen"
DEFAULT_LLM_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_BACKEND = "vllm"


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_jsonl_if_exists(path: str) -> list[dict[str, Any]]:
    return load_jsonl(path) if os.path.exists(path) else []


def uniform_frame_indices(total_frames: int, frame_count: int = DEFAULT_FRAME_COUNT) -> list[int]:
    """Match ``model.extract_frames`` for one full-video interval exactly."""
    if total_frames <= 0 or frame_count <= 0:
        return []
    count = min(frame_count, total_frames)
    end_frame = total_frames - 1
    step = end_frame / count
    return sorted({int(index * step) for index in range(count)})


def _file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_proofpack_indices(
    proofpack_path: str, rows: list[dict[str, Any]]
) -> list[list[int]]:
    """Load a saved proof-pack JSONL while preserving its row order exactly."""
    proofpack_rows = load_jsonl(proofpack_path)
    if len(proofpack_rows) != len(rows):
        raise RuntimeError(
            f"proofpack has {len(proofpack_rows)} rows but input has {len(rows)} rows"
        )

    all_indices: list[list[int]] = []
    for index, (proofpack, row) in enumerate(zip(proofpack_rows, rows)):
        if int(proofpack.get("index", -1)) != index:
            raise RuntimeError(f"proofpack index mismatch at row {index}")
        if str(proofpack.get("video_path", "")) != str(row.get("video_path", "")):
            raise RuntimeError(f"proofpack video mismatch at row {index}")
        stored_key = str(proofpack.get("sample_key", ""))
        if stored_key and stored_key != sample_key(row):
            raise RuntimeError(f"proofpack sample-key mismatch at row {index}")
        selected = proofpack.get("selected")
        if not isinstance(selected, list):
            raise RuntimeError(f"proofpack selected-frame list missing at row {index}")
        indices = [int(item["frame_index"]) for item in selected]
        if len(indices) != len(set(indices)):
            raise RuntimeError(f"proofpack has duplicate frame indices at row {index}")
        all_indices.append(indices)
    return all_indices


def frame_pack_fingerprint(args: argparse.Namespace, proofpack_sha1: str | None) -> str:
    payload = {
        "schema": FRAME_PACK_SCHEMA,
        "frame_source": args.frame_source,
        "frame_count": args.frame_count,
        "uniform_formula": "extract_frames_single_full_interval_v1",
        "proofpack_sha1": proofpack_sha1,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def inference_fingerprint(args: argparse.Namespace, pack_fingerprint: str) -> str:
    payload = {
        "frame_pack": pack_fingerprint,
        "model_type": args.model_type,
        "llm_model": args.llm_model,
        "backend": args.backend,
        "tp": args.tp,
        "concurrency": args.concurrency,
        "prompt_variant": args.prompt_variant,
        "max_new_tokens": args.max_new_tokens,
        "qwen_min_pixels": os.environ.get("QWEN_MIN_PIXELS", "784"),
        "qwen_max_pixels": os.environ.get("QWEN_MAX_PIXELS", "50176"),
        "vllm_max_model_len": os.environ.get("VLLM_QWEN_MAX_MODEL_LEN", "16384"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def _video_metadata(video_path: str) -> tuple[float, int]:
    import cv2

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


def _proofpack_selected_metadata(
    proofpack_row: dict[str, Any], fps: float
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in proofpack_row["selected"]:
        frame_index = int(item["frame_index"])
        selected.append(
            {
                "frame_index": frame_index,
                "timestamp": round(frame_index / fps, 3),
                "source": str(item.get("source", "proofpack")),
                "source_score": item.get("score"),
            }
        )
    return selected


def build_frame_pack_record(
    index: int,
    row: dict[str, Any],
    video_folder: str,
    frame_source: str,
    frame_count: int,
    pack_fingerprint: str,
    proofpack_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    video_relpath = str(row["video_path"])
    video_path = os.path.join(video_folder, video_relpath)
    fps, total_frames = _video_metadata(video_path)
    if frame_source == "uniform":
        frame_indices = uniform_frame_indices(total_frames, frame_count)
        selected = [
            {
                "frame_index": frame_index,
                "timestamp": round(frame_index / fps, 3),
                "source": "uniform",
                "source_score": None,
            }
            for frame_index in frame_indices
        ]
    else:
        if proofpack_row is None:
            raise RuntimeError("proofpack row is required for proofpack frame source")
        frame_indices = [int(item["frame_index"]) for item in proofpack_row["selected"]]
        if any(frame_index < 0 or frame_index >= total_frames for frame_index in frame_indices):
            raise RuntimeError(f"proofpack frame index is outside video bounds at row {index}")
        selected = _proofpack_selected_metadata(proofpack_row, fps)

    return {
        "schema": FRAME_PACK_SCHEMA,
        "index": index,
        "sample_key": sample_key(row),
        "video_path": video_relpath,
        "frame_source": frame_source,
        "frame_count": len(frame_indices),
        "frame_indices": frame_indices,
        "fps": fps,
        "total_frames": total_frames,
        "frame_pack_fingerprint": pack_fingerprint,
        "source_proofpack_fingerprint": (
            str(proofpack_row.get("proofpack_fingerprint", ""))
            if proofpack_row is not None
            else None
        ),
        "selected": selected,
    }


def _resume_prefix(
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    fingerprint_field: str,
    fingerprint: str,
    answer_required: bool = False,
) -> int:
    count = 0
    for index, record in enumerate(records[: len(rows)]):
        if int(record.get("index", index)) != index:
            break
        if str(record.get("video_path", "")) != str(rows[index].get("video_path", "")):
            break
        record_key = str(record.get("sample_key") or sample_key(record))
        if record_key != sample_key(rows[index]):
            break
        if str(record.get(fingerprint_field, "")) != fingerprint:
            break
        if answer_required and not str(record.get("mcq_answer", "")).strip():
            break
        count += 1
    return count


def _rewrite_prefix(path: str, records: list[dict[str, Any]], count: int) -> None:
    with open(path, "w") as handle:
        for record in records[:count]:
            handle.write(json.dumps(record) + "\n")


def _write_frame_packs(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    video_folder: str,
    frame_pack_output: str,
    pack_fingerprint: str,
    proofpack_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    cached = [] if args.no_resume_frame_packs else load_jsonl_if_exists(frame_pack_output)
    start = _resume_prefix(cached, rows, "frame_pack_fingerprint", pack_fingerprint)
    if len(cached) != start:
        _rewrite_prefix(frame_pack_output, cached, start)
    if start:
        print(f"Resuming frame packs from {start}/{len(rows)} rows")

    records = cached[:start]
    mode = "a" if start else "w"
    with open(frame_pack_output, mode) as handle:
        for index, row in enumerate(rows[start:], start=start):
            proofpack_row = proofpack_rows[index] if proofpack_rows is not None else None
            record = build_frame_pack_record(
                index,
                row,
                video_folder,
                args.frame_source,
                args.frame_count,
                pack_fingerprint,
                proofpack_row,
            )
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"  Frame-pack progress: {index + 1}/{len(rows)}")
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LongQA predictions from deterministic or saved frame packs."
    )
    parser.add_argument(
        "--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-pack-output", default=None)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-resume-frame-packs", action="store_true")
    parser.add_argument("--no-resume-predictions", action="store_true")

    parser.add_argument("--frame-source", choices=["uniform", "proofpack"], default="uniform")
    parser.add_argument("--proofpack", default=None)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="baseline")

    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE, choices=["qwen"])
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=["hf", "vllm"])
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args(argv)
    if args.frame_source == "proofpack" and not args.proofpack:
        parser.error("--proofpack is required when --frame-source proofpack")
    return args


def main() -> None:
    from model import create_model, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    input_path = _resolve_path(args.input)
    video_folder = _resolve_path(args.video_folder)
    output_path = _resolve_path(args.output)
    frame_pack_output = _resolve_path(
        args.frame_pack_output
        or os.path.join(os.path.dirname(output_path), "frame_packs.jsonl")
    )
    eval_output = _resolve_path(args.eval_output) if args.eval_output else None
    proofpack_path = _resolve_path(args.proofpack) if args.proofpack else None

    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    proofpack_rows = load_jsonl(proofpack_path) if proofpack_path else None
    proofpack_indices = (
        load_proofpack_indices(proofpack_path, rows) if proofpack_path else None
    )
    if proofpack_rows is not None and proofpack_indices is not None:
        for proofpack_row, indices in zip(proofpack_rows, proofpack_indices):
            if [int(item["frame_index"]) for item in proofpack_row["selected"]] != indices:
                raise RuntimeError("proofpack index loading changed saved frame order")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(frame_pack_output) or ".", exist_ok=True)
    proofpack_sha1 = _file_sha1(proofpack_path) if proofpack_path else None
    pack_fingerprint = frame_pack_fingerprint(args, proofpack_sha1)
    generation_fingerprint = inference_fingerprint(args, pack_fingerprint)
    print(
        f"Fixed-pack config: source={args.frame_source}, frame_count={args.frame_count}, "
        f"pack_fingerprint={pack_fingerprint}, inference_fingerprint={generation_fingerprint}"
    )

    frame_packs = _write_frame_packs(
        args,
        rows,
        video_folder,
        frame_pack_output,
        pack_fingerprint,
        proofpack_rows,
    )
    existing = [] if args.no_resume_predictions else load_jsonl_if_exists(output_path)
    start = _resume_prefix(
        existing,
        rows,
        "fixed_pack_inference_fingerprint",
        generation_fingerprint,
        answer_required=True,
    )
    if len(existing) != start:
        _rewrite_prefix(output_path, existing, start)
    if start:
        print(f"Resuming predictions from {start}/{len(rows)} rows")

    if start < len(rows):
        max_frames = max(pack["frame_count"] for pack in frame_packs)
        model = create_model(
            args.model_type,
            args.llm_model,
            backend=args.backend,
            tp_size=args.tp,
            concurrency=args.concurrency,
            max_frames=max_frames,
        )
        reset_prompt_token_stats()
        mode = "a" if start else "w"
        with model, open(output_path, mode) as handle:
            for index, (row, pack) in enumerate(
                zip(rows[start:], frame_packs[start:]), start=start
            ):
                video_path = os.path.join(video_folder, str(row["video_path"]))
                frame_indices = [int(frame_index) for frame_index in pack["frame_indices"]]
                frames = extract_frames_by_indices(video_path, frame_indices)
                extracted_indices = [
                    int(frame.info["source_frame_index"]) for frame in frames
                ]
                if extracted_indices != frame_indices:
                    raise RuntimeError(f"Could not extract exact frame pack at row {index}")
                prompt = build_longqa_prompt(
                    row["question"], row["mcq_options"], prompt_variant=args.prompt_variant
                )
                response = model.generate(
                    frames, [{"role": "user", "content": prompt}], args.max_new_tokens
                )
                prediction = build_prediction_row(
                    row, response, prompt_variant=args.prompt_variant
                )
                prediction["fixed_pack_frame_source"] = args.frame_source
                prediction["fixed_pack_frame_fingerprint"] = pack_fingerprint
                prediction["fixed_pack_inference_fingerprint"] = generation_fingerprint
                handle.write(json.dumps(prediction) + "\n")
                handle.flush()
                print(f"  Generation progress: {index + 1}/{len(rows)}")
        _print_context_summary(summarize_prompt_token_stats())

    print(f"Predictions written to {output_path}")
    print(f"Frame-pack metadata written to {frame_pack_output}")
    if not args.no_eval:
        _run_eval(input_path, output_path, eval_output)


if __name__ == "__main__":
    main()
