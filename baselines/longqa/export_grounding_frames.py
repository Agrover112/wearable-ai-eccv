#!/usr/bin/env python3
"""Export selected temporal-grounding frames for visual inspection.

This is an analysis utility for runs produced by run_generate_longqa_grounded.py.
It does not run the VLM again. It saves the selected/top-scoring frames and a
small metadata file so we can inspect which moments the text-image grounder put
in front of the final VLM.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_answer(raw: object) -> str:
    text = str(raw).strip()
    if not text:
        return ""
    upper = text.upper()
    if upper in {"A", "B", "C", "D"}:
        return upper
    m = re.search(r"\b(?:is|answer)\s*[:.]?\s*([A-Da-d])\s*\.?\s*$", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()
    return upper[0] if upper[0] in "ABCD" else ""


def is_correct(row: dict[str, Any], pred: dict[str, Any]) -> bool:
    return normalize_answer(pred.get("mcq_answer", "")) == normalize_answer(
        row.get("mcq_answer", "")
    )


def safe_stem(video_path: str, index: int) -> str:
    stem = Path(video_path).stem or f"sample_{index:04d}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def extract_frame(video_path: str, frame_index: int):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    finally:
        cap.release()


def make_contact_sheet(images: list[tuple[str, object]], output_path: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not images:
        return
    thumb_w, thumb_h = 256, 144
    label_h = 38
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, (label, image) in enumerate(images):
        thumb = image.copy()
        thumb.thumbnail((thumb_w, thumb_h))
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb_h + 4), label[:42], fill=(0, 0, 0), font=font)
    sheet.save(output_path, quality=92)


def select_records(
    rows: list[dict[str, Any]],
    grounded_preds: list[dict[str, Any]] | None,
    compare_preds: list[dict[str, Any]] | None,
    mode: str,
) -> list[int]:
    indices: list[int] = []
    for idx, row in enumerate(rows):
        grounded_ok = (
            is_correct(row, grounded_preds[idx]) if grounded_preds is not None else True
        )
        compare_ok = (
            is_correct(row, compare_preds[idx]) if compare_preds is not None else False
        )

        if mode == "all":
            keep = True
        elif mode == "grounded-correct":
            keep = grounded_ok
        elif mode == "grounded-wrong":
            keep = not grounded_ok
        elif mode == "improved":
            keep = grounded_ok and not compare_ok
        elif mode == "regressed":
            keep = (not grounded_ok) and compare_ok
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if keep:
            indices.append(idx)
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export selected/top-scoring frames from a grounded LongQA run."
    )
    parser.add_argument(
        "--input",
        default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl",
        help="Golden LongQA jsonl.",
    )
    parser.add_argument(
        "--grounding",
        required=True,
        help="grounding.jsonl produced by run_generate_longqa_grounded.py.",
    )
    parser.add_argument(
        "--grounded-predictions",
        default=None,
        help="Predictions from the grounded run, needed for correctness filters.",
    )
    parser.add_argument(
        "--compare-predictions",
        default=None,
        help="Baseline predictions to compare against for improved/regressed filters.",
    )
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get(
            "FRAME_AUDIT_DIR",
            os.path.join(os.environ.get("SCRATCH_CACHE_ROOT", "/tmp"), "frame_audits"),
        ),
        help="Directory for exported JPEGs/contact sheets/metadata.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "grounded-correct", "grounded-wrong", "improved", "regressed"],
        default="improved",
    )
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of highest-scoring retrieved frames to save per sample.",
    )
    parser.add_argument(
        "--include-anchors",
        action="store_true",
        help="Also save anchor frames after the top retrieved frames.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    def resolve(path: str | None) -> str | None:
        if path is None:
            return None
        p = Path(path)
        if p.is_absolute():
            return str(p)
        if p.exists():
            return str(p.resolve())
        return str(script_dir / p)

    input_path = resolve(args.input)
    grounding_path = resolve(args.grounding)
    grounded_pred_path = resolve(args.grounded_predictions)
    compare_pred_path = resolve(args.compare_predictions)
    video_folder = Path(resolve(args.video_folder) or "")
    output_dir = Path(resolve(args.output_dir) or "")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path or "")
    grounding = load_jsonl(grounding_path or "")
    grounded_preds = load_jsonl(grounded_pred_path) if grounded_pred_path else None
    compare_preds = load_jsonl(compare_pred_path) if compare_pred_path else None

    if len(grounding) != len(rows):
        raise ValueError(f"Grounding rows ({len(grounding)}) != input rows ({len(rows)})")
    if grounded_preds is not None and len(grounded_preds) != len(rows):
        raise ValueError("Grounded predictions length does not match input length")
    if compare_preds is not None and len(compare_preds) != len(rows):
        raise ValueError("Compare predictions length does not match input length")
    if args.mode in {"grounded-correct", "grounded-wrong", "improved", "regressed"}:
        if grounded_preds is None:
            raise ValueError(f"--mode {args.mode} requires --grounded-predictions")
    if args.mode in {"improved", "regressed"} and compare_preds is None:
        raise ValueError(f"--mode {args.mode} requires --compare-predictions")

    selected_indices = select_records(rows, grounded_preds, compare_preds, args.mode)
    selected_indices = selected_indices[: args.max_samples]

    manifest: list[dict[str, Any]] = []
    for idx in selected_indices:
        row = rows[idx]
        meta = grounding[idx]
        video_rel = str(row.get("video_path", meta.get("video_path", "")))
        video_path = video_folder / video_rel
        sample_dir = output_dir / f"{idx:04d}_{safe_stem(video_rel, idx)}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        frames = list(meta.get("selected", []))
        retrieved = [f for f in frames if str(f.get("source", "")).startswith("retrieved")]
        anchors = [f for f in frames if f.get("source") == "anchor"]
        retrieved = sorted(
            retrieved,
            key=lambda f: float(f.get("score", "-inf"))
            if f.get("score") is not None
            else float("-inf"),
            reverse=True,
        )[: args.top_n]
        to_save = retrieved + (anchors if args.include_anchors else [])

        saved: list[dict[str, Any]] = []
        sheet_items: list[tuple[str, object]] = []
        for rank, frame_meta in enumerate(to_save, start=1):
            frame_index = int(frame_meta["frame_index"])
            image = extract_frame(str(video_path), frame_index)
            source = str(frame_meta.get("source", "frame"))
            score = frame_meta.get("score")
            timestamp = frame_meta.get("timestamp")
            filename = f"{rank:02d}_{source}_f{frame_index}_t{timestamp}.jpg"
            out_path = sample_dir / filename
            image.save(out_path, quality=args.jpeg_quality)
            label = f"{rank}: {source} t={timestamp}s score={score}"
            sheet_items.append((label, image))
            saved.append(
                {
                    "rank": rank,
                    "file": str(out_path.relative_to(output_dir)),
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "score": score,
                    "source": source,
                }
            )

        make_contact_sheet(sheet_items, str(sample_dir / "contact_sheet.jpg"))
        grounded_answer = (
            normalize_answer(grounded_preds[idx].get("mcq_answer", ""))
            if grounded_preds is not None
            else None
        )
        compare_answer = (
            normalize_answer(compare_preds[idx].get("mcq_answer", ""))
            if compare_preds is not None
            else None
        )
        sample_meta = {
            "index": idx,
            "video_path": video_rel,
            "category": row.get("category", ""),
            "question": row.get("question", ""),
            "gold_answer": normalize_answer(row.get("mcq_answer", "")),
            "grounded_answer": grounded_answer,
            "compare_answer": compare_answer,
            "mode": args.mode,
            "frames": saved,
        }
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(sample_meta, f, indent=2)
        manifest.append(sample_meta)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported {len(manifest)} samples to {output_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
