#!/usr/bin/env python3
"""Export contact sheets from a key-aligned LongQA evidence-audit manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _sample_frames(frames: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(frames) <= count:
        return sorted(frames, key=lambda item: int(item["frame_index"]))
    priority = {
        "pivot": 0,
        "directional_target": 0,
        "event_1": 0,
        "event_2": 0,
        "event_3": 0,
        "qca_segment_anchor": 0,
        "bridge": 1,
        "pivot_context": 1,
        "directional_target_context": 1,
        "qca_diverse": 2,
        "anchor": 3,
        "uniform": 4,
        "verifier_union": 4,
    }
    ranked = sorted(
        frames,
        key=lambda item: (
            priority.get(str(item.get("source", "")), 2),
            -(float(item["score"]) if item.get("score") is not None else -1e9),
        ),
    )
    chosen = ranked[: count // 2]
    remaining = sorted(
        (item for item in frames if item not in chosen),
        key=lambda item: int(item["frame_index"]),
    )
    slots = count - len(chosen)
    if remaining and slots:
        positions = [int(index * len(remaining) / slots) for index in range(slots)]
        chosen.extend(remaining[min(index, len(remaining) - 1)] for index in positions)
    unique = {int(item["frame_index"]): item for item in chosen}
    return sorted(unique.values(), key=lambda item: int(item["frame_index"]))


def _extract(video_path: Path, frame_index: int):
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()


def _sheet(items: list[tuple[str, Any]], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not items:
        return
    width, height, label_height, columns = 320, 180, 44, 4
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(items):
        image = image.copy()
        image.thumbnail((width, height))
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        canvas.paste(image, (x, y))
        draw.text((x + 4, y + height + 3), label[:58], fill="black", font=font)
    canvas.save(output, quality=92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames-per-source", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    video_folder = Path(args.video_folder)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(manifest["samples"]):
        stem = f"{sample_index:03d}_{sample['group']}_{Path(sample['video_path']).stem}"
        sample_dir = output_dir / _safe(stem)
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_export = {key: value for key, value in sample.items() if key != "sources"}
        sample_export["sources"] = {}
        for source_name, frames in sample["sources"].items():
            selected = _sample_frames(frames, args.frames_per_source)
            items = []
            saved = []
            for rank, frame in enumerate(selected, start=1):
                frame_index = int(frame["frame_index"])
                image = _extract(video_folder / sample["video_path"], frame_index)
                timestamp = float(frame.get("timestamp", 0.0))
                role = str(frame.get("source", "frame"))
                name = f"{rank:02d}_{_safe(role)}_f{frame_index}_t{timestamp:.2f}.jpg"
                image.save(sample_dir / name, quality=92)
                items.append((f"{rank} {role} t={timestamp:.1f}s", image))
                saved.append({**frame, "file": str((sample_dir / name).relative_to(output_dir))})
            _sheet(items, sample_dir / f"contact_{_safe(source_name)}.jpg")
            sample_export["sources"][source_name] = saved
        (sample_dir / "metadata.json").write_text(json.dumps(sample_export, indent=2))
        exported.append(sample_export)
    (output_dir / "manifest.json").write_text(json.dumps(exported, indent=2))
    print(f"Exported {len(exported)} samples to {output_dir}")


if __name__ == "__main__":
    main()
