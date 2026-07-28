#!/usr/bin/env python3
"""Build a five-sample static viewer from saved agentic traces."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIEWER_ROOT = Path(__file__).resolve().parent
RUN_ROOT = (
    PROJECT_ROOT
    / "runs/egolongqa/qwen35_agentic_dev140_compact_retry_v2"
)
VIDEO_ROOT = PROJECT_ROOT / "data/videos"
SAMPLE_INDICES = (0, 4, 7, 12, 13)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def parse_options(text: str) -> dict[str, str]:
    matches = re.findall(
        r"(?:^|\s)([A-D])\.\s*(.*?)(?=\s+[A-D]\.\s|$)",
        text,
    )
    return dict(matches)


def citation_index(report: dict) -> dict[str, list[dict]]:
    citations: dict[str, list[dict]] = {}
    for verdict in report["verdicts"]:
        for item in verdict["evidence"]:
            citations.setdefault(str(item["frame_id"]), []).append(
                {
                    "option": verdict["option"],
                    "status": verdict["status"],
                    "observation": item["observation"],
                }
            )
    return citations


def export_frames(sample_number: int, trace: dict) -> list[dict]:
    output_dir = VIEWER_ROOT / "assets" / f"sample_{sample_number:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = VIDEO_ROOT / trace["video_path"]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    exported = []
    for item in trace["evidence"]:
        frame_id = int(item["frame_id"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(item["frame_index"]))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(
                f"Could not read frame {item['frame_index']} from {video_path}"
            )
        height, width = frame.shape[:2]
        if width > 560:
            target_height = round(height * 560 / width)
            frame = cv2.resize(
                frame,
                (560, target_height),
                interpolation=cv2.INTER_AREA,
            )
        relative_path = (
            Path("assets")
            / f"sample_{sample_number:02d}"
            / f"frame_{frame_id:02d}.jpg"
        )
        cv2.imwrite(
            str(VIEWER_ROOT / relative_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 82],
        )
        exported.append({**item, "image": relative_path.as_posix()})
    capture.release()
    return exported


def main() -> None:
    traces = load_jsonl(RUN_ROOT / "agent_traces.jsonl")
    predictions = load_jsonl(RUN_ROOT / "predictions.jsonl")
    evaluated = json.loads((RUN_ROOT / "results.json").read_text())["per_row"]
    samples = []
    for sample_number, index in enumerate(SAMPLE_INDICES, start=1):
        trace = traces[index]
        prediction = predictions[index]
        result = evaluated[index]
        temporal = trace["temporal_report"]
        visual = trace["visual_report"]
        samples.append(
            {
                "dev_index": index + 1,
                "video_path": trace["video_path"],
                "category": prediction["category"],
                "question": prediction["question"],
                "options": parse_options(prediction["mcq_options"]),
                "gold_answer": result["gold_answer"],
                "selected_answer": trace["selected_answer"],
                "correct": result["correct"],
                "evidence": export_frames(sample_number, trace),
                "citations": {
                    "temporal": citation_index(temporal),
                    "visual": citation_index(visual),
                },
                "agents": {
                    "hypothesis": trace["hypothesis_report"],
                    "temporal": temporal,
                    "visual": visual,
                    "organizer": trace["organizer_report"],
                },
                "timing": trace["timing"],
                "raw_trace": trace,
            }
        )
    payload = {
        "run": "qwen35_agentic_dev140_compact_retry_v2",
        "model": "Qwen/Qwen3.5-9B",
        "samples": samples,
    }
    data_path = VIEWER_ROOT / "data.js"
    data_path.write_text(
        "window.ORCHESTRATION_DATA = "
        + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        + ";\n"
    )
    print(f"Wrote {data_path}")
    print(f"Exported {sum(len(sample['evidence']) for sample in samples)} frames")


if __name__ == "__main__":
    main()
