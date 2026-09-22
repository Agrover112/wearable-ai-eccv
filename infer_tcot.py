"""Temporal chain of thought (TCoT)

The VLM looks at 256 candidate frames in four chronological segments at low
resolution and names the frames that show evidence. Those frames and their
neighbors, at most 64, are then used to answer
"""

import argparse
import json
from pathlib import Path
from typing import Any

from models.vlm import answer, question_prompt, request
from utils.data import (
    read_frames, read_samples, resize_for_selection, uniform_indices_before_end, video_info, write_prediction,
)

CANDIDATES = 256  # frames sampled uniformly for the selector to look at
FRAMES = 64  # most frames sent to the VLM for the answer
SEGMENTS = 4  # chronological segments, one selector request each
IDS_PER_SEGMENT = 6  # most frames the selector may name per segment
NEIGHBORS = 1  # candidates added on each side of every named frame


def selector_prompt(sample: dict[str, str], count: int) -> str:
    """Ask which of the segment's `count` frames show evidence, without answering"""
    return (
        f"You are given {count} images sampled chronologically from one segment "
        "of a long egocentric video. The images supplied before this text are FrameID "
        f"1 through FrameID {count}, in exactly that order.\n\n"
        "Select the smallest sufficient set of frames that contains visible evidence "
        "needed to answer the question. Include both the temporal anchor and the requested "
        "before/after event when relevant. For first, last, repeated, returned, again, or "
        "counting questions, retain every potentially relevant occurrence visible in this "
        "segment. If this segment contains no relevant visible evidence, return an empty "
        "frame_ids list. Do not answer the question. Do not select a frame solely because "
        "an answer option sounds plausible.\n\n"
        f"Question: {sample['question']}\n\nOptions:\n{sample['mcq_options']}\n\n"
        f"Return at most {IDS_PER_SEGMENT} FrameIDs and a short visible-evidence justification."
    )


def selector_format(count: int) -> dict[str, Any]:
    """JSON schema the server enforces: FrameIDs 1..count and a short justification"""
    schema = {
        "type": "object",
        "properties": {
            "frame_ids": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": count},
                "minItems": 0,
                "maxItems": min(count, IDS_PER_SEGMENT),
            },
            "justification": {"type": "string", "maxLength": 600},
        },
        "required": ["frame_ids", "justification"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "tcot_frame_selection", "strict": True, "schema": schema},
    }


def expand_selection(centers: list[int], count: int) -> list[int]:
    """Named positions plus their neighbors, evenly thinned to at most FRAMES"""
    # Fall back to uniform seeds when every segment returns no evidence
    if not centers:
        centers = uniform_indices_before_end(count, FRAMES)
    selected = sorted({
        i for center in centers
        for i in range(max(0, center - NEIGHBORS), min(count, center + NEIGHBORS + 1))
    })
    # Smaller evidence sets are answered as is, without padding to FRAMES
    if len(selected) > FRAMES:
        selected = [selected[i] for i in uniform_indices_before_end(len(selected), FRAMES)]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    args = parser.parse_args()

    samples = read_samples(args.questions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for sample in samples:
            video = args.videos / sample["video_path"]
            total, _ = video_info(video)
            candidates = read_frames(video, uniform_indices_before_end(total, CANDIDATES))
            # One low-resolution selector request per chronological segment
            segments = min(SEGMENTS, len(candidates))
            boundaries = [round(i * len(candidates) / segments) for i in range(segments + 1)]
            centers = []
            for start, end in zip(boundaries, boundaries[1:]):
                frames = [resize_for_selection(frame) for frame in candidates[start:end]]
                result = request(
                    args.server, args.model, frames, selector_prompt(sample, len(frames)),
                    max_tokens=256, response_format=selector_format(len(frames)),
                )
                selection = json.loads(result["message"]["content"])
                # FrameIDs are 1-based within the segment
                centers.extend(start + frame_id - 1 for frame_id in selection["frame_ids"])
            positions = expand_selection(centers, len(candidates))
            frames = [candidates[i] for i in positions]
            prediction = answer(args.server, args.model, frames, question_prompt(sample))
            write_prediction(output, sample, prediction)


if __name__ == "__main__":
    main()
