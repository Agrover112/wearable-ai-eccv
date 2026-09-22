"""Read questions, decode video frames, and write predictions"""

import json
from pathlib import Path
from typing import TextIO

import cv2
from PIL import Image


def read_samples(path: Path) -> list[dict[str, str]]:
    """One question per JSONL line"""
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def video_info(path: Path) -> tuple[int, float]:
    """Frame count and frames per second of a video"""
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count == 0 or fps <= 0:
        raise ValueError(f"Cannot read video metadata: {path}")
    return frame_count, fps


def uniform_indices(total: int, count: int) -> list[int]:
    """`count` evenly spaced frame indices, first and last frame included"""
    count = min(count, total)
    if count == 1:
        return [total // 2]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def uniform_indices_before_end(total: int, count: int) -> list[int]:
    """`count` evenly spaced frame indices that stop short of the final frame

    The uncertainty and TCoT results were produced with this older grid
    """
    count = min(count, total)
    return sorted({int(i * (total - 1) / count) for i in range(count)})


def read_frames(path: Path, indices: list[int]) -> list[Image.Image]:
    """Decode the frames at `indices` as RGB images"""
    capture = cv2.VideoCapture(str(path))
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, bgr = capture.read()
        if not success:
            capture.release()
            raise ValueError(f"Cannot decode frame {index}: {path}")
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    capture.release()
    return frames


def resize_for_selection(frame: Image.Image) -> Image.Image:
    """Downscale to at most 50,176 pixels (about 224x224) for cheap selection requests"""
    width, height = frame.size
    if width * height <= 50176:
        return frame
    scale = (50176 / (width * height)) ** 0.5
    return frame.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)


def write_prediction(stream: TextIO, sample: dict[str, str], answer: str) -> None:
    """Append one prediction line; flushed so an interrupted run keeps its answers"""
    prediction = {
        "video_path": sample["video_path"],
        "question": sample["question"],
        "mcq_answer": answer,
    }
    stream.write(json.dumps(prediction) + "\n")
    stream.flush()
    print(f"{sample['video_path']}: {answer}", flush=True)
