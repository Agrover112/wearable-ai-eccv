"""Baseline: answer from 64 evenly spaced frames, first and last frame included"""

import argparse
from pathlib import Path

from models.vlm import answer, question_prompt
from utils.data import read_frames, read_samples, uniform_indices, video_info, write_prediction

FRAMES = 64  # frames sent to the VLM


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
            frames = read_frames(video, uniform_indices(total, FRAMES))
            prediction = answer(args.server, args.model, frames, question_prompt(sample))
            write_prediction(output, sample, prediction)


if __name__ == "__main__":
    main()
