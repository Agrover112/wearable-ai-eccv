"""Dual-view fusion of uniform and temporal frame selection

The VLM answers on 64 uniform frames and on 64 temporally selected frames. If
the two answers agree that answer is kept; otherwise the two A-D probability
distributions are averaged and the most probable letter wins
"""

import argparse
from pathlib import Path

from models.siglip import load_siglip
from models.vlm import answer, option_probabilities, question_prompt
from utils.data import read_frames, read_samples, uniform_indices, video_info, write_prediction
from infer_temporal import CANDIDATES, FRAMES, select_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--siglip-model", default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--siglip-device", default="cuda")
    args = parser.parse_args()

    model, processor = load_siglip(args.siglip_model, args.siglip_device)
    samples = read_samples(args.questions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for sample in samples:
            video = args.videos / sample["video_path"]
            total, fps = video_info(video)
            indices = uniform_indices(total, CANDIDATES)
            candidates = read_frames(video, indices)
            # Temporal view: 64 of the 128 candidates, as in infer_temporal.py
            positions = select_frames(sample, candidates, [i / fps for i in indices], model, processor)
            temporal_frames = [candidates[i] for i in positions]
            # Uniform view: its own 64-frame grid, not every other 128-grid candidate
            uniform_frames = read_frames(video, uniform_indices(total, FRAMES))
            prompt = question_prompt(sample)
            uniform_answer = answer(args.server, args.model, uniform_frames, prompt)
            temporal_answer = answer(args.server, args.model, temporal_frames, prompt)
            if uniform_answer and uniform_answer == temporal_answer:
                prediction = uniform_answer
            else:
                # Disagreement or an unparsable answer: average the two views
                uniform_probs = option_probabilities(args.server, args.model, uniform_frames, prompt)
                temporal_probs = option_probabilities(args.server, args.model, temporal_frames, prompt)
                mean_probs = {
                    letter: (uniform_probs[letter] + temporal_probs[letter]) / 2
                    for letter in "ABCD"
                }
                best = max(mean_probs.values())
                tied = [letter for letter in "ABCD" if abs(mean_probs[letter] - best) < 1e-12]
                # Exact ties go to the uniform answer, then the temporal one
                fallback = uniform_answer or temporal_answer or "A"
                prediction = fallback if fallback in tied else tied[0]
            write_prediction(output, sample, prediction)


if __name__ == "__main__":
    main()
