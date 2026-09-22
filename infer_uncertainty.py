"""Uncertainty-guided selection

The VLM scores each of 128 candidate frames on its own: how confidently it
answers the question from that frame (evidence), and whether the frame shows
the question's reference event (pivot). The best frames of each kind, their
neighbors, and coverage frames make up the 64 frames it answers on
"""

import argparse
import math
import re
from pathlib import Path
from typing import Any

from models.vlm import answer, question_prompt, request
from utils.data import (
    read_frames, read_samples, resize_for_selection,
    uniform_indices, uniform_indices_before_end, video_info, write_prediction,
)
from utils.frame_selection import (
    add_frame, add_neighbors, allowed_targets, bridge_positions, finish_selection, rank_frames,
)

CANDIDATES = 128  # frames sampled uniformly, each scored by the VLM
FRAMES = 64  # frames sent to the VLM for the answer
TARGETS = 8  # most confident evidence frames kept
COVERAGE = 24  # evenly spaced frames kept for context over the whole video


def temporal_question(question: str) -> tuple[str, str]:
    """Parse the question into a reference event and where the evidence lies

    This is the simpler parser the uncertainty results were produced with; it differs
    from infer_temporal.temporal_question (no multi-event pairs, longer embedded
    references). Directions have the same meaning
    """
    text = " ".join(question.split())
    leading = re.match(r"(?i)^(after|before)\s+(.+?),\s*(.+)$", text)
    if leading:
        return leading[2].strip(), leading[1].lower()
    embedded = re.search(r"(?i)\b(after|before)\s+(.+?)(?:\?|$)", text)
    if embedded:
        return embedded[2].strip(" ,.?"), embedded[1].lower()
    if re.search(r"(?i)\b(first|earliest|at the start|beginning)\b", text):
        return text, "any"
    if re.search(r"(?i)\b(last|latest|finally|at the end|end of)\b", text):
        return text, "latest"
    if re.search(r"(?i)\b(chang(?:e|ed)|different|compared|between)\b", text):
        return text, "any"
    return "", "global"


def entropy(tokens: list[dict[str, Any]]) -> float:
    """Entropy of the next-token distribution from its top returned tokens"""
    probabilities = [math.exp(t["logprob"]) for t in tokens if math.isfinite(t["logprob"])]
    # Unreturned vocabulary entries form one outcome, giving an entropy lower bound
    tail = max(0.0, 1.0 - sum(probabilities))
    return -sum(p * math.log(p) for p in probabilities + [tail] if p > 0.0)


def letter_probability(tokens: list[dict[str, Any]], letter: str) -> float:
    """Probability of the most likely spelling of `letter`, 0 if not returned"""
    return max(
        (math.exp(t["logprob"]) for t in tokens if t["token"].strip().upper() == letter),
        default=0.0,
    )


def select_frames(direction: str, times: list[float], targets: list[float], pivots: list[float]) -> list[int]:
    """Choose FRAMES candidate positions by priority, returned in time order

    Questions with a reference event: pivots, targets on the allowed side of the
    strongest pivot, bridges between them, coverage. Global questions: targets and
    coverage only. See utils/frame_selection.py for the priorities
    """
    # targets: negative answer entropy per candidate; pivots: P(yes) - P(no) per candidate
    count = len(times)
    selected = {}
    if direction == "global":
        # No reference event: most confident frames and coverage only
        for center in rank_frames(targets, times, TARGETS):
            add_neighbors(selected, center, count, 0, targets[center])
        for i in uniform_indices(count, COVERAGE):
            add_frame(selected, i, 3, targets[i])
        # No embeddings here, so the boundary fill degenerates to the first eight
        # unselected frames in index order, frame 0 last
        unselected = [i for i in [*range(1, count), 0] if i not in selected]
        for i in unselected[:8]:
            add_frame(selected, i, 4, float(i > 0))
    else:
        # Up to two moments showing the reference event; the strongest one
        # decides the before/after side and anchors the bridges
        references = rank_frames(pivots, times, 2)
        primary = references[0]
        for center in references:
            add_neighbors(selected, center, count, 0, pivots[center])
        allowed = allowed_targets(direction, primary, targets, times)
        evidence = rank_frames(targets, times, TARGETS, allowed)
        for center in evidence:
            add_neighbors(selected, center, count, 1, targets[center])
        # Bridges show what happens between the pivot and the first two targets
        for target in evidence[:2]:
            for i in bridge_positions(primary, target):
                add_frame(selected, i, 2, targets[i])
        for i in uniform_indices(count, COVERAGE):
            add_frame(selected, i, 3, targets[i])

    # Fill with the frame farthest from every selected frame, earliest on ties
    while len(selected) < min(FRAMES, count):
        remaining = [i for i in range(count) if i not in selected]
        position = max(remaining, key=lambda i: (min(abs(i - j) for j in selected), -i))
        add_frame(selected, position, 4, -1e9)
    return finish_selection(selected, FRAMES)


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
            total, fps = video_info(video)
            indices = uniform_indices_before_end(total, CANDIDATES)
            candidates = read_frames(video, indices)
            reference, direction = temporal_question(sample["question"])
            target_prompt = (
                "Use only the supplied chronological visual evidence. Answer the "
                "multiple-choice question. Return only A, B, C, or D.\n\n"
                f"Question: {sample['question']}\n\nOptions:\n{sample['mcq_options']}"
            )
            pivot_prompt = (
                "Decide whether the supplied visual evidence visibly contains the "
                "specified event. Do not infer it from general plausibility.\n\n"
                f"Event: {reference or sample['question']}\n\nA. Yes\nB. No\n\nReturn only A or B."
            )
            # Each candidate is scored alone at low resolution, once per prompt:
            # low answer entropy marks evidence, P(yes) - P(no) marks the reference event
            target_scores, pivot_scores = [], []
            for frame in candidates:
                small_frame = resize_for_selection(frame)
                result = request(
                    args.server, args.model, [small_frame], target_prompt,
                    max_tokens=1, top_logprobs=100,
                )
                tokens = result["logprobs"]["content"][0]["top_logprobs"]
                target_scores.append(-entropy(tokens))
                result = request(
                    args.server, args.model, [small_frame], pivot_prompt,
                    max_tokens=1, top_logprobs=100,
                )
                tokens = result["logprobs"]["content"][0]["top_logprobs"]
                pivot_scores.append(letter_probability(tokens, "A") - letter_probability(tokens, "B"))
            positions = select_frames(direction, [i / fps for i in indices], target_scores, pivot_scores)
            frames = [candidates[i] for i in positions]
            prediction = answer(args.server, args.model, frames, question_prompt(sample))
            write_prediction(output, sample, prediction)


if __name__ == "__main__":
    main()
