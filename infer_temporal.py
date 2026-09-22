"""Option-conditioned temporal selection

SigLIP2 retrieval picks 64 of 128 candidate frames around the question's reference
event and its answer options; the VLM then answers once on those frames
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

from models.siglip import image_features, load_siglip, relevance_scores
from models.vlm import answer, question_prompt
from utils.data import read_frames, read_samples, uniform_indices, video_info, write_prediction
from utils.frame_selection import (
    add_frame, add_neighbors, allowed_targets, bridge_positions, finish_selection, rank_frames,
)

CANDIDATES = 128  # frames sampled uniformly and scored by SigLIP2
FRAMES = 64  # frames sent to the VLM
TARGETS = 8  # evidence moments: one per option, then the best question matches
COVERAGE = 24  # evenly spaced frames kept for context over the whole video


def temporal_question(question: str) -> tuple[str, str]:
    """Parse the question into a reference event and where the evidence lies

    "What did I do after X?" gives ("X", "after"). direction is "after" or
    "before" the reference event, "latest", "any" side of it, or "global" when
    the question names no reference event
    """
    text = " ".join(question.split())
    # Questions that combine cues, e.g. both "before" and "after", span several events
    pairs = (("before", "after"), ("first", "last"), ("earlier", "later"))
    if any(all(re.search(rf"\b{word}\b", text, re.I) for word in pair) for pair in pairs):
        return "", "any"
    leading = re.match(r"(?i)^(after|before)\s+(.+?),\s*(.+)$", text)
    if leading:
        return leading[2].strip(), leading[1].lower()
    # "... after X, what ..." or "... after X did I ...": X ends where the next clause starts
    embedded = re.search(r"(?i)\b(after|before)\b", text)
    if embedded:
        remainder = text[embedded.end():].strip(" ,.?-")
        parts = re.split(
            r"(?i)(?:[,;.]\s*(?:and\s+)?(?=(?:what|which|where|when|how|who)\b)"
            r"|\s+(?=(?:did|do|was|were|had|have)\s+I\b))", remainder, maxsplit=1,
        )
        return parts[0].strip(" ,.?-"), embedded[1].lower()
    if re.search(r"(?i)\b(first|earliest|at the start|beginning)\b", text):
        return text, "any"
    if re.search(r"(?i)\b(last|latest|finally|at the end|end of)\b", text):
        return text, "latest"
    if re.search(r"(?i)\b(chang(?:e|ed)|different|compared|between)\b", text):
        return text, "any"
    return "", "global"


def head_tail(tokens: list[int], budget: int) -> list[int]:
    """Shorten to `budget` tokens by keeping the start and the end"""
    if len(tokens) <= budget:
        return tokens
    head = math.ceil(budget / 2)
    return tokens[:head] + tokens[-(budget - head):]


def option_queries(sample: dict[str, str], tokenizer: PreTrainedTokenizerBase) -> dict[str, str]:
    """Retrieval queries keyed "question", "A", ..., "D", each within SigLIP2's 64 tokens"""
    question = tokenizer.encode(sample["question"], add_special_tokens=False)

    def decode(tokens: list[int]) -> str:
        return tokenizer.decode(tokens, skip_special_tokens=True).strip()

    queries = {"question": f"Question: {decode(head_tail(question, 60))}"}
    # Split "A. ... B. ... C. ... D. ..." into the four option texts
    text = sample["mcq_options"]
    matches = list(re.finditer(r"(?ms)(?:^|\s)([A-D])\.\s*", text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        option = tokenizer.encode(text[match.end():end].strip(), add_special_tokens=False)
        # Trim the longer of question and option until "Question: ... Answer: ..." fits
        question_budget, option_budget = min(len(question), 30), min(len(option), 26)
        while True:
            query = (
                f"Question: {decode(head_tail(question, question_budget))}\n"
                f"Answer: {decode(head_tail(option, option_budget))}"
            )
            if len(tokenizer.encode(query, add_special_tokens=True)) <= 64:
                break
            if question_budget >= option_budget and question_budget > 10:
                question_budget -= 1
            elif option_budget > 10:
                option_budget -= 1
            else:
                raise ValueError("Question and option exceed the 64-token retrieval budget")
        queries[match[1]] = query
    return queries


def balanced_scores(scores: dict[str, list[float]]) -> list[float]:
    """Per-frame relevance: half question match, half mean option match, all z-scored"""
    def zscore(values: list[float]) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        std = float(values.std())
        if std < 1e-6:
            return np.zeros_like(values)
        return (values - float(values.mean())) / std

    option_mean = np.mean(np.stack([zscore(scores[letter]) for letter in "ABCD"]), axis=0)
    return (0.5 * zscore(scores["question"]) + 0.5 * option_mean).tolist()


def assemble_selection(
    direction: str, times: list[float], scores: dict[str, list[float]],
    pivot_scores: list[float], features: np.ndarray,
) -> list[int]:
    """Choose FRAMES candidate positions by priority, returned in time order

    Pivots (the reference event) and targets (evidence for the options) come
    first with their neighbors, then bridges between them, then coverage, then
    the biggest visual changes. See utils/frame_selection.py for the priorities
    """
    # features: [N, D] float32 L2-normalized; scores and times have length N
    count = len(times)
    relevance = balanced_scores(scores)
    # Up to two moments that look like the reference event; "after"/"before"
    # and bridges are measured from the strongest one (the middle if none)
    pivots = [] if direction == "global" else rank_frames(pivot_scores, times, 2)
    primary = pivots[0] if pivots else count // 2
    allowed = allowed_targets(direction, primary, relevance, times)

    def rank_targets(values: list[float], number: int) -> list[int]:
        # An empty side of the pivot falls back to the whole video
        return rank_frames(values, times, number, allowed) or rank_frames(values, times, number)

    selected = {}
    for center in pivots:
        add_neighbors(selected, center, count, 0, pivot_scores[center])

    # Targets: the best frame for each option, then question matches, then the mix
    targets = []
    for letter in "ABCD":
        for center in rank_targets(scores[letter], 1):
            if center not in targets:
                targets.append(center)
    remaining = TARGETS - len(targets)
    targets.extend([i for i in rank_targets(scores["question"], TARGETS) if i not in targets][:remaining])
    if len(targets) < TARGETS:
        targets.extend(i for i in rank_targets(relevance, TARGETS) if i not in targets)
    targets = targets[:TARGETS]
    for center in targets:
        add_neighbors(selected, center, count, 1, relevance[center])
    # Bridges show what happens between the pivot and the first two targets
    for target in targets[:2]:
        for i in bridge_positions(primary, target):
            add_frame(selected, i, 2, relevance[i])
    for i in uniform_indices(count, COVERAGE):
        add_frame(selected, i, 3, relevance[i])

    # Fill with the largest visual changes between consecutive candidates
    boundaries = [0.0] + [float(1.0 - np.dot(features[i], features[i - 1])) for i in range(1, count)]
    for i in sorted(range(count), key=boundaries.__getitem__, reverse=True):
        if len(selected) >= FRAMES:
            break
        add_frame(selected, i, 5, boundaries[i])
    return finish_selection(selected, FRAMES)


def select_frames(
    sample: dict[str, str], frames: list[Image.Image], times: list[float],
    model: PreTrainedModel, processor: ProcessorMixin,
) -> list[int]:
    """Score the candidate frames with SigLIP2 and return the positions of FRAMES of them"""
    features = image_features(model, processor, frames)
    queries = option_queries(sample, processor.tokenizer)
    scores = {name: relevance_scores(model, processor, features, query) for name, query in queries.items()}
    # A separate query locates the reference event itself
    reference, direction = temporal_question(sample["question"])
    query = (
        "Find this temporal reference event in the video.\n"
        f"Reference event: {reference or sample['question']}"
    )
    pivot_scores = relevance_scores(model, processor, features, query)
    return assemble_selection(direction, times, scores, pivot_scores, features)


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
            # Timestamps in seconds keep the chosen moments apart
            positions = select_frames(sample, candidates, [i / fps for i in indices], model, processor)
            frames = [candidates[i] for i in positions]
            prediction = answer(args.server, args.model, frames, question_prompt(sample))
            write_prediction(output, sample, prediction)


if __name__ == "__main__":
    main()
