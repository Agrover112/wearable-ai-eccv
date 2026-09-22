"""Query-conditioned allocation (QCA)

The video is split into 16 segments. Each segment gets a share of the 64 frames
from its SigLIP2 relevance to the question and how visually distinct it is,
then fills that share with its most relevant frame plus diverse frames
"""

import argparse
from pathlib import Path

import numpy as np

from models.siglip import image_features, load_siglip, relevance_scores
from models.vlm import answer, question_prompt
from utils.data import read_frames, read_samples, uniform_indices, video_info, write_prediction

CANDIDATES = 128  # frames sampled uniformly and embedded by SigLIP2
FRAMES = 64  # frames sent to the VLM
SEGMENTS = 16  # consecutive groups of candidates that share the frame budget


def select_frames(features: np.ndarray, scores: list[float]) -> list[int]:
    """Positions of FRAMES candidates, in time order"""
    # features: [N, D] float32 L2-normalized; scores: [N] cosine similarities
    relevance = np.asarray(scores, dtype=np.float32)
    budget = min(FRAMES, len(features))
    segments = [part for part in np.array_split(np.arange(len(features)), SEGMENTS) if len(part)]
    # Per segment: mean relevance, and visual deviation (distance of its mean
    # embedding from the video mean plus its own spread)
    global_mean = features.mean(axis=0)
    matching = np.asarray([relevance[part].mean() for part in segments])
    deviation = np.asarray([
        np.square(features[part].mean(axis=0) - global_mean).sum()
        + features[part].var(axis=0).sum()
        for part in segments
    ])

    def softmax(values: np.ndarray) -> np.ndarray:
        weights = np.exp(values - values.max())
        return weights / weights.sum()

    # Budgets follow the square root of an equal mix of both softmaxes, rounded
    # so they sum exactly to the budget (largest remainders round up)
    contribution = 0.5 * softmax(matching) + 0.5 * softmax(deviation)
    weights = np.power(np.maximum(contribution, 1e-12), 0.5)
    weights /= weights.sum()
    raw_quotas = weights * budget
    quotas = np.floor(raw_quotas).astype(int)
    for i in np.argsort(-(raw_quotas - quotas))[:budget - int(quotas.sum())]:
        quotas[i] += 1

    selected = []
    for segment, quota in zip(segments, quotas):
        if quota == 0:
            continue
        # Start from the most relevant frame; draw the rest from frames at least
        # 70% as relevant, loosening that bar until the pool is large enough
        anchor = int(segment[np.argmax(relevance[segment])])
        chosen = [anchor]
        threshold = 0.7
        pool = segment[relevance[segment] >= threshold * relevance[anchor]]
        while len(pool) < quota and threshold > 0:
            threshold = max(0.0, threshold - 0.1)
            pool = segment[relevance[segment] >= threshold * relevance[anchor]]
        if len(pool) < quota:
            pool = segment
        # Greedily add the frame least similar to everything chosen so far
        while len(chosen) < quota:
            remaining = [int(i) for i in pool if int(i) not in chosen]
            if not remaining:
                break
            diversity = [
                min(1.0 - float(np.dot(features[i], features[j])) for j in chosen)
                for i in remaining
            ]
            chosen.append(remaining[int(np.argmax(diversity))])
        selected.extend(chosen)

    # Segments shorter than their budget leave gaps; top up by relevance
    if len(selected) < budget:
        for i in np.argsort(-relevance):
            if int(i) not in selected:
                selected.append(int(i))
            if len(selected) == budget:
                break
    return sorted(selected[:budget])


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
            total, _ = video_info(video)
            candidates = read_frames(video, uniform_indices(total, CANDIDATES))
            features = image_features(model, processor, candidates)
            # One joint query. SigLIP2 truncates it to 64 tokens, so long questions
            # lose their later options; the reported results were produced this way
            query = (
                "Find video evidence needed to answer this multiple-choice question.\n"
                f"Question: {sample['question']}\nOptions:\n{sample['mcq_options']}"
            )
            scores = relevance_scores(model, processor, features, query)
            positions = select_frames(features, scores)
            frames = [candidates[i] for i in positions]
            prediction = answer(args.server, args.model, frames, question_prompt(sample))
            write_prediction(output, sample, prediction)


if __name__ == "__main__":
    main()
