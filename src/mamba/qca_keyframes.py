"""Training-free QCA-style keyframe selection for cached VL embeddings."""

from __future__ import annotations

import numpy as np


def select_qca_keyframes(
    features: np.ndarray,
    relevance: np.ndarray,
    budget: int = 32,
    num_segments: int = 12,
    alpha: float = 0.5,
    beta: float = 0.5,
    temperature: float = 0.5,
    relevance_threshold: float = 0.7,
) -> dict[str, object]:
    """Select relevant, diverse frames while allocating budget across time."""
    features = np.asarray(features, dtype=np.float32)
    relevance = np.asarray(relevance, dtype=np.float32)
    if features.ndim != 2 or relevance.shape != (len(features),):
        raise ValueError("expected features [frames, dim] and relevance [frames]")
    if not 0 < budget <= len(features):
        raise ValueError("budget must be between 1 and the candidate count")

    segments = [x for x in np.array_split(np.arange(len(features)), num_segments) if len(x)]
    global_mean = features.mean(axis=0)
    matching = np.asarray([relevance[x].mean() for x in segments])
    deviation = np.asarray([
        np.square(features[x].mean(axis=0) - global_mean).sum()
        + features[x].var(axis=0).sum()
        for x in segments
    ])

    def softmax(values: np.ndarray) -> np.ndarray:
        values = np.exp(values - values.max())
        return values / values.sum()

    contribution = alpha * softmax(matching) + beta * softmax(deviation)
    weights = np.power(contribution, temperature)
    weights /= weights.sum()
    quotas = np.floor(weights * budget).astype(int)
    remainder = budget - int(quotas.sum())
    for index in np.argsort(-contribution)[:remainder]:
        quotas[index] += 1

    selected: list[int] = []
    for segment, quota in zip(segments, quotas):
        if quota <= 0:
            continue
        anchor = int(segment[np.argmax(relevance[segment])])
        chosen = [anchor]
        gamma = relevance_threshold
        pool = segment[relevance[segment] >= gamma * relevance[anchor]]
        while len(pool) < quota and gamma > 0:
            gamma -= 0.1
            pool = segment[relevance[segment] >= gamma * relevance[anchor]]
        if len(pool) < quota:
            pool = segment
        while len(chosen) < quota:
            remaining = [int(i) for i in pool if int(i) not in chosen]
            if not remaining:
                break
            diversity = np.asarray([
                sum(np.linalg.norm(features[i] - features[j]) for j in chosen)
                for i in remaining
            ])
            chosen.append(remaining[int(diversity.argmax())])
        selected.extend(chosen)

    if len(selected) < budget:
        for index in np.argsort(-relevance):
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == budget:
                break

    return {
        "indices": sorted(selected[:budget]),
        "quotas": quotas.tolist(),
        "segment_matching": matching.tolist(),
        "segment_deviation": deviation.tolist(),
    }
