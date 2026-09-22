"""Shared frame assembly for the temporal and uncertainty selectors

Candidates are addressed by position in the candidate grid. A selection maps
position -> (priority, score), and lower priority wins:
0 pivot, 1 pivot neighbor or target, 2 target neighbor or bridge, 3 coverage,
4-5 fill. The final set keeps the best-priority frames, breaks ties by score,
and returns them in time order
"""

from utils.data import uniform_indices

SEPARATION_SECONDS = 10.0  # chosen moments must be this far apart, relaxed if too few remain


def rank_frames(
    scores: list[float], times: list[float], count: int, allowed: set[int] | None = None,
) -> list[int]:
    """Up to `count` top-scoring positions, SEPARATION_SECONDS apart, from `allowed`"""
    ranked = sorted(
        (i for i in range(len(scores)) if allowed is None or i in allowed),
        key=scores.__getitem__, reverse=True,
    )
    selected = []
    for i in ranked:
        if all(abs(times[i] - times[j]) >= SEPARATION_SECONDS for j in selected):
            selected.append(i)
            if len(selected) == count:
                return selected
    # Too few well-separated moments: fill the rest by score alone
    for i in ranked:
        if i not in selected:
            selected.append(i)
            if len(selected) == count:
                break
    return selected


def allowed_targets(direction: str, pivot: int, scores: list[float], times: list[float]) -> set[int] | None:
    """Positions where target evidence may lie; None means anywhere

    "after" and "before" are relative to the pivot; "latest" keeps the 16 latest
    of the 32 most relevant moments
    """
    if direction == "after":
        return set(range(pivot + 1, len(scores)))
    if direction == "before":
        return set(range(pivot))
    if direction == "latest":
        relevant = rank_frames(scores, times, 32)
        return set(sorted(relevant, reverse=True)[:16])
    return None


def add_frame(selected: dict[int, tuple[int, float]], index: int, priority: int, score: float) -> None:
    """Add a position, or upgrade it if it is already selected at a worse priority"""
    if index not in selected or priority < selected[index][0]:
        selected[index] = (priority, score)


def add_neighbors(
    selected: dict[int, tuple[int, float]], center: int, count: int, priority: int, score: float,
) -> None:
    """Add a center and its two adjacent candidates one priority level lower"""
    for i in range(max(0, center - 1), min(count, center + 2)):
        add_frame(selected, i, priority + (i != center), score)


def bridge_positions(start: int, end: int) -> list[int]:
    """Up to four evenly spaced positions strictly between two moments"""
    if start == end:
        return []
    low, high = sorted((start, end))
    return [low + i for i in uniform_indices(high - low + 1, 6)[1:-1]]


def finish_selection(selected: dict[int, tuple[int, float]], frames: int) -> list[int]:
    """Keep the `frames` best positions by priority, then score, in time order"""
    ranked = sorted(selected, key=lambda i: (selected[i][0], -selected[i][1], i))
    return sorted(ranked[:frames])
