#!/usr/bin/env python3
"""Measure candidate stability across cyclic option placements."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def winner(scores: dict, candidates: set[str]) -> str:
    return max(
        candidates,
        key=lambda candidate: float(scores["probabilities"][candidate]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.features)
    views = ("pivot", "uniform", "mixed", "blind")
    summaries = {}
    for view in views:
        stable = 0
        mean_agreement = 0.0
        aggregate_matches_mode = 0
        for row in rows:
            candidates = set(row["candidate_answers"].values())
            rotation_winners = []
            for rotation in row.get("rotation_scores", []):
                scores = (
                    rotation["blind_scores"]
                    if view == "blind"
                    else rotation["view_scores"][view]
                )
                rotation_winners.append(winner(scores, candidates))
            if not rotation_winners:
                raise RuntimeError(
                    "Rotation records are missing; use --option-rotations > 1"
                )
            counts = Counter(rotation_winners)
            mode, mode_count = counts.most_common(1)[0]
            aggregate_scores = (
                row["blind_scores"]
                if view == "blind"
                else row["view_scores"][view]
            )
            aggregate = winner(aggregate_scores, candidates)
            stable += int(len(counts) == 1)
            mean_agreement += mode_count / len(rotation_winners)
            aggregate_matches_mode += int(aggregate == mode)
        summaries[view] = {
            "rows": len(rows),
            "fully_stable": stable,
            "fully_stable_rate": round(stable / len(rows), 6),
            "mean_rotation_agreement": round(
                mean_agreement / len(rows), 6
            ),
            "aggregate_matches_rotation_mode": aggregate_matches_mode,
        }
    payload = {
        "rows": len(rows),
        "option_rotations": sorted(
            {int(row.get("option_rotations", 1)) for row in rows}
        ),
        "candidate_restricted_stability": summaries,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
