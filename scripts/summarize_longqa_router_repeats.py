#!/usr/bin/env python3
"""Aggregate repeated confidence-router summary JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [json.loads(Path(path).read_text()) for path in args.inputs]
    scores = np.asarray(
        [row["implied_full_router_correct"] for row in rows],
        dtype=np.float64,
    )
    payload = {
        "repeats": len(rows),
        "scores": scores.astype(int).tolist(),
        "mean_correct": round(float(scores.mean()), 3),
        "mean_accuracy": round(float(scores.mean() / 700.0), 6),
        "std_correct": round(float(scores.std(ddof=1)), 3)
        if len(scores) > 1
        else 0.0,
        "min_correct": int(scores.min()),
        "max_correct": int(scores.max()),
        "feature_policy": rows[0].get("feature_policy"),
        "views": rows[0].get("views"),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
