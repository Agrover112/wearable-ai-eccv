#!/usr/bin/env python3
"""Write a transparent stage-by-stage latency summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--stage", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-seconds", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    stages: dict[str, float] = {}
    for value in args.stage:
        name, seconds = value.split("=", 1)
        if name in stages:
            raise ValueError(f"duplicate stage {name!r}")
        stages[name] = float(seconds)
    total = sum(stages.values())
    per_question = total / args.samples
    result = {
        "samples": args.samples,
        "stage_wall_seconds": stages,
        "total_wall_seconds": total,
        "amortized_seconds_per_question": per_question,
        "limit_seconds_per_question": args.limit_seconds,
        "within_amortized_limit": per_question <= args.limit_seconds,
        "measurement_scope": (
            "Fresh frame-selection caches and fresh predictions; model startup and "
            "shutdown are included once per stage."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
