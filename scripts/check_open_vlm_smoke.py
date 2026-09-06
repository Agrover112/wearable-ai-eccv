#!/usr/bin/env python3
"""Validate a small open-VLM run before promoting it to a larger split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--expected", type=int, default=5)
    parser.add_argument("--latency-limit", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.predictions)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != args.expected:
        raise SystemExit(f"FAIL: expected {args.expected} rows, found {len(rows)}")

    invalid = [
        index
        for index, row in enumerate(rows)
        if str(row.get("mcq_answer_parsed", "")).strip() not in {"A", "B", "C", "D"}
    ]
    if invalid:
        raise SystemExit(f"FAIL: invalid parsed answers at rows {invalid}")

    timings = [float(row["generation_seconds"]) for row in rows]
    over_limit = [round(value, 3) for value in timings if value > args.latency_limit]
    print(
        "SMOKE PASS: "
        f"rows={len(rows)} mean_generation_seconds={sum(timings) / len(timings):.3f} "
        f"max_generation_seconds={max(timings):.3f}"
    )
    if over_limit:
        raise SystemExit(
            f"FAIL: {len(over_limit)} samples exceed {args.latency_limit}s: {over_limit}"
        )


if __name__ == "__main__":
    main()
