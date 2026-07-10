#!/usr/bin/env python3
"""Create a stable stratified EgoLongQA dev subset."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "baselines" / "longqa"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import has_temporal_cue, load_jsonl, normalize_answer, sample_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make a stratified LongQA subset.")
    parser.add_argument(
        "--annotations",
        default=str(
            REPO_ROOT / "data/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
        ),
    )
    parser.add_argument("--n", type=int, default=140)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.annotations)
    rng = random.Random(args.seed)
    strata: dict[tuple[str, str, bool], list[dict]] = defaultdict(list)
    for idx, row in enumerate(rows):
        item = dict(row)
        item["_index"] = idx
        strata[
            (
                str(row.get("category", "")),
                normalize_answer(row.get("mcq_answer", "")),
                has_temporal_cue(row.get("question", "")),
            )
        ].append(item)

    selected: list[dict] = []
    quotas: list[tuple[float, tuple[str, str, bool], int]] = []
    for key, items in strata.items():
        exact = len(items) * args.n / max(len(rows), 1)
        quota = int(exact)
        if quota > 0:
            chosen = rng.sample(items, min(quota, len(items)))
            selected.extend(chosen)
        quotas.append((exact - quota, key, quota))

    selected_keys = {sample_key(row) for row in selected}
    for _frac, key, _quota in sorted(quotas, reverse=True):
        if len(selected) >= args.n:
            break
        remaining = [row for row in strata[key] if sample_key(row) not in selected_keys]
        if not remaining:
            continue
        row = rng.choice(remaining)
        selected.append(row)
        selected_keys.add(sample_key(row))

    if len(selected) < args.n:
        remaining = [row for row in rows if sample_key(row) not in selected_keys]
        selected.extend(rng.sample(remaining, min(args.n - len(selected), len(remaining))))

    selected = sorted(selected[: args.n], key=lambda row: int(row.get("_index", 0)))
    output = {
        "source": os.path.abspath(args.annotations),
        "n": len(selected),
        "seed": args.seed,
        "samples": [
            {
                "index": row["_index"],
                "key": sample_key(row),
                "video_path": row.get("video_path", ""),
                "question": row.get("question", ""),
                "category": row.get("category", ""),
                "mcq_answer": row.get("mcq_answer", ""),
                "temporal": has_temporal_cue(row.get("question", "")),
            }
            for row in selected
        ],
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {len(selected)} samples to {args.output}")


if __name__ == "__main__":
    main()
