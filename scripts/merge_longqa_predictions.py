#!/usr/bin/env python3
"""Merge disjoint LongQA prediction subsets in annotation order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import load_jsonl, sample_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    annotations = load_jsonl(args.annotations)
    indexed = {}
    source_by_key = {}
    for path in args.predictions:
        for row in load_jsonl(path):
            key = sample_key(row)
            if key in indexed:
                raise RuntimeError(
                    f"Prediction key occurs in multiple inputs: {key}"
                )
            indexed[key] = row
            source_by_key[key] = str(Path(path).resolve())

    annotation_keys = [sample_key(row) for row in annotations]
    missing = set(annotation_keys) - set(indexed)
    extra = set(indexed) - set(annotation_keys)
    if missing or extra:
        raise RuntimeError(
            f"Prediction partition mismatch: missing={len(missing)}, "
            f"extra={len(extra)}"
        )
    output_rows = []
    for key in annotation_keys:
        row = dict(indexed[key])
        row["prediction_subset_source"] = source_by_key[key]
        output_rows.append(row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    print(
        f"Merged {len(output_rows)} predictions from "
        f"{len(args.predictions)} disjoint inputs"
    )


if __name__ == "__main__":
    main()
