#!/usr/bin/env python3
"""Filter a keyed LongQA JSONL artifact into annotation/subset order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import apply_subset, load_jsonl, sample_key


def _artifact_key(row: dict[str, Any]) -> str:
    return str(row.get("sample_key") or sample_key(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = apply_subset(load_jsonl(args.annotations), args.subset_file)
    indexed: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(args.input):
        key = _artifact_key(record)
        if key in indexed:
            raise RuntimeError(f"duplicate input key: {key}")
        indexed[key] = record
    keys = [sample_key(row) for row in rows]
    missing = set(keys) - set(indexed)
    if missing:
        raise RuntimeError(f"input artifact is missing {len(missing)} subset rows")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for key in keys:
            handle.write(json.dumps(indexed[key]) + "\n")
    print(f"Wrote {len(keys)} ordered rows to {output}")


if __name__ == "__main__":
    main()
