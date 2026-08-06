#!/usr/bin/env python3
"""Merge complete LongQA JSONL shards in the order of a target subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "baselines" / "longqa"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import load_jsonl, sample_key


def record_key(row: dict) -> str:
    return str(row.get("sample_key") or sample_key(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--subset")
    target.add_argument("--annotations")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    subset = (
        json.loads(Path(args.subset).read_text()).get("samples", [])
        if args.subset
        else load_jsonl(args.annotations)
    )
    expected = [record_key(row) for row in subset]
    merged: dict[str, dict] = {}
    for path in args.input:
        for row in load_jsonl(path):
            key = record_key(row)
            if key in merged:
                raise RuntimeError(f"duplicate row across shards: {key}")
            merged[key] = row
    missing = [key for key in expected if key not in merged]
    extras = sorted(set(merged) - set(expected))
    if missing or extras:
        raise RuntimeError(
            f"shard coverage mismatch: missing={len(missing)}, extras={len(extras)}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for key in expected:
            handle.write(json.dumps(merged[key]) + "\n")
    print(f"Merged {len(expected)} rows into {output}")


if __name__ == "__main__":
    main()
