#!/usr/bin/env python3
"""Split a LongQA subset deterministically into round-robin shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-pattern", required=True)
    args = parser.parse_args()
    if args.shards < 1 or "{shard}" not in args.output_pattern:
        raise ValueError("--shards must be positive and output pattern must contain {shard}")

    document = json.loads(Path(args.input).read_text())
    samples = document.get("samples")
    if not isinstance(samples, list):
        raise ValueError("subset JSON must contain a samples list")
    for shard in range(args.shards):
        selected = samples[shard::args.shards]
        output = Path(args.output_pattern.format(shard=shard))
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **{key: value for key, value in document.items() if key != "samples"},
            "n": len(selected),
            "shard": shard,
            "num_shards": args.shards,
            "parent_subset": str(Path(args.input).resolve()),
            "samples": selected,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote shard {shard}: {len(selected)} rows -> {output}")


if __name__ == "__main__":
    main()
