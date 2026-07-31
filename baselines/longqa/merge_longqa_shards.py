#!/usr/bin/env python3
"""Merge disjoint LongQA prediction or metadata shards in annotation order."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from longqa_utils import load_jsonl, sample_key


def merge_rows(
    reference_rows: list[dict[str, Any]],
    shards: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    # Metadata rows carry sample_key directly; predictions carry question fields
    indexed = {
        str(row.get("sample_key") or sample_key(row)): row
        for shard in shards
        for row in shard
    }
    merged = [dict(indexed[sample_key(row)]) for row in reference_rows]
    for index, row in enumerate(merged):
        if "sample_key" in row and "index" in row:
            row["index"] = index
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", default=None)
    return parser.parse_args()


def main() -> None:
    from run_evaluation import evaluate_longqa, write_results

    args = parse_args()
    reference = load_jsonl(args.reference)
    merged = merge_rows(reference, [load_jsonl(path) for path in args.shard])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        for row in merged:
            handle.write(json.dumps(row) + "\n")
    print(f"Merged {len(merged)} rows into {args.output}")

    if args.eval_output:
        results = evaluate_longqa(reference, merged)
        write_results(args.eval_output, results)
        print(
            f"LongQA Accuracy: {results['accuracy']:.4f} "
            f"({results['correct']}/{results['total']})"
        )
        print(f"Results written to {args.eval_output}")


if __name__ == "__main__":
    main()
