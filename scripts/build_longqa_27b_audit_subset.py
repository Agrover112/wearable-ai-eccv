#!/usr/bin/env python3
"""Build a reproducible 30-row diagnostic subset from the primary error audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_order(row: dict, seed: int) -> str:
    return hashlib.sha1(f"{seed}|{row['sample_key']}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    annotations = load_jsonl(args.annotations)
    by_key = {
        f"{row['video_path']}||{row['question']}": (index, row)
        for index, row in enumerate(annotations)
    }
    errors = [row for row in load_jsonl(args.audit) if not row["rotation_correct"]]
    selected: list[tuple[str, dict]] = []
    used: set[str] = set()

    def take(label: str, predicate, count: int) -> None:
        candidates = sorted(
            (row for row in errors if row["sample_key"] not in used and predicate(row)),
            key=lambda row: stable_order(row, args.seed),
        )
        if len(candidates) < count:
            raise RuntimeError(f"Only {len(candidates)} rows available for {label}")
        for row in candidates[:count]:
            used.add(row["sample_key"])
            selected.append((label, row))

    take(
        "cross_time",
        lambda row: "cross_time_ordering" in row.get("question_type_tags", []),
        10,
    )
    take(
        "shopping_or_ocr",
        lambda row: row.get("category") == "Shopping"
        or "ocr_named_detail" in row.get("question_type_tags", []),
        10,
    )
    take(
        "missed_by_all_runs",
        lambda row: row.get("error_bucket") == "missing_all_runs",
        10,
    )

    samples = []
    for stratum, audited in selected:
        index, annotation = by_key[audited["sample_key"]]
        samples.append(
            {
                "index": index,
                "key": audited["sample_key"],
                "video_path": annotation["video_path"],
                "question": annotation["question"],
                "category": annotation.get("category"),
                "mcq_answer": annotation.get("mcq_answer"),
                "audit_stratum": stratum,
                "primary_error_bucket": audited.get("error_bucket"),
            }
        )
    payload = {
        "source": str(Path(args.annotations).resolve()),
        "audit_source": str(Path(args.audit).resolve()),
        "n": len(samples),
        "seed": args.seed,
        "samples": samples,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(samples)} rows to {args.output}")


if __name__ == "__main__":
    main()
