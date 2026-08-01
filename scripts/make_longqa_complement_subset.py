#!/usr/bin/env python3
"""Create the ordered complement of an existing LongQA subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "baselines" / "longqa"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import has_temporal_cue, load_jsonl, load_subset_keys, sample_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--exclude-subset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    excluded = load_subset_keys(args.exclude_subset)
    if excluded is None:
        raise RuntimeError("An exclusion subset is required")
    rows = load_jsonl(args.annotations)
    selected = [
        (index, row)
        for index, row in enumerate(rows)
        if sample_key(row) not in excluded
    ]
    payload = {
        "source": str(Path(args.annotations).resolve()),
        "n": len(selected),
        "complement_of": str(Path(args.exclude_subset).resolve()),
        "samples": [
            {
                "index": index,
                "key": sample_key(row),
                "video_path": row.get("video_path", ""),
                "question": row.get("question", ""),
                "category": row.get("category", ""),
                "mcq_answer": row.get("mcq_answer", ""),
                "temporal": has_temporal_cue(row.get("question", "")),
            }
            for index, row in selected
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(selected)} samples to {args.output}")


if __name__ == "__main__":
    main()
