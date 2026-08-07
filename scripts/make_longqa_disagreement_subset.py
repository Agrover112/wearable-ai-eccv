#!/usr/bin/env python3
"""Create a subset containing rows where supplied prediction files disagree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import apply_subset, load_jsonl, normalize_answer, sample_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--parent-subset", required=True)
    parser.add_argument("--predictions", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = apply_subset(load_jsonl(args.annotations), args.parent_subset)
    indexed = []
    for path in args.predictions:
        indexed.append({sample_key(row): row for row in load_jsonl(path)})
    selected = []
    for row in rows:
        key = sample_key(row)
        answers = {
            normalize_answer(values[key].get("mcq_answer_parsed") or values[key].get("mcq_answer"))
            for values in indexed
        }
        if len(answers) > 1:
            selected.append(row)
    payload = {
        "source": str(Path(args.annotations).resolve()),
        "parent_subset": str(Path(args.parent_subset).resolve()),
        "selection": "semantic disagreement among supplied predictions",
        "n": len(selected),
        "samples": selected,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(selected)} disagreement rows to {args.output}")


if __name__ == "__main__":
    main()
