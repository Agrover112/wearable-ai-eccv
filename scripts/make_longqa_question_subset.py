#!/usr/bin/env python3
"""Create deterministic OCR or object-reidentification LongQA subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import apply_subset, load_jsonl, sample_key


OCR_PATTERN = re.compile(
    r"\b(read|reading|written|wrote|word|words|text|sign|signage|label|"
    r"named|name|brand|price|cost|number|title|book|screen|displayed|printed|"
    r"chapter|marker|directory|warning|speed limit)\b",
    re.IGNORECASE,
)
REID_PATTERN = re.compile(
    r"\b(same|again|reappeared|reappear|first time|last time|second time|"
    r"earlier.*later|later.*earlier|where did .+ end up|what changed about|"
    r"different the second|replacing|returned|remaining roll)\b",
    re.IGNORECASE,
)


def matches_gate(row: dict, gate: str) -> bool:
    text = f"{row.get('question', '')} {row.get('mcq_options', '')}"
    return bool((OCR_PATTERN if gate == "ocr" else REID_PATTERN).search(text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--base-subset", default=None)
    parser.add_argument("--gate", choices=("ocr", "reid"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = apply_subset(load_jsonl(args.annotations), args.base_subset)
    selected = [row for row in source if matches_gate(row, args.gate)]
    payload = {
        "source": Path(args.annotations).name,
        "base_subset": Path(args.base_subset).name if args.base_subset else None,
        "gate": args.gate,
        "n": len(selected),
        "samples": [
            {
                "index": index,
                "key": sample_key(row),
                "video_path": row.get("video_path"),
                "question": row.get("question"),
                "category": row.get("category"),
                "mcq_answer": row.get("mcq_answer"),
            }
            for index, row in enumerate(selected)
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(selected)} {args.gate} rows to {args.output}")


if __name__ == "__main__":
    main()
