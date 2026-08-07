#!/usr/bin/env python3
"""Shortcut-aware diagnostics for EgoLongQA predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import compute_diagnostics, load_jsonl, sample_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LongQA diagnostics.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--annotations",
        default=str(
            REPO_ROOT
            / "data/wearable-ai/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    golden = load_jsonl(args.annotations)
    preds = load_jsonl(args.predictions)
    golden_by_key = {sample_key(row): row for row in golden}
    aligned_golden = []
    aligned_preds = []
    seen = set()
    for pred in preds:
        key = sample_key(pred)
        if key in golden_by_key and key not in seen:
            aligned_golden.append(golden_by_key[key])
            aligned_preds.append(pred)
            seen.add(key)
    golden, preds = aligned_golden, aligned_preds
    if not preds:
        raise RuntimeError("No prediction rows matched the annotations")
    result = compute_diagnostics(golden, preds, run_id=args.run_id)
    print(
        f"accuracy={result['accuracy_raw']:.4f} "
        f"correct={result['correct']}/{result['total']} "
        f"always_c={result['always_c_accuracy']:.4f} "
        f"margin_c={result['margin_over_always_c']:+.4f} "
        f"non_c={result['non_c_accuracy']:.4f} "
        f"temporal={result['temporal_question_accuracy']:.4f}"
    )
    print("predicted distribution:", result["answer_distribution_predicted"])
    print("gold distribution:", result["answer_distribution_gold"])
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote diagnostics to {args.output}")


if __name__ == "__main__":
    main()
