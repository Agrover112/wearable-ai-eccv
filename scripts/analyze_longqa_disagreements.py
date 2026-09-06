#!/usr/bin/env python3
"""Analyze paired LongQA predictions and export disagreement rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = ROOT / "data/wearable-ai/starter_kit"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import load_jsonl, normalize_answer, sample_key  # noqa: E402
from run_generate_longqa_proofpack import compile_temporal_program  # noqa: E402


def _by_key(path: str) -> dict[str, dict[str, Any]]:
    return {sample_key(row): row for row in load_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LongQA run disagreements.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--first-label", default="first")
    parser.add_argument("--second-label", default="second")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    annotations = load_jsonl(args.annotations)
    gold = {sample_key(row): row for row in annotations}
    first = _by_key(args.first)
    second = _by_key(args.second)
    keys = [key for key in gold if key in first and key in second]

    rows = []
    outcomes = Counter()
    by_operator: dict[str, Counter[str]] = defaultdict(Counter)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for key in keys:
        gold_answer = normalize_answer(gold[key].get("mcq_answer", ""))
        first_answer = normalize_answer(
            first[key].get("mcq_answer_parsed") or first[key].get("mcq_answer", "")
        )
        second_answer = normalize_answer(
            second[key].get("mcq_answer_parsed") or second[key].get("mcq_answer", "")
        )
        first_ok = first_answer == gold_answer
        second_ok = second_answer == gold_answer
        if first_ok and second_ok:
            outcome = "both_correct"
        elif first_ok:
            outcome = "first_only"
        elif second_ok:
            outcome = "second_only"
        else:
            outcome = "both_wrong"
        operator = compile_temporal_program(gold[key]["question"]).operator
        category = str(gold[key].get("category", ""))
        outcomes[outcome] += 1
        by_operator[operator][outcome] += 1
        by_category[category][outcome] += 1
        if first_answer != second_answer:
            rows.append(
                {
                    "sample_key": key,
                    "video_path": gold[key].get("video_path", ""),
                    "question": gold[key].get("question", ""),
                    "mcq_options": gold[key].get("mcq_options", ""),
                    "gold_answer": gold_answer,
                    args.first_label: first_answer,
                    args.second_label: second_answer,
                    "first_correct": first_ok,
                    "second_correct": second_ok,
                    "operator": operator,
                    "category": category,
                }
            )

    summary = {
        "first_label": args.first_label,
        "second_label": args.second_label,
        "aligned_samples": len(keys),
        "disagreements": len(rows),
        "outcomes": dict(outcomes),
        "oracle_correct": outcomes["both_correct"]
        + outcomes["first_only"]
        + outcomes["second_only"],
        "by_operator": {key: dict(value) for key, value in sorted(by_operator.items())},
        "by_category": {key: dict(value) for key, value in sorted(by_category.items())},
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Disagreement rows written to {output_jsonl}")


if __name__ == "__main__":
    main()
