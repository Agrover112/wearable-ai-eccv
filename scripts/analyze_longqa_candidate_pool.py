#!/usr/bin/env python3
"""Summarize accuracy, agreement, majority, and oracle for LongQA candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import apply_subset, load_jsonl, normalize_answer, sample_key


def load_candidate(spec: str) -> tuple[str, dict[str, dict]]:
    name, path = spec.split("=", 1)
    rows = load_jsonl(path)
    indexed = {sample_key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError(f"Duplicate keys in {name}: {path}")
    return name, indexed


def answer(row: dict) -> str:
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file")
    parser.add_argument("--pred", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = apply_subset(load_jsonl(args.annotations), args.subset_file)
    candidates = dict(load_candidate(spec) for spec in args.pred)
    names = list(candidates)
    if len(names) < 2:
        raise ValueError("At least two candidates are required")

    required = {sample_key(row) for row in rows}
    for name, indexed in candidates.items():
        missing = required - set(indexed)
        if missing:
            raise RuntimeError(f"{name} is missing {len(missing)} selected rows")

    candidate_correct = Counter()
    unique_correct = Counter()
    agreement = Counter()
    agreement_correct = Counter()
    majority_correct = 0
    oracle_correct = 0
    all_different_fallbacks = 0
    for row in rows:
        key = sample_key(row)
        truth = normalize_answer(row.get("mcq_answer"))
        answers = {name: answer(candidates[name][key]) for name in names}
        correct_names = [name for name, value in answers.items() if value == truth]
        for name in correct_names:
            candidate_correct[name] += 1
        if len(correct_names) == 1:
            unique_correct[correct_names[0]] += 1
        oracle_correct += int(bool(correct_names))

        counts = Counter(answers.values())
        top_count = max(counts.values())
        if top_count == len(names):
            pattern = "unanimous"
        elif top_count >= 2:
            pattern = "majority"
        else:
            pattern = "all_different"
            all_different_fallbacks += 1
        agreement[pattern] += 1

        tied = {value for value, count in counts.items() if count == top_count}
        selected = next(answers[name] for name in names if answers[name] in tied)
        selected_correct = selected == truth
        majority_correct += int(selected_correct)
        agreement_correct[pattern] += int(selected_correct)

    summary = {
        "rows": len(rows),
        "candidate_order": names,
        "all_different_fallback": names[0],
        "candidate_correct": dict(candidate_correct),
        "candidate_accuracy": {
            name: candidate_correct[name] / len(rows) for name in names
        },
        "unique_correct": dict(unique_correct),
        "majority_correct": majority_correct,
        "majority_accuracy": majority_correct / len(rows),
        "oracle_correct": oracle_correct,
        "oracle_accuracy": oracle_correct / len(rows),
        "agreement": dict(agreement),
        "agreement_correct": dict(agreement_correct),
        "all_different_fallbacks": all_different_fallbacks,
        "uses_labels_for_predictions": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
