#!/usr/bin/env python3
"""Evaluate direct complete-answer likelihood rules on a fixed subset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "baselines" / "longqa"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import apply_subset, load_jsonl, normalize_answer, sample_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--majority-predictions", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def majority_answer(record: dict) -> str:
    answers = list(record["candidate_answers"].values())
    counts = Counter(answers)
    top = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == top}
    return next(answer for answer in answers if answer in tied)


def score(record: dict, candidate: str, method: str) -> float:
    if method in {"pivot", "uniform", "mixed"}:
        return float(
            record["view_scores"][method][candidate]["mean_logprob"]
        )
    blind = float(record["blind_scores"][candidate]["mean_logprob"])
    visual = sum(
        float(record["view_scores"][view][candidate]["mean_logprob"])
        for view in ("pivot", "uniform", "mixed")
    ) / 3.0
    if method == "blind":
        return blind
    if method == "visual_mean":
        return visual
    if method == "all_mean":
        return (3.0 * visual + blind) / 4.0
    if method.startswith("visual_minus_blind_"):
        weight = float(method.rsplit("_", 1)[-1])
        return visual - weight * blind
    raise ValueError(method)


def main() -> None:
    args = parse_args()
    scores = load_jsonl(args.scores)
    annotations = apply_subset(
        load_jsonl(args.annotations), args.subset_file
    )
    gold = {
        sample_key(row): normalize_answer(
            row.get("mcq_answer") or row.get("answer")
        )
        for row in annotations
    }
    majority = {
        sample_key(row): normalize_answer(
            row.get("mcq_answer_parsed") or row.get("mcq_answer")
        )
        for row in load_jsonl(args.majority_predictions)
    }
    methods = (
        "pivot",
        "uniform",
        "mixed",
        "blind",
        "visual_mean",
        "all_mean",
        "visual_minus_blind_0.25",
        "visual_minus_blind_0.5",
        "visual_minus_blind_1.0",
    )
    subset_majority_correct = sum(
        majority[key] == answer for key, answer in gold.items()
    )
    disagreement_majority_correct = sum(
        majority_answer(row) == gold[row["sample_key"]] for row in scores
    )
    results = {}
    for method in methods:
        correct = 0
        changed = 0
        for row in scores:
            candidates = sorted(set(row["candidate_answers"].values()))
            selected = max(
                candidates,
                key=lambda candidate: score(row, candidate, method),
            )
            correct += int(selected == gold[row["sample_key"]])
            changed += int(selected != majority_answer(row))
        implied_correct = (
            subset_majority_correct - disagreement_majority_correct + correct
        )
        results[method] = {
            "disagreement_correct": correct,
            "disagreement_rows": len(scores),
            "changed_from_majority": changed,
            "implied_subset_correct": implied_correct,
            "implied_subset_accuracy": round(
                implied_correct / len(gold), 6
            ),
        }
    payload = {
        "subset_rows": len(gold),
        "disagreement_rows": len(scores),
        "subset_majority_correct": subset_majority_correct,
        "disagreement_majority_correct": disagreement_majority_correct,
        "methods": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
