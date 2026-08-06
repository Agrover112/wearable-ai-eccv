#!/usr/bin/env python3
"""Grouped OOF router for independent candidates and a larger-model judge."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "baselines" / "longqa"
sys.path.insert(0, str(STARTER_KIT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_longqa_confidence_router import fit_logistic, fold_for, sigmoid, standardized
from longqa_utils import load_jsonl, normalize_answer, sample_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--fallback-predictions", required=True)
    parser.add_argument("--judge-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=0.1)
    return parser.parse_args()


def answer(row: dict) -> str:
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))


def main() -> None:
    args = parse_args()
    annotations = load_jsonl(args.annotations)
    gold = {
        sample_key(row): normalize_answer(row.get("mcq_answer") or row.get("answer"))
        for row in annotations
    }
    fallback = {sample_key(row): answer(row) for row in load_jsonl(args.fallback_predictions)}
    records = [
        row
        for row in load_jsonl(args.judge_predictions)
        if row.get("multicandidate_judge_applied")
    ]
    labels = sorted(
        {
            label
            for row in records
            for label in row.get("candidate_answers", {})
        }
    )
    features: list[list[float]] = []
    targets: list[float] = []
    owners: list[tuple[int, str]] = []
    groups: list[str] = []
    for row_index, row in enumerate(records):
        key = sample_key(row)
        answers = row["candidate_answers"]
        counts = Counter(answers.values())
        proposed = answer(row)
        candidates = sorted(set(answers.values()) | {proposed})
        top_vote = max(counts.values())
        for candidate in candidates:
            values = [
                float(counts.get(candidate, 0)) / max(1, len(answers)),
                float(counts.get(candidate, 0) == top_vote),
                float(candidate == proposed),
                float(candidate == fallback[key]),
                float(len(counts)) / max(1, len(answers)),
            ]
            values.extend(float(answers.get(label) == candidate) for label in labels)
            features.append(values)
            targets.append(float(candidate == gold[key]))
            owners.append((row_index, candidate))
            groups.append(str(row["video_path"]))

    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    folds = np.asarray([fold_for(group, args.folds, "q35-27b-router") for group in groups])
    probabilities = np.zeros(len(matrix), dtype=np.float64)
    for fold in range(args.folds):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        if not len(train) or not len(test):
            continue
        train_x, test_x = standardized(matrix[train], matrix[test])
        weights = fit_logistic(train_x, target[train], args.regularization)
        probabilities[test] = sigmoid(test_x @ weights)

    selected: dict[int, tuple[str, float]] = {}
    for probability, (row_index, candidate) in zip(probabilities, owners):
        current = selected.get(row_index)
        if current is None or probability > current[1]:
            selected[row_index] = (candidate, float(probability))

    fallback_correct = sum(fallback[key] == value for key, value in gold.items())
    disagreement_fallback_correct = 0
    router_correct = 0
    oracle_correct = 0
    output_rows = []
    for row_index, row in enumerate(records):
        key = sample_key(row)
        chosen, probability = selected[row_index]
        candidates = set(row["candidate_answers"].values()) | {answer(row)}
        disagreement_fallback_correct += int(fallback[key] == gold[key])
        router_correct += int(chosen == gold[key])
        oracle_correct += int(gold[key] in candidates)
        output_rows.append(
            {
                "sample_key": key,
                "video_path": row.get("video_path"),
                "gold_answer": gold[key],
                "fallback_answer": fallback[key],
                "judge_answer": answer(row),
                "router_answer": chosen,
                "router_probability": probability,
                "router_correct": chosen == gold[key],
            }
        )
    implied = fallback_correct - disagreement_fallback_correct + router_correct
    summary = {
        "disagreement_rows": len(records),
        "candidate_labels": labels,
        "fallback_correct": fallback_correct,
        "disagreement_fallback_correct": disagreement_fallback_correct,
        "router_correct": router_correct,
        "disagreement_oracle_correct": oracle_correct,
        "implied_full_router_correct": implied,
        "implied_full_router_accuracy": round(implied / len(gold), 6),
        "folds": args.folds,
        "regularization": args.regularization,
        "feature_policy": "votes_source_identity_large_judge_match_no_option_identity",
        "evaluation_note": "Grouped out-of-fold analysis; no deployable fit is exported.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    Path(args.summary_output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
