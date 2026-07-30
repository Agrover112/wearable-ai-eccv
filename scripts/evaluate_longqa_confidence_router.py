#!/usr/bin/env python3
"""Evaluate a confidence-only disagreement router with nested grouped CV."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "baselines" / "longqa"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import load_jsonl, normalize_answer, sample_key


SYSTEMS = ("q35_pivot", "q35_uniform", "q3_verifier")
VIEWS = ("pivot", "uniform", "mixed", "blind")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--majority-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    return parser.parse_args()


def fold_for(group: str, folds: int, salt: str) -> int:
    digest = hashlib.sha1(f"{salt}|{group}".encode()).hexdigest()
    return int(digest[:12], 16) % folds


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    regularization: float,
) -> np.ndarray:
    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = features @ weights
        probabilities = sigmoid(logits)
        loss = -np.mean(
            labels * np.log(np.maximum(probabilities, 1e-12))
            + (1.0 - labels) * np.log(np.maximum(1.0 - probabilities, 1e-12))
        )
        penalty = 0.5 * regularization * np.sum(weights[1:] ** 2)
        gradient = features.T @ (probabilities - labels) / len(labels)
        gradient[1:] += regularization * weights[1:]
        return float(loss + penalty), gradient

    result = minimize(
        objective,
        np.zeros(features.shape[1], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"Logistic router fit failed: {result.message}")
    return np.asarray(result.x)


def standardized(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_scaled = (train - mean) / scale
    test_scaled = (test - mean) / scale
    return (
        np.column_stack([np.ones(len(train_scaled)), train_scaled]),
        np.column_stack([np.ones(len(test_scaled)), test_scaled]),
    )


def candidate_features(record: dict[str, Any], candidate: str) -> list[float]:
    view_scores = dict(record["view_scores"])
    view_scores["blind"] = record["blind_scores"]
    values: list[float] = []
    for view in VIEWS:
        scores = view_scores[view]
        probabilities = scores["probabilities"]
        values.extend(
            [
                float(probabilities[candidate]),
                float(scores["logprobs"][candidate]),
                float(probabilities[candidate])
                - max(
                    float(probabilities[other])
                    for other in "ABCD"
                    if other != candidate
                ),
                float(scores["entropy"]),
                float(scores["predicted"] == candidate),
            ]
        )
    vote_counts = record["vote_counts"]
    values.extend(
        [
            float(vote_counts.get(candidate, 0)),
            float(candidate == "C"),
            float(record["agreement_pattern"] == "all_different"),
        ]
    )
    return values


def build_examples(
    records: list[dict[str, Any]],
    gold_by_key: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]], list[str]]:
    features: list[list[float]] = []
    labels: list[float] = []
    owners: list[tuple[int, str]] = []
    groups: list[str] = []
    for row_index, record in enumerate(records):
        candidates = sorted(set(record["candidate_answers"].values()))
        for candidate in candidates:
            features.append(candidate_features(record, candidate))
            labels.append(float(candidate == gold_by_key[record["sample_key"]]))
            owners.append((row_index, candidate))
            groups.append(str(record["video_path"]))
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
        owners,
        groups,
    )


def select_rows(
    probabilities: np.ndarray,
    owners: list[tuple[int, str]],
) -> dict[int, tuple[str, float]]:
    selected: dict[int, tuple[str, float]] = {}
    for probability, (row_index, candidate) in zip(probabilities, owners):
        current = selected.get(row_index)
        if current is None or probability > current[1]:
            selected[row_index] = (candidate, float(probability))
    return selected


def routing_accuracy(
    probabilities: np.ndarray,
    owners: list[tuple[int, str]],
    records: list[dict[str, Any]],
    gold_by_key: dict[str, str],
) -> float:
    selected = select_rows(probabilities, owners)
    correct = sum(
        selected[index][0] == gold_by_key[record["sample_key"]]
        for index, record in enumerate(records)
    )
    return correct / max(1, len(records))


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.features)
    annotations = load_jsonl(args.annotations)
    majority_predictions = load_jsonl(args.majority_predictions)
    gold_by_key = {
        sample_key(row): normalize_answer(row.get("mcq_answer") or row.get("answer"))
        for row in annotations
    }
    majority_by_key = {
        sample_key(row): normalize_answer(
            row.get("mcq_answer_parsed") or row.get("mcq_answer")
        )
        for row in majority_predictions
    }
    missing_majority = set(gold_by_key) - set(majority_by_key)
    if missing_majority:
        raise RuntimeError(
            f"Majority predictions are missing {len(missing_majority)} annotation rows"
        )
    features, labels, owners, groups = build_examples(records, gold_by_key)
    groups_array = np.asarray(groups)
    row_groups = [str(record["video_path"]) for record in records]
    outer_folds = np.asarray(
        [fold_for(group, args.folds, "outer") for group in groups]
    )
    row_outer_folds = [
        fold_for(group, args.folds, "outer") for group in row_groups
    ]
    candidates_c = (0.03, 0.1, 0.3, 1.0, 3.0)
    oof_probabilities = np.zeros(len(features), dtype=np.float64)
    chosen_regularization: dict[int, float] = {}

    for outer_fold in range(args.folds):
        train_mask = outer_folds != outer_fold
        test_mask = outer_folds == outer_fold
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)
        inner_scores: dict[float, list[float]] = defaultdict(list)
        for inner_fold in range(args.inner_folds):
            inner_assignments = np.asarray(
                [
                    fold_for(group, args.inner_folds, f"inner-{outer_fold}")
                    for group in groups_array[train_mask]
                ]
            )
            inner_train_local = np.flatnonzero(inner_assignments != inner_fold)
            inner_test_local = np.flatnonzero(inner_assignments == inner_fold)
            inner_train = train_indices[inner_train_local]
            inner_test = train_indices[inner_test_local]
            if not len(inner_train) or not len(inner_test):
                continue
            train_x, test_x = standardized(
                features[inner_train], features[inner_test]
            )
            inner_records_indices = sorted(
                {owners[index][0] for index in inner_test}
            )
            row_remap = {
                original: local
                for local, original in enumerate(inner_records_indices)
            }
            inner_owners = [
                (row_remap[owners[index][0]], owners[index][1])
                for index in inner_test
            ]
            inner_records = [records[index] for index in inner_records_indices]
            for regularization in candidates_c:
                weights = fit_logistic(
                    train_x, labels[inner_train], regularization
                )
                probabilities = sigmoid(test_x @ weights)
                inner_scores[regularization].append(
                    routing_accuracy(
                        probabilities,
                        inner_owners,
                        inner_records,
                        gold_by_key,
                    )
                )
        best_regularization = max(
            candidates_c,
            key=lambda value: (
                np.mean(inner_scores[value]) if inner_scores[value] else -1.0,
                -value,
            ),
        )
        chosen_regularization[outer_fold] = best_regularization
        train_x, test_x = standardized(
            features[train_indices], features[test_indices]
        )
        weights = fit_logistic(
            train_x, labels[train_indices], best_regularization
        )
        oof_probabilities[test_indices] = sigmoid(test_x @ weights)

    selected = select_rows(oof_probabilities, owners)
    output_rows = []
    router_correct = 0
    majority_correct = 0
    oracle_correct = 0
    for index, record in enumerate(records):
        gold = gold_by_key[record["sample_key"]]
        answers = list(record["candidate_answers"].values())
        counts = Counter(answers)
        top = max(counts.values())
        tied = {answer for answer, count in counts.items() if count == top}
        majority = next(answer for answer in answers if answer in tied)
        chosen, probability = selected[index]
        router_correct += int(chosen == gold)
        majority_correct += int(majority == gold)
        oracle_correct += int(gold in set(answers))
        output_rows.append(
            {
                **record,
                "gold_answer": gold,
                "majority_answer": majority,
                "router_answer": chosen,
                "router_probability": probability,
                "router_correct": chosen == gold,
                "majority_correct": majority == gold,
                "oracle_correct": gold in set(answers),
                "outer_fold": row_outer_folds[index],
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    summary = {
        "rows": len(records),
        "router_correct": router_correct,
        "router_accuracy": round(router_correct / len(records), 6),
        "majority_correct": majority_correct,
        "majority_accuracy": round(majority_correct / len(records), 6),
        "oracle_correct": oracle_correct,
        "oracle_accuracy": round(oracle_correct / len(records), 6),
        "full_majority_correct": sum(
            majority_by_key[key] == gold for key, gold in gold_by_key.items()
        ),
        "implied_full_router_correct": (
            sum(majority_by_key[key] == gold for key, gold in gold_by_key.items())
            - majority_correct
            + router_correct
        ),
        "implied_full_router_accuracy": round(
            (
                sum(
                    majority_by_key[key] == gold
                    for key, gold in gold_by_key.items()
                )
                - majority_correct
                + router_correct
            )
            / len(gold_by_key),
            6,
        ),
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "chosen_regularization": chosen_regularization,
        "feature_policy": "confidence_only_no_category_or_operator",
        "evaluation_note": (
            "Implied full score uses out-of-fold choices on disagreements and "
            "the fixed majority elsewhere; it is an analysis estimate, not a "
            "single fitted deployment model."
        ),
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
