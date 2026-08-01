#!/usr/bin/env python3
"""Train on dev140 disagreements and evaluate on the disjoint val560 rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "baselines" / "longqa"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import load_jsonl, normalize_answer, sample_key
from evaluate_longqa_confidence_router import (
    VIEWS,
    build_examples,
    fit_logistic,
    fold_for,
    routing_accuracy,
    select_rows,
    sigmoid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--majority-predictions", required=True)
    parser.add_argument("--train-subset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--inner-salt", default="strict-holdout")
    parser.add_argument(
        "--views",
        nargs="+",
        choices=VIEWS,
        default=list(VIEWS),
    )
    return parser.parse_args()


def subset_keys(path: str) -> set[str]:
    payload = json.loads(Path(path).read_text())
    keys = {
        str(sample["key"])
        for sample in payload.get("samples", [])
        if sample.get("key")
    }
    if not keys:
        raise RuntimeError(f"Subset has no sample keys: {path}")
    return keys


def scale_features(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_scaled = np.column_stack([np.ones(len(train)), (train - mean) / scale])
    test_scaled = np.column_stack([np.ones(len(test)), (test - mean) / scale])
    return train_scaled, test_scaled, mean, scale


def majority_answer(record: dict[str, Any]) -> str:
    answers = list(record["candidate_answers"].values())
    counts = Counter(answers)
    top = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == top}
    return next(answer for answer in answers if answer in tied)


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.features)
    annotations = load_jsonl(args.annotations)
    majority_predictions = load_jsonl(args.majority_predictions)
    train_keys = subset_keys(args.train_subset)
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
    train_records = [row for row in records if row["sample_key"] in train_keys]
    test_records = [row for row in records if row["sample_key"] not in train_keys]
    if not train_records or not test_records:
        raise RuntimeError(
            f"Expected non-empty train/test disagreements, got "
            f"{len(train_records)}/{len(test_records)}"
        )

    views = tuple(args.views)
    train_features, train_labels, train_owners, train_groups = build_examples(
        train_records,
        gold_by_key,
        views=views,
        include_option_identity=False,
    )
    test_features, _test_labels, test_owners, _test_groups = build_examples(
        test_records,
        gold_by_key,
        views=views,
        include_option_identity=False,
    )

    candidates_c = (0.03, 0.1, 0.3, 1.0, 3.0)
    inner_assignments = np.asarray(
        [
            fold_for(group, args.inner_folds, args.inner_salt)
            for group in train_groups
        ]
    )
    inner_scores: dict[float, list[float]] = defaultdict(list)
    for inner_fold in range(args.inner_folds):
        inner_train = np.flatnonzero(inner_assignments != inner_fold)
        inner_test = np.flatnonzero(inner_assignments == inner_fold)
        if not len(inner_train) or not len(inner_test):
            continue
        inner_train_x, inner_test_x, _mean, _scale = scale_features(
            train_features[inner_train], train_features[inner_test]
        )
        inner_record_indices = sorted(
            {train_owners[index][0] for index in inner_test}
        )
        row_remap = {
            original: local
            for local, original in enumerate(inner_record_indices)
        }
        inner_owners = [
            (row_remap[train_owners[index][0]], train_owners[index][1])
            for index in inner_test
        ]
        inner_records = [train_records[index] for index in inner_record_indices]
        for regularization in candidates_c:
            weights = fit_logistic(
                inner_train_x,
                train_labels[inner_train],
                regularization,
            )
            inner_scores[regularization].append(
                routing_accuracy(
                    sigmoid(inner_test_x @ weights),
                    inner_owners,
                    inner_records,
                    gold_by_key,
                )
            )
    regularization = max(
        candidates_c,
        key=lambda value: (
            np.mean(inner_scores[value]) if inner_scores[value] else -1.0,
            -value,
        ),
    )
    train_x, test_x, feature_mean, feature_scale = scale_features(
        train_features, test_features
    )
    weights = fit_logistic(train_x, train_labels, regularization)
    selected = select_rows(sigmoid(test_x @ weights), test_owners)

    output_rows = []
    router_correct = 0
    disagreement_majority_correct = 0
    oracle_correct = 0
    for index, record in enumerate(test_records):
        gold = gold_by_key[record["sample_key"]]
        majority = majority_answer(record)
        chosen, probability = selected[index]
        router_correct += int(chosen == gold)
        disagreement_majority_correct += int(majority == gold)
        oracle_correct += int(gold in set(record["candidate_answers"].values()))
        output_rows.append(
            {
                **record,
                "gold_answer": gold,
                "majority_answer": majority,
                "router_answer": chosen,
                "router_probability": probability,
                "router_correct": chosen == gold,
                "majority_correct": majority == gold,
                "oracle_correct": gold
                in set(record["candidate_answers"].values()),
                "evaluation_split": "val560_heldout",
            }
        )

    heldout_keys = set(gold_by_key) - train_keys
    heldout_majority_correct = sum(
        majority_by_key[key] == gold_by_key[key] for key in heldout_keys
    )
    heldout_router_correct = (
        heldout_majority_correct
        - disagreement_majority_correct
        + router_correct
    )
    summary = {
        "train_subset": str(Path(args.train_subset).resolve()),
        "train_questions": len(train_keys),
        "heldout_questions": len(heldout_keys),
        "train_disagreements": len(train_records),
        "heldout_disagreements": len(test_records),
        "heldout_disagreement_router_correct": router_correct,
        "heldout_disagreement_majority_correct": disagreement_majority_correct,
        "heldout_disagreement_oracle_correct": oracle_correct,
        "heldout_majority_correct": heldout_majority_correct,
        "heldout_router_correct": heldout_router_correct,
        "heldout_majority_accuracy": round(
            heldout_majority_correct / len(heldout_keys), 6
        ),
        "heldout_router_accuracy": round(
            heldout_router_correct / len(heldout_keys), 6
        ),
        "regularization": regularization,
        "inner_fold_scores": {
            str(value): [round(score, 6) for score in scores]
            for value, scores in inner_scores.items()
        },
        "views": views,
        "feature_policy": "option_invariant_no_category_or_operator",
    }
    model_payload = {
        "format": "longqa_confidence_router_v1",
        "regularization": regularization,
        "views": views,
        "include_option_identity": False,
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "weights": weights.tolist(),
        "train_subset": str(Path(args.train_subset).resolve()),
        "train_sample_keys": sorted(train_keys),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    Path(args.summary_output).write_text(json.dumps(summary, indent=2) + "\n")
    Path(args.model_output).write_text(json.dumps(model_payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
