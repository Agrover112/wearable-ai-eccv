#!/usr/bin/env python3
"""Audit direct LongQA candidates and classical hard-vote fusion rules."""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import apply_subset, load_jsonl, normalize_answer, sample_key


def parse_candidate(spec: str) -> tuple[str, dict[str, str]]:
    name, raw_path = spec.split("=", 1)
    rows = load_jsonl(raw_path)
    answers: dict[str, str] = {}
    for row in rows:
        key = sample_key(row)
        if key in answers:
            raise RuntimeError(f"Duplicate key in candidate {name}: {key}")
        answers[key] = normalize_answer(
            row.get("mcq_answer_parsed") or row.get("mcq_answer")
        )
    return name, answers


def accuracy(
    predictions: dict[str, str], truth: dict[str, str], keys: set[str]
) -> int:
    return sum(predictions[key] == truth[key] for key in keys)


def choose_vote(
    key: str,
    names: tuple[str, ...],
    candidates: dict[str, dict[str, str]],
    default_name: str,
    weights: dict[str, float] | None = None,
) -> str:
    scores: Counter[str] = Counter()
    for name in names:
        scores[candidates[name][key]] += 1.0 if weights is None else weights[name]
    best = max(scores.values())
    tied = {answer for answer, score in scores.items() if score == best}
    default_answer = candidates[default_name][key]
    if default_answer in tied:
        return default_answer
    return next(answer for answer in "ABCD" if answer in tied)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--dev-subset", required=True)
    parser.add_argument("--default-name", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--smoothing", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.annotations)
    truth = {sample_key(row): normalize_answer(row.get("mcq_answer")) for row in rows}
    if len(truth) != len(rows):
        raise RuntimeError("Annotations contain duplicate sample keys")
    dev_rows = apply_subset(rows, args.dev_subset)
    dev_keys = {sample_key(row) for row in dev_rows}
    full_keys = set(truth)
    heldout_keys = full_keys - dev_keys

    candidates = dict(parse_candidate(spec) for spec in args.candidate)
    names = tuple(candidates)
    if args.default_name not in candidates:
        raise ValueError("Default candidate must be included in --candidate")
    for name, predictions in candidates.items():
        missing = full_keys - set(predictions)
        if missing:
            raise RuntimeError(f"Candidate {name} is missing {len(missing)} rows")

    splits = {"dev": dev_keys, "heldout": heldout_keys, "full": full_keys}
    individual = {
        name: {
            split: {
                "correct": accuracy(predictions, truth, keys),
                "total": len(keys),
                "accuracy": accuracy(predictions, truth, keys) / len(keys),
            }
            for split, keys in splits.items()
        }
        for name, predictions in candidates.items()
    }

    default = candidates[args.default_name]
    pairwise = {}
    for name in names:
        if name == args.default_name:
            continue
        disagreements = {key for key in full_keys if candidates[name][key] != default[key]}
        pairwise[name] = {
            "disagreements": len(disagreements),
            "default_only_correct": sum(
                default[key] == truth[key] for key in disagreements
            ),
            "candidate_only_correct": sum(
                candidates[name][key] == truth[key] for key in disagreements
            ),
            "pair_oracle_correct": sum(
                default[key] == truth[key] or candidates[name][key] == truth[key]
                for key in full_keys
            ),
        }

    pool_oracle = {
        split: {
            "correct": sum(
                any(candidates[name][key] == truth[key] for name in names)
                for key in keys
            ),
            "total": len(keys),
        }
        for split, keys in splits.items()
    }
    for value in pool_oracle.values():
        value["accuracy"] = value["correct"] / value["total"]

    dev_weights = {}
    for name in names:
        correct = individual[name]["dev"]["correct"]
        incorrect = len(dev_keys) - correct
        dev_weights[name] = math.log(
            (correct + args.smoothing) / (incorrect + args.smoothing)
        )

    fusion = {}
    for label, weights in (("plurality", None), ("dev_log_odds_weighted", dev_weights)):
        predictions = {
            key: choose_vote(
                key, names, candidates, args.default_name, weights=weights
            )
            for key in full_keys
        }
        fusion[label] = {
            split: {
                "correct": accuracy(predictions, truth, keys),
                "total": len(keys),
                "accuracy": accuracy(predictions, truth, keys) / len(keys),
            }
            for split, keys in splits.items()
        }

    subset_majorities = []
    for count in range(2, len(names) + 1):
        for group in itertools.combinations(names, count):
            if args.default_name not in group:
                continue
            predictions = {
                key: choose_vote(key, group, candidates, args.default_name)
                for key in full_keys
            }
            subset_majorities.append(
                {
                    "candidates": list(group),
                    "dev_correct": accuracy(predictions, truth, dev_keys),
                    "heldout_correct": accuracy(predictions, truth, heldout_keys),
                    "full_correct": accuracy(predictions, truth, full_keys),
                }
            )
    subset_majorities.sort(
        key=lambda row: (row["dev_correct"], row["heldout_correct"]), reverse=True
    )

    agreement = {}
    for support in range(1, len(names) + 1):
        agreement[str(support)] = {}
        for split, keys in splits.items():
            selected = {
                key
                for key in keys
                if sum(
                    candidates[name][key] == default[key] for name in names
                )
                == support
            }
            agreement[str(support)][split] = {
                "rows": len(selected),
                "default_correct": accuracy(default, truth, selected),
            }

    report = {
        "default_candidate": args.default_name,
        "candidate_order": list(names),
        "individual": individual,
        "pairwise_vs_default": pairwise,
        "pool_oracle": pool_oracle,
        "fusion": fusion,
        "dev_reliability_log_odds": dev_weights,
        "default_support_buckets": agreement,
        "top_subset_majorities_selected_by_dev": subset_majorities[:20],
        "notes": [
            "All candidates supplied to this audit should be direct answerers, not derived judges.",
            "Hard-answer voting has no within-model option ranking; reciprocal-rank fusion belongs at frame retrieval unless option log probabilities are recorded.",
            "Held-out metrics are diagnostic and must not be used repeatedly to tune a claimed frozen rule.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
