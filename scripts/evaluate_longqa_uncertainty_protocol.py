#!/usr/bin/env python3
"""Evaluate aligned LongQA systems with stratified paired bootstraps."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "baselines" / "longqa"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import apply_subset, load_jsonl, normalize_answer, sample_key
from run_generate_longqa_proofpack import compile_temporal_program


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        default=str(
            REPO_ROOT
            / "data/egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
        ),
    )
    parser.add_argument("--subset-file")
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat for each aligned prediction file.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def answer(row: dict) -> str:
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))


def parse_named_paths(values: list[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, received {value!r}")
        name, path = value.split("=", 1)
        if not name or name in parsed:
            raise ValueError(f"Invalid or duplicate method name: {name!r}")
        parsed[name] = path
    return parsed


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def method_summary(rows: list[dict], predictions: dict[str, dict]) -> dict:
    correct = []
    by_category: dict[str, list[bool]] = defaultdict(list)
    by_gold = {letter: [] for letter in "ABCD"}
    predicted = Counter()
    temporal = []
    non_temporal = []
    for row in rows:
        key = sample_key(row)
        truth = answer(row)
        selected = answer(predictions[key])
        is_correct = selected == truth
        correct.append(is_correct)
        by_category[str(row.get("category", "unknown"))].append(is_correct)
        if truth in by_gold:
            by_gold[truth].append(is_correct)
        predicted[selected] += 1
        target = (
            non_temporal
            if compile_temporal_program(str(row["question"])).operator == "GLOBAL"
            else temporal
        )
        target.append(is_correct)
    category_accuracy = {
        category: sum(values) / len(values)
        for category, values in sorted(by_category.items())
    }
    non_c = [value for letter, values in by_gold.items() if letter != "C" for value in values]
    return {
        "correct": sum(correct),
        "total": len(correct),
        "accuracy": sum(correct) / len(correct),
        "macro_category_accuracy": sum(category_accuracy.values()) / len(category_accuracy),
        "worst_category_accuracy": min(category_accuracy.values()),
        "category_accuracy": category_accuracy,
        "gold_letter_accuracy": {
            letter: (sum(values) / len(values) if values else None)
            for letter, values in by_gold.items()
        },
        "non_c_accuracy": sum(non_c) / len(non_c),
        "temporal_accuracy": sum(temporal) / len(temporal),
        "non_temporal_accuracy": sum(non_temporal) / len(non_temporal),
        "predicted_distribution": dict(sorted(predicted.items())),
    }


def paired_comparison(
    rows: list[dict],
    first: dict[str, dict],
    second: dict[str, dict],
    repetitions: int,
    rng: random.Random,
) -> dict:
    by_category: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    transitions = Counter()
    for row in rows:
        key = sample_key(row)
        truth = answer(row)
        a = answer(first[key]) == truth
        b = answer(second[key]) == truth
        by_category[str(row.get("category", "unknown"))].append((a, b))
        transitions[f"first_{'correct' if a else 'wrong'}__second_{'correct' if b else 'wrong'}"] += 1
    observed = (
        sum(a for pairs in by_category.values() for a, _ in pairs)
        - sum(b for pairs in by_category.values() for _, b in pairs)
    ) / len(rows)
    deltas = []
    categories = sorted(by_category)
    for _ in range(repetitions):
        first_correct = 0
        second_correct = 0
        sampled = 0
        for category in categories:
            pairs = by_category[category]
            for _ in range(len(pairs)):
                a, b = pairs[rng.randrange(len(pairs))]
                first_correct += int(a)
                second_correct += int(b)
                sampled += 1
        deltas.append((first_correct - second_correct) / sampled)
    return {
        "accuracy_difference": observed,
        "paired_bootstrap_95_ci": [quantile(deltas, 0.025), quantile(deltas, 0.975)],
        "transitions": dict(transitions),
        "bootstrap_repetitions": repetitions,
    }


def main() -> None:
    args = parse_args()
    rows = apply_subset(load_jsonl(args.annotations), args.subset_file)
    required = {sample_key(row) for row in rows}
    methods = {}
    for name, path in parse_named_paths(args.prediction).items():
        indexed = {sample_key(row): row for row in load_jsonl(path)}
        missing = required - set(indexed)
        if missing:
            raise RuntimeError(f"{name} is missing {len(missing)} required predictions")
        methods[name] = indexed

    summaries = {
        name: method_summary(rows, predictions)
        for name, predictions in methods.items()
    }
    rng = random.Random(args.seed)
    comparisons = {}
    for first_name, second_name in combinations(methods, 2):
        comparisons[f"{first_name}__minus__{second_name}"] = paired_comparison(
            rows,
            methods[first_name],
            methods[second_name],
            args.bootstrap_repetitions,
            rng,
        )
    result = {
        "rows": len(rows),
        "subset_file": args.subset_file,
        "seed": args.seed,
        "methods": summaries,
        "paired_comparisons": comparisons,
        "uncertainty_scope": "category-stratified dataset-sampling uncertainty",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
