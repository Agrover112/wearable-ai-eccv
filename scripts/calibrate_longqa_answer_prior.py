#!/usr/bin/env python3
"""Calibrate LongQA predictions from a labeled development subset only."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


LABELS = "ABCD"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_key(row: dict) -> str:
    return f"{row['video_path']}||{row['question']}"


def answer(row: dict) -> str:
    value = row.get("mcq_answer_parsed") or row.get("mcq_answer")
    return value if value in LABELS else ""


class CategoricalCalibrator:
    def __init__(self, rows: list[tuple[str, str, str]], alpha: float, weight: float):
        self.alpha = alpha
        self.weight = weight
        self.label_counts = Counter(label for label, _, _ in rows)
        self.feature_counts = {
            "primary": {
                label: Counter(primary for gold, primary, _ in rows if gold == label)
                for label in LABELS
            },
            "endpoint": {
                label: Counter(endpoint for gold, _, endpoint in rows if gold == label)
                for label in LABELS
            },
        }
        self.total = len(rows)

    def scores(self, primary: str, endpoint: str) -> dict[str, float]:
        result = {}
        for label in LABELS:
            label_count = self.label_counts[label]
            score = math.log(
                (label_count + self.alpha) / (self.total + len(LABELS) * self.alpha)
            )
            for feature, value in (("primary", primary), ("endpoint", endpoint)):
                count = self.feature_counts[feature][label][value]
                score += self.weight * math.log(
                    (count + self.alpha) / (label_count + len(LABELS) * self.alpha)
                )
            result[label] = score
        return result

    def predict(self, primary: str, endpoint: str) -> tuple[str, dict[str, float]]:
        raw_scores = self.scores(primary, endpoint)
        maximum = max(raw_scores.values())
        normalizer = sum(math.exp(value - maximum) for value in raw_scores.values())
        probabilities = {
            label: math.exp(value - maximum) / normalizer
            for label, value in raw_scores.items()
        }
        return max(LABELS, key=probabilities.get), probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--dev-subset", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    args = parser.parse_args()

    annotations = load_jsonl(args.annotations)
    gold = {row_key(row): row["mcq_answer"] for row in annotations}
    dev_config = json.loads(args.dev_subset.read_text())
    dev_keys = {sample["key"] for sample in dev_config["samples"]}
    primary_rows = load_jsonl(args.primary)
    endpoint_rows = load_jsonl(args.endpoint)
    primary = {row_key(row): answer(row) for row in primary_rows}
    endpoint = {row_key(row): answer(row) for row in endpoint_rows}

    missing = [key for key in gold if not primary.get(key) or not endpoint.get(key)]
    if missing:
        raise ValueError(f"Missing candidate predictions for {len(missing)} rows")

    training = [(gold[key], primary[key], endpoint[key]) for key in dev_keys]
    calibrator = CategoricalCalibrator(training, args.alpha, args.feature_weight)

    predictions = []
    correct = Counter()
    changed = Counter()
    for row in primary_rows:
        key = row_key(row)
        predicted, probabilities = calibrator.predict(primary[key], endpoint[key])
        split = "dev140" if key in dev_keys else "val560"
        correct[split] += predicted == gold[key]
        changed[split] += predicted != primary[key]
        output_row = dict(row)
        output_row.update(
            {
                "mcq_answer": predicted,
                "mcq_answer_raw": predicted,
                "mcq_answer_parsed": predicted,
                "prompt_variant": "dev_prior_calibration",
                "calibration_primary": primary[key],
                "calibration_endpoint": endpoint[key],
                "calibration_probabilities": {
                    label: round(probabilities[label], 8) for label in LABELS
                },
                "calibration_training_split": "dev140_only",
                "calibration_alpha": args.alpha,
                "calibration_feature_weight": args.feature_weight,
            }
        )
        predictions.append(output_row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")

    summary = {
        "method": "categorical_bayes_answer_calibration",
        "training_rows": len(training),
        "training_split": "dev140_only",
        "alpha": args.alpha,
        "feature_weight": args.feature_weight,
        "label_counts": dict(calibrator.label_counts),
        "dev140_correct": correct["dev140"],
        "dev140_total": len(dev_keys),
        "val560_correct": correct["val560"],
        "val560_total": len(gold) - len(dev_keys),
        "full_correct": sum(correct.values()),
        "full_total": len(gold),
        "changed_from_primary": dict(changed),
        "prediction_distribution": dict(Counter(answer(row) for row in predictions)),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
