#!/usr/bin/env python3
"""Apply the label-free rotation-averaged pivot rule to a LongQA subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "baselines" / "longqa"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import apply_subset, load_jsonl, normalize_answer, sample_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--majority-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def answer(row: dict) -> str:
    return normalize_answer(
        row.get("mcq_answer_parsed") or row.get("mcq_answer")
    )


def set_answer(row: dict, selected: str) -> dict:
    updated = dict(row)
    updated["mcq_answer"] = selected
    updated["mcq_answer_raw"] = selected
    updated["mcq_answer_parsed"] = selected
    return updated


def main() -> None:
    args = parse_args()
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
        sample_key(row): row for row in load_jsonl(args.majority_predictions)
    }
    features = {
        row["sample_key"]: row for row in load_jsonl(args.features)
    }
    missing = set(gold) - set(majority)
    if missing:
        raise RuntimeError(f"Majority predictions are missing {len(missing)} rows")

    output_rows = []
    majority_correct = 0
    pivot_correct = 0
    disagreement_majority_correct = 0
    disagreement_pivot_correct = 0
    changed = 0
    fixes = 0
    regressions = 0
    both_wrong_changes = 0
    for annotation in annotations:
        key = sample_key(annotation)
        previous = answer(majority[key])
        selected = previous
        feature = features.get(key)
        candidate_probabilities = None
        if feature is not None:
            candidates = sorted(set(feature["candidate_answers"].values()))
            pivot_scores = feature["view_scores"]["pivot"]
            candidate_probabilities = {
                candidate: float(pivot_scores["probabilities"][candidate])
                for candidate in candidates
            }
            selected = max(
                candidates, key=candidate_probabilities.get
            )
        truth = gold[key]
        majority_is_correct = previous == truth
        pivot_is_correct = selected == truth
        majority_correct += int(majority_is_correct)
        pivot_correct += int(pivot_is_correct)
        if feature is not None:
            disagreement_majority_correct += int(majority_is_correct)
            disagreement_pivot_correct += int(pivot_is_correct)
            changed += int(selected != previous)
            fixes += int(not majority_is_correct and pivot_is_correct)
            regressions += int(majority_is_correct and not pivot_is_correct)
            both_wrong_changes += int(
                not majority_is_correct
                and not pivot_is_correct
                and selected != previous
            )
        prediction = set_answer(majority[key], selected)
        prediction.update(
            {
                "rotation_pivot_applied": feature is not None,
                "rotation_pivot_previous": previous,
                "rotation_pivot_probabilities": candidate_probabilities,
                "rotation_pivot_option_rotations": (
                    feature.get("option_rotations") if feature else None
                ),
            }
        )
        output_rows.append(prediction)

    summary = {
        "subset_rows": len(annotations),
        "disagreement_rows": len(features),
        "majority_correct": majority_correct,
        "majority_accuracy": round(majority_correct / len(annotations), 6),
        "rotation_pivot_correct": pivot_correct,
        "rotation_pivot_accuracy": round(pivot_correct / len(annotations), 6),
        "disagreement_majority_correct": disagreement_majority_correct,
        "disagreement_rotation_pivot_correct": disagreement_pivot_correct,
        "changed": changed,
        "fixes": fixes,
        "regressions": regressions,
        "both_wrong_changes": both_wrong_changes,
        "policy": "candidate_restricted_rotation_averaged_pivot_argmax",
        "uses_labels": False,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    Path(args.summary_output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
