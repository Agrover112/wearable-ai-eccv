#!/usr/bin/env python3
"""Apply a saved option-invariant confidence router to majority predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import load_jsonl, normalize_answer, sample_key
from evaluate_longqa_confidence_router import candidate_features, sigmoid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--majority-predictions", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--annotations", default=None)
    return parser.parse_args()


def set_answer(row: dict[str, Any], answer: str) -> dict[str, Any]:
    updated = dict(row)
    updated["mcq_answer"] = answer
    updated["mcq_answer_raw"] = answer
    updated["mcq_answer_parsed"] = answer
    return updated


def main() -> None:
    args = parse_args()
    features = {
        row["sample_key"]: row for row in load_jsonl(args.features)
    }
    majority_rows = load_jsonl(args.majority_predictions)
    model = json.loads(Path(args.model).read_text())
    if model.get("include_option_identity"):
        raise RuntimeError("Refusing to deploy a router with option identity")
    views = tuple(model["views"])
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    fingerprint = hashlib.sha1(
        json.dumps(model, sort_keys=True).encode()
    ).hexdigest()[:16]

    output_rows = []
    changed = 0
    for majority_row in majority_rows:
        key = sample_key(majority_row)
        record = features.get(key)
        if record is None:
            output_rows.append(dict(majority_row))
            continue
        candidates = sorted(set(record["candidate_answers"].values()))
        candidate_probabilities: dict[str, float] = {}
        for candidate in candidates:
            values = np.asarray(
                candidate_features(
                    record,
                    candidate,
                    views=views,
                    include_option_identity=False,
                ),
                dtype=np.float64,
            )
            if len(values) != len(mean):
                raise RuntimeError(
                    f"Feature width mismatch for {key}: {len(values)} != {len(mean)}"
                )
            scaled = np.concatenate(([1.0], (values - mean) / scale))
            candidate_probabilities[candidate] = float(
                sigmoid(np.asarray([scaled @ weights]))[0]
            )
        answer = max(candidate_probabilities, key=candidate_probabilities.get)
        previous = normalize_answer(
            majority_row.get("mcq_answer_parsed")
            or majority_row.get("mcq_answer")
        )
        updated = set_answer(majority_row, answer)
        updated.update(
            {
                "confidence_router_applied": True,
                "confidence_router_previous": previous,
                "confidence_router_candidates": candidates,
                "confidence_router_probabilities": candidate_probabilities,
                "confidence_router_fingerprint": fingerprint,
            }
        )
        changed += int(answer != previous)
        output_rows.append(updated)

    summary: dict[str, Any] = {
        "rows": len(output_rows),
        "router_rows": len(features),
        "changed_from_majority": changed,
        "model": str(Path(args.model).resolve()),
        "router_fingerprint": fingerprint,
        "evaluation_note": (
            "The saved model was trained on the configured development subset. "
            "Use heldout_accuracy for model selection."
        ),
    }
    if args.annotations:
        gold = {
            sample_key(row): normalize_answer(
                row.get("mcq_answer") or row.get("answer")
            )
            for row in load_jsonl(args.annotations)
        }
        train_keys = set(model.get("train_sample_keys", []))
        full_correct = sum(
            normalize_answer(
                row.get("mcq_answer_parsed") or row.get("mcq_answer")
            )
            == gold[sample_key(row)]
            for row in output_rows
        )
        heldout_rows = [
            row for row in output_rows if sample_key(row) not in train_keys
        ]
        heldout_correct = sum(
            normalize_answer(
                row.get("mcq_answer_parsed") or row.get("mcq_answer")
            )
            == gold[sample_key(row)]
            for row in heldout_rows
        )
        summary.update(
            {
                "full_correct_analysis_only": full_correct,
                "full_accuracy_analysis_only": round(
                    full_correct / len(output_rows), 6
                ),
                "heldout_rows": len(heldout_rows),
                "heldout_correct": heldout_correct,
                "heldout_accuracy": round(
                    heldout_correct / len(heldout_rows), 6
                ),
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    Path(args.summary_output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
