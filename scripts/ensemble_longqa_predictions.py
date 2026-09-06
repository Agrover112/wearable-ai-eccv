#!/usr/bin/env python3
"""Ensemble/routing utilities for EgoLongQA prediction files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import (
    apply_subset,
    build_prediction_row,
    compute_diagnostics,
    load_jsonl,
    normalize_answer,
    sample_key,
)

HYBRID_CATEGORIES = {
    "Travel-Sightseeing (Outdoors)",
    "Gardening",
    "Outdoor Activities and Sports",
    "Daily Activities",
    "Travel-Sightseeing (Indoors)",
    "Hiking-Outdoors",
}

UNIFORM_CATEGORIES = {
    "Pets, social gatherings with friends and family",
    "Hobbies-Daily Activities",
    "Sightseeing",
    "Fashion Advice",
    "Shopping",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble LongQA predictions.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument(
        "--subset-file",
        default=None,
        help="Optional JSON subset; only these annotation rows are ensembled.",
    )
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Named prediction path, e.g. uniform64=path/to/predictions.jsonl",
    )
    parser.add_argument(
        "--mode",
        choices=["majority_vote", "agreement_category_route"],
        default="majority_vote",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", default=None)
    return parser.parse_args()


def load_named_preds(items: list[str]) -> tuple[list[str], dict[str, dict[str, dict]]]:
    names: list[str] = []
    by_name: dict[str, dict[str, dict]] = {}
    for item in items:
        name, path = item.split("=", 1)
        names.append(name)
        rows = load_jsonl(path)
        by_name[name] = {sample_key(row): row for row in rows}
    return names, by_name


def choose_majority(row: dict, names: list[str], preds: dict[str, dict[str, dict]]) -> str:
    key = sample_key(row)
    answers = [normalize_answer(preds[name].get(key, {}).get("mcq_answer", "")) for name in names]
    counts = Counter(a for a in answers if a)
    if not counts:
        return ""
    top_count = counts.most_common(1)[0][1]
    tied = {letter for letter, count in counts.items() if count == top_count}
    for answer in answers:
        if answer in tied:
            return answer
    return ""


def choose_route(row: dict, names: list[str], preds: dict[str, dict[str, dict]]) -> str:
    key = sample_key(row)
    uniform_name = "uniform64" if "uniform64" in preds else names[0]
    hybrid_name = "hybrid" if "hybrid" in preds else names[min(1, len(names) - 1)]
    uniform = normalize_answer(preds[uniform_name].get(key, {}).get("mcq_answer", ""))
    hybrid = normalize_answer(preds[hybrid_name].get(key, {}).get("mcq_answer", ""))
    if uniform and uniform == hybrid:
        return uniform
    category = str(row.get("category", ""))
    if category in UNIFORM_CATEGORIES and uniform:
        return uniform
    if hybrid:
        return hybrid
    return uniform or choose_majority(row, names, preds)


def main() -> None:
    args = parse_args()
    golden = apply_subset(load_jsonl(args.annotations), args.subset_file)
    names, preds = load_named_preds(args.pred)
    missing_by_model = {
        name: [
            sample_key(row)
            for row in golden
            if sample_key(row) not in preds[name]
        ]
        for name in names
    }
    missing_by_model = {
        name: keys for name, keys in missing_by_model.items() if keys
    }
    if missing_by_model:
        details = ", ".join(
            f"{name}={len(keys)}" for name, keys in missing_by_model.items()
        )
        raise RuntimeError(
            f"Prediction inputs do not cover the selected rows: {details}"
        )
    out_rows = []
    disagreements = 0
    for row in golden:
        key = sample_key(row)
        available = [normalize_answer(preds[name].get(key, {}).get("mcq_answer", "")) for name in names]
        disagreements += int(len({a for a in available if a}) > 1)
        answer = (
            choose_majority(row, names, preds)
            if args.mode == "majority_vote"
            else choose_route(row, names, preds)
        )
        pred = build_prediction_row(row, answer, prompt_variant="ensemble")
        pred["ensemble_mode"] = args.mode
        out_rows.append(pred)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote ensemble predictions to {args.output}")
    print(f"Disagreements: {disagreements}/{len(golden)}")

    if args.eval_output:
        result = compute_diagnostics(golden, out_rows, run_id=args.mode)
        os.makedirs(os.path.dirname(args.eval_output) or ".", exist_ok=True)
        with open(args.eval_output, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"accuracy={result['accuracy_raw']:.4f} "
            f"correct={result['correct']}/{result['total']}"
        )
        print(f"Wrote ensemble diagnostics to {args.eval_output}")


if __name__ == "__main__":
    main()
