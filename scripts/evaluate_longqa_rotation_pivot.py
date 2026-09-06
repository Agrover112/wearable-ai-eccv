#!/usr/bin/env python3
"""Apply the label-free rotation-averaged pivot rule to a LongQA subset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "data" / "wearable-ai" / "starter_kit"
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
    parser.add_argument(
        "--view",
        default="pivot",
        help="Single view in features.jsonl to use (default: pivot).",
    )
    parser.add_argument(
        "--fusion-views",
        nargs="+",
        default=None,
        help="Average log scores from these views before candidate-restricted selection.",
    )
    parser.add_argument(
        "--allow-missing-fusion-views",
        action="store_true",
        help="Average only available requested views on rows lacking optional evidence.",
    )
    parser.add_argument(
        "--agreement-policy",
        choices=("all_disagreements", "all_different_only", "conservative"),
        default="all_disagreements",
    )
    parser.add_argument(
        "--two-one-min-margin",
        type=float,
        default=0.10,
        help="For conservative routing, minimum fused probability gain over majority.",
    )
    parser.add_argument(
        "--two-one-min-view-support",
        type=int,
        default=2,
        help="For conservative routing, views preferring the dissenting answer.",
    )
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


def normalized_probabilities(logprobs: dict[str, float]) -> dict[str, float]:
    maximum = max(logprobs.values())
    weights = {
        letter: math.exp(float(value) - maximum)
        for letter, value in logprobs.items()
    }
    denominator = sum(weights.values())
    return {
        letter: value / max(denominator, 1e-12)
        for letter, value in weights.items()
    }


def select_view_probabilities(feature: dict, args: argparse.Namespace) -> dict[str, float]:
    view_scores = feature["view_scores"]
    if args.fusion_views:
        missing = set(args.fusion_views) - set(view_scores)
        if missing and not args.allow_missing_fusion_views:
            raise RuntimeError(f"Feature row is missing fusion views: {sorted(missing)}")
        available = [view for view in args.fusion_views if view in view_scores]
        if not available:
            raise RuntimeError("Feature row has none of the requested fusion views")
        averaged = {
            letter: sum(
                float(view_scores[view]["logprobs"][letter])
                for view in available
            )
            / len(available)
            for letter in "ABCD"
        }
        return normalized_probabilities(averaged)
    if args.view not in view_scores:
        raise RuntimeError(f"Feature row is missing requested view: {args.view}")
    return {
        letter: float(probability)
        for letter, probability in view_scores[args.view]["probabilities"].items()
    }


def should_apply_selection(
    feature: dict,
    previous: str,
    selected: str,
    probabilities: dict[str, float],
    args: argparse.Namespace,
) -> bool:
    if selected == previous:
        return False
    pattern = feature.get("agreement_pattern")
    if args.agreement_policy == "all_disagreements":
        return True
    if pattern == "all_different":
        return True
    if args.agreement_policy == "all_different_only":
        return False
    margin = probabilities[selected] - probabilities[previous]
    if margin < args.two_one_min_margin:
        return False
    requested = args.fusion_views or [args.view]
    support = 0
    for view in requested:
        scores = feature.get("view_scores", {}).get(view)
        if scores and (
            float(scores["probabilities"][selected])
            > float(scores["probabilities"][previous])
        ):
            support += 1
    return support >= args.two_one_min_view_support


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
            view_probabilities = select_view_probabilities(feature, args)
            candidate_probabilities = {
                candidate: float(view_probabilities[candidate])
                for candidate in candidates
            }
            proposed = max(
                candidates, key=candidate_probabilities.get
            )
            if should_apply_selection(
                feature, previous, proposed, candidate_probabilities, args
            ):
                selected = proposed
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
                "rotation_score_view": args.view if not args.fusion_views else None,
                "rotation_fusion_views": args.fusion_views,
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
        "policy": (
            "candidate_restricted_rotation_averaged_"
            + (
                "fusion_" + "_".join(args.fusion_views)
                if args.fusion_views
                else args.view
            )
            + "_argmax"
        ),
        "agreement_policy": args.agreement_policy,
        "two_one_min_margin": args.two_one_min_margin,
        "two_one_min_view_support": args.two_one_min_view_support,
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
