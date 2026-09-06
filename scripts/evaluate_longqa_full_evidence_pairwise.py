#!/usr/bin/env python3
"""Apply frozen abstaining policies to full-evidence pairwise scores."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import apply_subset, build_prediction_row, compute_diagnostics, load_jsonl


def pairwise_decision(
    record: dict[str, Any],
    policy: str,
    min_challenger_probability: float = 0.55,
    max_insufficient_probability: float = 0.35,
    min_mean_log_odds: float = math.log(1.5),
) -> tuple[str, dict[str, Any]]:
    baseline = str(record["baseline_answer"])
    challenger = str(record["challenger_answer"])
    if not record.get("pairwise_applied") or baseline == challenger:
        return baseline, {"switched": False, "reason": "agreement"}

    forward = record["forward_probabilities"]
    reverse = record["reverse_probabilities"]
    challenger_probs = [float(forward[challenger]), float(reverse[challenger])]
    baseline_probs = [float(forward[baseline]), float(reverse[baseline])]
    insufficient_probs = [
        float(forward["INSUFFICIENT"]), float(reverse["INSUFFICIENT"])
    ]
    challenger_wins_both = all(
        challenger_prob > max(baseline_prob, insufficient_prob)
        for challenger_prob, baseline_prob, insufficient_prob in zip(
            challenger_probs, baseline_probs, insufficient_probs
        )
    )
    mean_log_odds = sum(
        math.log(max(challenger_prob, 1e-12) / max(baseline_prob, 1e-12))
        for challenger_prob, baseline_prob in zip(challenger_probs, baseline_probs)
    ) / 2.0
    strict_checks = {
        "challenger_wins_both_orders": challenger_wins_both,
        "minimum_challenger_probability": min(challenger_probs)
        >= min_challenger_probability,
        "maximum_insufficient_probability": max(insufficient_probs)
        <= max_insufficient_probability,
        "mean_log_odds": mean_log_odds >= min_mean_log_odds,
    }
    switch = challenger_wins_both if policy == "consensus" else all(strict_checks.values())
    return (challenger if switch else baseline), {
        "switched": switch,
        "reason": policy if switch else "abstain",
        "challenger_probabilities": challenger_probs,
        "baseline_probabilities": baseline_probs,
        "insufficient_probabilities": insufficient_probs,
        "mean_log_odds": mean_log_odds,
        "strict_checks": strict_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--features", required=True)
    parser.add_argument("--policy", choices=("conservative", "consensus"), required=True)
    parser.add_argument("--min-challenger-probability", type=float, default=0.55)
    parser.add_argument("--max-insufficient-probability", type=float, default=0.35)
    parser.add_argument("--min-mean-log-odds", type=float, default=math.log(1.5))
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = apply_subset(load_jsonl(args.annotations), args.subset_file)
    features = load_jsonl(args.features)
    if len(features) != len(rows):
        raise RuntimeError(f"Feature rows={len(features)} but annotation rows={len(rows)}")

    predictions: list[dict[str, Any]] = []
    switches = 0
    for row, feature in zip(rows, features):
        if row.get("video_path") != feature.get("video_path"):
            raise RuntimeError("Feature and annotation ordering differs")
        answer, decision = pairwise_decision(
            feature,
            args.policy,
            args.min_challenger_probability,
            args.max_insufficient_probability,
            args.min_mean_log_odds,
        )
        prediction = build_prediction_row(
            row, answer, prompt_variant=f"full_evidence_pairwise_{args.policy}"
        )
        prediction.update(
            {
                "pairwise_policy": args.policy,
                "pairwise_baseline_answer": feature["baseline_answer"],
                "pairwise_challenger_answer": feature["challenger_answer"],
                "pairwise_applied": feature.get("pairwise_applied", False),
                "pairwise_decision": decision,
            }
        )
        switches += int(decision["switched"])
        predictions.append(prediction)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction) + "\n")
    result = compute_diagnostics(rows, predictions, run_id=f"pairwise_{args.policy}")
    result["pairwise_switches"] = switches
    result["pairwise_policy"] = args.policy
    with open(args.summary_output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(
        f"policy={args.policy} switches={switches} "
        f"accuracy={result['correct']}/{result['total']} ({result['accuracy']:.4f})"
    )


if __name__ == "__main__":
    main()
