#!/usr/bin/env python3
"""Evaluate label-free option-ranking fusion policies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import apply_subset, build_prediction_row, compute_diagnostics, load_jsonl, sample_key


POLICIES = (
    "endpoint",
    "endpoint_view",
    "option_view",
    "view_consensus",
    "mean_logprob",
    "mean_probability",
    "margin_weighted_probability",
    "rrf",
    "candidate_mean_logprob",
    "candidate_mean_probability",
    "cross_view_confirmation",
)


def _argmax(scores: dict[str, float], fallback: str) -> str:
    best = max(scores.values())
    tied = {letter for letter, score in scores.items() if abs(score - best) < 1e-12}
    if fallback in tied:
        return fallback
    return next(letter for letter in "ABCD" if letter in tied)


def select_answer(record: dict[str, Any], policy: str, rrf_k: float = 60.0) -> str:
    baseline = str(record["baseline_answer"])
    challenger = str(record["challenger_answer"])
    if not record.get("rank_fusion_applied"):
        return baseline
    endpoint = record["endpoint_view"]
    option = record["option_view"]
    if policy == "endpoint":
        return baseline
    if policy == "endpoint_view":
        return str(endpoint["top_answer"])
    if policy == "option_view":
        return str(option["top_answer"])
    if policy == "view_consensus":
        return str(endpoint["top_answer"]) if endpoint["top_answer"] == option["top_answer"] else baseline
    if policy == "mean_logprob":
        scores = {
            letter: (float(endpoint["logprobs"][letter]) + float(option["logprobs"][letter])) / 2
            for letter in "ABCD"
        }
        return _argmax(scores, baseline)
    if policy == "mean_probability":
        scores = {
            letter: (float(endpoint["probabilities"][letter]) + float(option["probabilities"][letter])) / 2
            for letter in "ABCD"
        }
        return _argmax(scores, baseline)
    if policy == "margin_weighted_probability":
        endpoint_weight = max(0.0, float(endpoint.get("top_margin", 0.0)))
        option_weight = max(0.0, float(option.get("top_margin", 0.0)))
        total_weight = endpoint_weight + option_weight
        if total_weight <= 1e-12:
            endpoint_weight = option_weight = 0.5
        else:
            endpoint_weight /= total_weight
            option_weight /= total_weight
        scores = {
            letter: endpoint_weight * float(endpoint["probabilities"][letter])
            + option_weight * float(option["probabilities"][letter])
            for letter in "ABCD"
        }
        return _argmax(scores, baseline)
    if policy == "rrf":
        scores = {
            letter: 1.0 / (rrf_k + int(endpoint["ranks"][letter]))
            + 1.0 / (rrf_k + int(option["ranks"][letter]))
            for letter in "ABCD"
        }
        return _argmax(scores, baseline)
    if policy == "candidate_mean_logprob":
        scores = {
            letter: (float(endpoint["logprobs"][letter]) + float(option["logprobs"][letter])) / 2
            for letter in {baseline, challenger}
        }
        return _argmax(scores, baseline)
    if policy == "candidate_mean_probability":
        scores = {
            letter: (
                float(endpoint["probabilities"][letter])
                + float(option["probabilities"][letter])
            )
            / 2
            for letter in {baseline, challenger}
        }
        return _argmax(scores, baseline)
    if policy == "cross_view_confirmation":
        challenger_wins_both = all(
            int(view["ranks"][challenger]) < int(view["ranks"][baseline])
            for view in (endpoint, option)
        )
        return challenger if challenger_wins_both else baseline
    raise ValueError(f"Unknown policy: {policy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--features", required=True)
    parser.add_argument("--reference-predictions", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = apply_subset(load_jsonl(args.annotations), args.subset_file)
    features = load_jsonl(args.features)
    if len(rows) != len(features):
        raise RuntimeError(f"Annotation rows={len(rows)}, feature rows={len(features)}")
    for row, feature in zip(rows, features):
        if sample_key(row) != feature.get("sample_key"):
            raise RuntimeError("Feature and annotation order differs")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "rows": len(rows),
        "scored_disagreements": sum(bool(row.get("rank_fusion_applied")) for row in features),
        "rrf_k": args.rrf_k,
        "uses_labels_for_predictions": False,
        "policies": {},
    }
    if args.reference_predictions:
        reference = {sample_key(row): row for row in load_jsonl(args.reference_predictions)}
        aligned = [reference[sample_key(row)] for row in rows]
        summary["reference"] = compute_diagnostics(rows, aligned, run_id="reference")

    for policy in POLICIES:
        predictions = []
        changes = 0
        for row, feature in zip(rows, features):
            answer = select_answer(feature, policy, args.rrf_k)
            changes += int(answer != feature["baseline_answer"])
            pred = build_prediction_row(row, answer, prompt_variant=f"evidence_rank_{policy}")
            pred.update(
                {
                    "rank_fusion_policy": policy,
                    "rank_fusion_baseline": feature["baseline_answer"],
                    "rank_fusion_challenger": feature["challenger_answer"],
                }
            )
            predictions.append(pred)
        diagnostics = compute_diagnostics(rows, predictions, run_id=policy)
        diagnostics["changes_from_endpoint"] = changes
        summary["policies"][policy] = diagnostics
        with (output_dir / f"predictions_{policy}.jsonl").open("w") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction) + "\n")
        print(
            f"{policy}: {diagnostics['correct']}/{diagnostics['total']} "
            f"({diagnostics['accuracy']:.4f}), changes={changes}"
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
