#!/usr/bin/env python3
"""Evaluate fixed, label-free policies over chronological block scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import apply_subset, build_prediction_row, compute_diagnostics, load_jsonl, normalize_answer, sample_key


POLICIES = (
    "primary",
    "block_mean_probability",
    "view_top2_probability",
    "adjacent_pair_probability",
    "block_rrf",
    "stable_adjacent_3",
    "stable_adjacent_4",
    "unanimous_adjacent",
)


def _index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("sample_key") or sample_key(row)): row for row in rows}


def _argmax(scores: dict[str, float], fallback: str) -> str:
    best = max(scores.values())
    tied = {letter for letter, value in scores.items() if abs(value - best) < 1e-12}
    return fallback if fallback in tied else next(letter for letter in "ABCD" if letter in tied)


def _blocks(feature: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return list(feature["endpoint_blocks"]), list(feature["option_blocks"])


def _aggregate(feature: dict[str, Any], policy: str) -> dict[str, float]:
    endpoint, option = _blocks(feature)
    all_blocks = endpoint + option
    if policy == "block_mean_probability":
        return {
            letter: sum(float(block["probabilities"][letter]) for block in all_blocks)
            / len(all_blocks)
            for letter in "ABCD"
        }
    if policy == "view_top2_probability":
        scores = {}
        for letter in "ABCD":
            endpoint_top = sorted(
                (float(block["probabilities"][letter]) for block in endpoint), reverse=True
            )[:2]
            option_top = sorted(
                (float(block["probabilities"][letter]) for block in option), reverse=True
            )[:2]
            scores[letter] = sum(endpoint_top + option_top) / 4
        return scores
    if policy in {
        "adjacent_pair_probability",
        "stable_adjacent_3",
        "stable_adjacent_4",
        "unanimous_adjacent",
    }:
        scores = {}
        for letter in "ABCD":
            view_scores = []
            for view in (endpoint, option):
                adjacent = [
                    (
                        float(view[index]["probabilities"][letter])
                        + float(view[index + 1]["probabilities"][letter])
                    )
                    / 2
                    for index in range(len(view) - 1)
                ]
                view_scores.append(max(adjacent))
            scores[letter] = sum(view_scores) / len(view_scores)
        return scores
    if policy == "block_rrf":
        return {
            letter: sum(1.0 / (60.0 + int(block["ranks"][letter])) for block in all_blocks)
            for letter in "ABCD"
        }
    raise ValueError(f"Unknown aggregate policy: {policy}")


def select_answer(feature: dict[str, Any], primary_answer: str, policy: str) -> str:
    if policy == "primary" or not feature.get("segment_fusion_applied"):
        return primary_answer
    scores = _aggregate(feature, policy)
    candidate = _argmax(scores, primary_answer)
    if policy == "unanimous_adjacent":
        trigger = feature.get("consensus_challenge", {})
        answers = list(trigger.get("answers", []))
        unanimous = (
            bool(answers)
            and int(trigger.get("alternative_votes", 0)) == len(answers)
            and trigger.get("alternative") == candidate
        )
        return candidate if unanimous else primary_answer
    if not policy.startswith("stable_adjacent_") or candidate == primary_answer:
        return candidate
    minimum_blocks = int(policy.rsplit("_", 1)[1])
    endpoint, option = _blocks(feature)
    support_by_view = [
        sum(
            float(block["probabilities"][candidate])
            > float(block["probabilities"][primary_answer])
            for block in view
        )
        for view in (endpoint, option)
    ]
    if sum(support_by_view) >= minimum_blocks and all(count >= 1 for count in support_by_view):
        return candidate
    return primary_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--primary-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = apply_subset(load_jsonl(args.annotations), args.subset_file)
    features = load_jsonl(args.features)
    primary = _index(load_jsonl(args.primary_predictions))
    if len(rows) != len(features):
        raise RuntimeError(f"Annotation rows={len(rows)}, feature rows={len(features)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "rows": len(rows),
        "target_rows": sum(bool(row.get("segment_fusion_applied")) for row in features),
        "direct_disagreements": sum(bool(row.get("direct_disagreement")) for row in features),
        "consensus_challenges": sum(
            bool(row.get("consensus_challenge", {}).get("applied")) for row in features
        ),
        "uses_labels_for_predictions": False,
        "policies": {},
    }
    for policy in POLICIES:
        predictions = []
        changes = 0
        for row, feature in zip(rows, features):
            key = sample_key(row)
            if feature.get("sample_key") != key:
                raise RuntimeError("Feature and annotation order differs")
            primary_answer = normalize_answer(
                primary[key].get("mcq_answer_parsed") or primary[key].get("mcq_answer")
            )
            answer = select_answer(feature, primary_answer, policy)
            changes += int(answer != primary_answer)
            prediction = build_prediction_row(row, answer, prompt_variant=f"segment_{policy}")
            prediction.update(
                {
                    "segment_fusion_policy": policy,
                    "segment_fusion_primary": primary_answer,
                }
            )
            predictions.append(prediction)
        diagnostics = compute_diagnostics(rows, predictions, run_id=policy)
        diagnostics["changes_from_primary"] = changes
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
