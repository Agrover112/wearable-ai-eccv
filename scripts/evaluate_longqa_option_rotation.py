#!/usr/bin/env python3
"""Create label-free option-rotation policies and report their accuracy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "baselines" / "longqa"
sys.path.insert(0, str(STARTER))

from longqa_utils import apply_subset, load_jsonl, normalize_answer, sample_key


def indexed(path: str) -> dict[str, dict]:
    return {sample_key(row): row for row in load_jsonl(path)}


def answer(row: dict) -> str:
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))


def set_answer(row: dict, selected: str, policy: str, feature: dict) -> dict:
    output = dict(row)
    output.update(
        {
            "mcq_answer": selected,
            "mcq_answer_raw": selected,
            "mcq_answer_parsed": selected,
            "prompt_variant": f"qwen35_27b_option_rotation_{policy}",
            "option_rotation_policy": policy,
            "option_rotation_previous": answer(row),
            "option_rotation_average": feature["rotation_average_answer"],
            "option_rotation_probabilities": feature["averaged_probabilities"],
            "option_rotation_vote_counts": feature["semantic_vote_counts"],
            "option_rotation_count": feature["option_rotations"],
            "option_rotation_seconds": feature["rotation_seconds"],
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    annotations = apply_subset(load_jsonl(args.annotations), args.subset_file)
    features = {row["sample_key"]: row for row in load_jsonl(args.features)}
    primary = indexed(args.primary)
    endpoint = indexed(args.endpoint)
    required = {sample_key(row) for row in annotations}
    for name, values in (("features", features), ("primary", primary), ("endpoint", endpoint)):
        missing = required - set(values)
        if missing:
            raise RuntimeError(f"{name} is missing {len(missing)} rows")

    policies = ("average", "consensus", "endpoint_confirmed")
    outputs = {policy: [] for policy in policies}
    summaries = {
        policy: {"correct": 0, "changed": 0, "fixes": 0, "regressions": 0}
        for policy in policies
    }
    primary_correct = 0
    for row in annotations:
        key = sample_key(row)
        feature = features[key]
        previous = answer(primary[key])
        endpoint_answer = answer(endpoint[key])
        truth = normalize_answer(row["mcq_answer"])
        primary_is_correct = previous == truth
        primary_correct += primary_is_correct
        average = feature["rotation_average_answer"]
        vote_counts = {k: int(v) for k, v in feature["semantic_vote_counts"].items()}
        consensus_answer, consensus_count = max(
            vote_counts.items(), key=lambda item: (item[1], item[0])
        )
        selected = {
            "average": average,
            "consensus": consensus_answer if consensus_count >= 3 else previous,
            "endpoint_confirmed": (
                average if average != previous and average == endpoint_answer else previous
            ),
        }
        for policy, value in selected.items():
            outputs[policy].append(set_answer(primary[key], value, policy, feature))
            is_correct = value == truth
            summaries[policy]["correct"] += is_correct
            summaries[policy]["changed"] += value != previous
            summaries[policy]["fixes"] += not primary_is_correct and is_correct
            summaries[policy]["regressions"] += primary_is_correct and not is_correct

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for policy, rows in outputs.items():
        with (output_dir / f"predictions_{policy}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        summary = summaries[policy]
        summary.update(
            {
                "policy": policy,
                "rows": len(annotations),
                "primary_correct": primary_correct,
                "accuracy": round(summary["correct"] / len(annotations), 6),
                "net_gain": summary["fixes"] - summary["regressions"],
                "uses_labels_for_selection": False,
                "prediction_distribution": dict(Counter(answer(row) for row in rows)),
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
