#!/usr/bin/env python3
"""Apply a fixed routing policy to multi-candidate judge predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = REPO_ROOT / "baselines" / "longqa"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import apply_subset, load_jsonl, normalize_answer, sample_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--fallback-predictions", required=True)
    parser.add_argument("--judge-predictions", required=True)
    parser.add_argument(
        "--policy",
        choices=(
            "all_disagreements",
            "candidate_supported",
            "plurality_ties_only",
            "all_different_only",
        ),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def _answer(row: dict) -> str:
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))


def _set_answer(row: dict, answer: str) -> dict:
    updated = dict(row)
    updated["mcq_answer"] = answer
    updated["mcq_answer_raw"] = answer
    updated["mcq_answer_parsed"] = answer
    return updated


def main() -> None:
    args = parse_args()
    annotations = apply_subset(load_jsonl(args.annotations), args.subset_file)
    fallback = {sample_key(row): row for row in load_jsonl(args.fallback_predictions)}
    judge = {sample_key(row): row for row in load_jsonl(args.judge_predictions)}
    output_rows = []
    correct = fallback_correct = changed = fixes = regressions = 0
    candidate_missing_fixes = 0
    applied = 0
    judge_seconds = []
    for annotation in annotations:
        key = sample_key(annotation)
        if key not in fallback or key not in judge:
            raise RuntimeError(f"Missing prediction row: {key}")
        previous = _answer(fallback[key])
        proposed = _answer(judge[key])
        candidates = set(judge[key].get("candidate_answers", {}).values())
        pattern = judge[key].get("candidate_agreement_pattern")
        use = bool(judge[key].get("multicandidate_judge_applied"))
        if args.policy == "candidate_supported":
            use = use and proposed in candidates
        elif args.policy == "plurality_ties_only":
            counts = judge[key].get("candidate_vote_counts", {})
            maximum = max(counts.values())
            use = use and sum(count == maximum for count in counts.values()) > 1
        elif args.policy == "all_different_only":
            use = use and pattern == "all_different"
        selected = proposed if use else previous
        gold = normalize_answer(annotation.get("mcq_answer") or annotation.get("answer"))
        was_correct = previous == gold
        is_correct = selected == gold
        fallback_correct += int(was_correct)
        correct += int(is_correct)
        applied += int(use)
        if judge[key].get("multicandidate_judge_applied"):
            judge_seconds.append(float(judge[key]["multicandidate_judge_seconds"]))
        changed += int(selected != previous)
        fixes += int(not was_correct and is_correct)
        regressions += int(was_correct and not is_correct)
        candidate_missing_fixes += int(not was_correct and is_correct and gold not in candidates)
        prediction = _set_answer(fallback[key], selected)
        prediction.update(
            {
                "large_judge_policy": args.policy,
                "large_judge_applied": use,
                "large_judge_previous": previous,
                "large_judge_proposed": proposed,
                "large_judge_candidate_supported": proposed in candidates,
            }
        )
        output_rows.append(prediction)
    total = len(annotations)
    summary = {
        "policy": args.policy,
        "rows": total,
        "fallback_correct": fallback_correct,
        "fallback_accuracy": round(fallback_correct / total, 6),
        "judge_correct": correct,
        "judge_accuracy": round(correct / total, 6),
        "applied": applied,
        "changed": changed,
        "fixes": fixes,
        "regressions": regressions,
        "net_gain": fixes - regressions,
        "candidate_missing_fixes": candidate_missing_fixes,
        "judge_seconds_mean": (
            round(statistics.mean(judge_seconds), 3) if judge_seconds else None
        ),
        "judge_seconds_max": round(max(judge_seconds), 3) if judge_seconds else None,
        "judge_rows_over_300_seconds": sum(value > 300 for value in judge_seconds),
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
