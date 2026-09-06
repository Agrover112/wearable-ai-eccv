#!/usr/bin/env python3
"""Build compact rescue and regression-guard sets for LongQA iteration."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import load_jsonl, normalize_answer, sample_key
from run_generate_longqa_proofpack import compile_temporal_program_v2


RESCUE_QUOTAS = {
    "endpoint_option_consensus_wrong_unscored": 14,
    "correct_direct_candidate_available_but_rank_views_miss": 10,
    "fusion_regression": 8,
    "correct_seen_by_one_rank_view_but_not_selected": 5,
    "all_direct_candidates_and_both_rank_views_miss": 3,
}


def _answers(path: str) -> dict[str, str]:
    answers: dict[str, str] = {}
    for row in load_jsonl(path):
        key = sample_key(row)
        if key in answers:
            raise RuntimeError(f"duplicate prediction key in {path}: {key}")
        answers[key] = normalize_answer(
            row.get("mcq_answer_parsed") or row.get("mcq_answer")
        )
    return answers


def _stable_value(seed: int, key: str) -> int:
    digest = hashlib.sha1(f"{seed}|{key}".encode()).hexdigest()
    return int(digest, 16)


def _diverse_take(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    selected_counts: dict[str, Counter[str]],
    base_score,
) -> list[dict[str, Any]]:
    remaining = list(rows)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < count:
        def score(row: dict[str, Any]) -> tuple[float, int]:
            category = str(row.get("category", ""))
            operator = str(row.get("operator", ""))
            tags = [str(tag) for tag in row.get("question_tags", ["other"])]
            diversity = 3.0 / (1 + selected_counts["category"][category])
            diversity += 2.0 / (1 + selected_counts["operator"][operator])
            diversity += sum(1.0 / (1 + selected_counts["tag"][tag]) for tag in tags)
            return base_score(row) + diversity, -_stable_value(seed, row["sample_key"])

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        selected_counts["category"][str(chosen.get("category", ""))] += 1
        selected_counts["operator"][str(chosen.get("operator", ""))] += 1
        for tag in chosen.get("question_tags", ["other"]):
            selected_counts["tag"][str(tag)] += 1
    if len(selected) != count:
        raise RuntimeError(f"requested {count} rows but only selected {len(selected)}")
    return selected


def _sample_record(
    index: int,
    annotation: dict[str, Any],
    role: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    key = sample_key(annotation)
    return {
        "index": index,
        "key": key,
        "video_path": annotation.get("video_path", ""),
        "question": annotation.get("question", ""),
        "category": annotation.get("category", ""),
        "answer": normalize_answer(annotation.get("mcq_answer")),
        "operator": compile_temporal_program_v2(
            str(annotation.get("question", ""))
        ).operator,
        "selection_role": role,
        **metadata,
    }


def _write_subset(
    path: str,
    annotations_path: str,
    seed: int,
    samples: list[dict[str, Any]],
    description: str,
) -> None:
    payload = {
        "source": str(Path(annotations_path).resolve()),
        "n": len(samples),
        "seed": seed,
        "diagnostic_only": True,
        "selection_uses_validation_labels": True,
        "description": description,
        "promotion_rule": (
            "Use this set for rapid diagnosis only. Confirm any frozen method on a "
            "balanced fold that was not used to design it before full validation."
        ),
        "samples": samples,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(samples)} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--fusion-predictions", required=True)
    parser.add_argument("--errors", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--rescue-output", required=True)
    parser.add_argument("--guard-output", required=True)
    parser.add_argument("--combined-output", required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    annotations = load_jsonl(args.annotations)
    indexed = {sample_key(row): (index, row) for index, row in enumerate(annotations)}
    truth = {
        key: normalize_answer(row.get("mcq_answer"))
        for key, (_, row) in indexed.items()
    }
    fusion = _answers(args.fusion_predictions)
    candidates: dict[str, dict[str, str]] = {}
    for spec in args.candidate:
        name, path = spec.split("=", 1)
        candidates[name] = _answers(path)

    required = set(indexed)
    for name, values in {"fusion": fusion, **candidates}.items():
        missing = required - set(values)
        if missing:
            raise RuntimeError(f"{name} is missing {len(missing)} rows")

    error_rows = load_jsonl(args.errors)
    errors_by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in error_rows:
        errors_by_bucket.setdefault(str(row["failure_bucket"]), []).append(row)

    counts = {"category": Counter(), "operator": Counter(), "tag": Counter()}
    rescue_rows: list[dict[str, Any]] = []
    for bucket, quota in RESCUE_QUOTAS.items():
        chosen = _diverse_take(
            errors_by_bucket.get(bucket, []),
            quota,
            args.seed,
            counts,
            lambda row: 0.25 * len(set(row.get("candidate_answers", {}).values())),
        )
        for row in chosen:
            index, annotation = indexed[row["sample_key"]]
            rescue_rows.append(
                _sample_record(
                    index,
                    annotation,
                    "rescue",
                    {
                        "failure_bucket": bucket,
                        "question_tags": row.get("question_tags", ["other"]),
                        "correct_direct_candidates": row.get("correct_candidates", []),
                        "direct_answer_count": len(
                            set(row.get("candidate_answers", {}).values())
                        ),
                    },
                )
            )

    error_keys = {row["sample_key"] for row in error_rows}
    guard_candidates: list[dict[str, Any]] = []
    for key, (index, annotation) in indexed.items():
        if key in error_keys or fusion[key] != truth[key]:
            continue
        answers = [values[key] for values in candidates.values()]
        distinct = len(set(answers))
        alternatives = sum(answer != fusion[key] for answer in answers)
        if distinct < 2:
            continue
        guard_candidates.append(
            {
                "sample_key": key,
                "index": index,
                "annotation": annotation,
                "category": annotation.get("category", ""),
                "operator": compile_temporal_program_v2(
                    str(annotation.get("question", ""))
                ).operator,
                "question_tags": ["candidate_disagreement"],
                "direct_answer_count": distinct,
                "alternative_votes": alternatives,
            }
        )

    guard_counts = {"category": Counter(), "operator": Counter(), "tag": Counter()}
    chosen_guards = _diverse_take(
        guard_candidates,
        20,
        args.seed + 1,
        guard_counts,
        lambda row: 3.0 * row["direct_answer_count"] + row["alternative_votes"],
    )
    guard_rows = [
        _sample_record(
            row["index"],
            row["annotation"],
            "guard",
            {
                "failure_bucket": None,
                "question_tags": row["question_tags"],
                "direct_answer_count": row["direct_answer_count"],
                "alternative_votes": row["alternative_votes"],
                "reference_fusion_correct": True,
            },
        )
        for row in chosen_guards
    ]

    rescue_rows.sort(key=lambda row: row["index"])
    guard_rows.sort(key=lambda row: row["index"])
    combined = sorted(rescue_rows + guard_rows, key=lambda row: row["index"])
    _write_subset(
        args.rescue_output,
        args.annotations,
        args.seed,
        rescue_rows,
        "Forty diverse errors made by the 618/700 mean-probability fusion pipeline.",
    )
    _write_subset(
        args.guard_output,
        args.annotations,
        args.seed,
        guard_rows,
        "Twenty difficult fusion-correct rows with strong direct-candidate disagreement.",
    )
    _write_subset(
        args.combined_output,
        args.annotations,
        args.seed,
        combined,
        "Combined rescue40 and guard20 diagnostic suite for rapid iteration.",
    )


if __name__ == "__main__":
    main()
