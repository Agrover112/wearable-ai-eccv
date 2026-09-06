#!/usr/bin/env python3
"""Classify errors from a LongQA option-probability fusion candidate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))

from longqa_utils import load_jsonl, normalize_answer, parse_mcq_options, sample_key
from run_generate_longqa_proofpack import compile_temporal_program_v2


def _answers(path: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in load_jsonl(path):
        key = sample_key(row)
        if key in result:
            raise RuntimeError(f"duplicate key in {path}: {key}")
        result[key] = normalize_answer(
            row.get("mcq_answer_parsed") or row.get("mcq_answer")
        )
    return result


def _features(paths: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            key = str(row.get("sample_key") or sample_key(row))
            if key in result:
                raise RuntimeError(f"duplicate feature key across inputs: {key}")
            result[key] = row
    return result


def _question_tags(question: str) -> list[str]:
    text = question.lower()
    tags: list[str] = []
    rules = {
        "first_last": ("first", "last"),
        "before_after": ("before", "after"),
        "repeated_occurrence": ("again", "another", "other", "second", "later", "returned"),
        "ocr_or_fine_text": ("name", "sign", "written", "label", "price", "number", "year", "tag"),
        "attribute": ("color", "wearing", "type", "kind", "shape"),
        "counting": ("how many", "number of"),
        "location": ("where", "which street", "which room"),
    }
    for label, cues in rules.items():
        if any(cue in text for cue in cues):
            tags.append(label)
    return tags or ["other"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--fusion-predictions", required=True)
    parser.add_argument("--endpoint-predictions", required=True)
    parser.add_argument("--feature", action="append", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-errors", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.annotations)
    truth = {sample_key(row): normalize_answer(row.get("mcq_answer")) for row in rows}
    fusion = _answers(args.fusion_predictions)
    endpoint = _answers(args.endpoint_predictions)
    candidates: dict[str, dict[str, str]] = {"endpoint27": endpoint}
    for spec in args.candidate:
        name, path = spec.split("=", 1)
        candidates[name] = _answers(path)
    features = _features(args.feature)
    required = set(truth)
    for name, predictions in {"fusion": fusion, **candidates}.items():
        if required - set(predictions):
            raise RuntimeError(f"{name} is missing {len(required - set(predictions))} rows")
    if required - set(features):
        raise RuntimeError(f"rank features are missing {len(required - set(features))} rows")

    errors: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    by_category: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_operator: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_tag: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = sample_key(row)
        gold = truth[key]
        predicted = fusion[key]
        if predicted == gold:
            continue
        feature = features[key]
        scored = bool(feature.get("rank_fusion_applied"))
        endpoint_top = feature.get("endpoint_view", {}).get("top_answer") if scored else None
        option_top = feature.get("option_view", {}).get("top_answer") if scored else None
        correct_candidates = [name for name, values in candidates.items() if values[key] == gold]
        if endpoint[key] == gold:
            bucket = "fusion_regression"
        elif scored and gold in {endpoint_top, option_top}:
            bucket = "correct_seen_by_one_rank_view_but_not_selected"
        elif correct_candidates:
            bucket = "correct_direct_candidate_available_but_rank_views_miss"
        elif scored:
            bucket = "all_direct_candidates_and_both_rank_views_miss"
        else:
            bucket = "endpoint_option_consensus_wrong_unscored"

        operator = compile_temporal_program_v2(row.get("question", "")).operator
        tags = _question_tags(str(row.get("question", "")))
        candidate_answers = {name: values[key] for name, values in candidates.items()}
        record: dict[str, Any] = {
            "sample_key": key,
            "video_path": row.get("video_path"),
            "category": row.get("category"),
            "operator": operator,
            "question_tags": tags,
            "question": row.get("question"),
            "options": parse_mcq_options(row.get("mcq_options", "")),
            "gold": gold,
            "fusion_answer": predicted,
            "endpoint_answer": endpoint[key],
            "candidate_answers": candidate_answers,
            "correct_candidates": correct_candidates,
            "candidate_vote_counts": dict(Counter(candidate_answers.values())),
            "rank_scored": scored,
            "failure_bucket": bucket,
        }
        if scored:
            record["endpoint_rank_view"] = feature["endpoint_view"]
            record["option_rank_view"] = feature["option_view"]
            record["gold_endpoint_rank"] = feature["endpoint_view"]["ranks"][gold]
            record["gold_option_rank"] = feature["option_view"]["ranks"][gold]
        errors.append(record)
        bucket_counts[bucket] += 1
        by_category[str(row.get("category", ""))][bucket] += 1
        by_operator[operator][bucket] += 1
        for tag in tags:
            by_tag[tag][bucket] += 1

    summary = {
        "rows": len(rows),
        "fusion_correct": len(rows) - len(errors),
        "fusion_errors": len(errors),
        "failure_buckets": dict(bucket_counts),
        "candidate_pool_oracle_correct": sum(
            any(values[key] == truth[key] for values in candidates.values())
            for key in truth
        ),
        "errors_with_any_correct_direct_candidate": sum(
            bool(error["correct_candidates"]) for error in errors
        ),
        "errors_unanimous_across_direct_candidates": sum(
            len(set(error["candidate_answers"].values())) == 1 for error in errors
        ),
        "by_category": {name: dict(counts) for name, counts in by_category.items()},
        "by_operator": {name: dict(counts) for name, counts in by_operator.items()},
        "by_question_tag": {name: dict(counts) for name, counts in by_tag.items()},
        "candidate_names": list(candidates),
        "notes": [
            "Failure buckets are diagnostic proxies, not human frame-level judgments.",
            "A rank-view miss can reflect missing frames, perceptual failure, or temporal reasoning failure.",
            "Gold labels are used only for this offline analysis.",
        ],
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    output_errors = Path(args.output_errors)
    output_errors.parent.mkdir(parents=True, exist_ok=True)
    with output_errors.open("w") as handle:
        for error in errors:
            handle.write(json.dumps(error) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Error records: {output_errors}")


if __name__ == "__main__":
    main()
