#!/usr/bin/env python3
"""Add one independent evidence view to cached endpoint/option rank features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from statistics import median
from typing import Any

from longqa_utils import (
    apply_subset,
    build_longqa_prompt,
    build_prediction_row,
    compute_diagnostics,
    load_jsonl,
    sample_key,
)
from run_generate_longqa_grounded import extract_frames_by_indices
from run_score_longqa_evidence_rank_fusion import _scored_view


LETTERS = "ABCD"


def _index(paths: list[str], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            key = str(row.get("sample_key") or sample_key(row))
            if key in indexed:
                raise ValueError(f"Duplicate {label} key: {key}")
            indexed[key] = row
    return indexed


def _nested_value(record: dict[str, Any], key: str) -> object:
    value: object = record
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _indices(record: dict[str, Any], key: str) -> list[int]:
    values = _nested_value(record, key)
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("frame_index")
        if value is not None:
            result.append(int(value))
    return result


def _argmax(scores: dict[str, float], fallback: str) -> str:
    maximum = max(scores.values())
    tied = [letter for letter in LETTERS if abs(scores[letter] - maximum) < 1e-12]
    return fallback if fallback in tied else tied[0]


def _confidence(view: dict[str, Any]) -> float:
    probabilities = [max(float(view["probabilities"][letter]), 1e-12) for letter in LETTERS]
    entropy = -sum(value * math.log(value) for value in probabilities) / math.log(len(LETTERS))
    return max(0.05, 1.0 - entropy)


def select_answer(record: dict[str, Any], policy: str) -> str:
    fallback = str(record["baseline_answer"])
    if not record.get("rank_fusion_applied") or "third_view" not in record:
        return fallback
    views = [record["endpoint_view"], record["option_view"], record["third_view"]]
    if policy == "endpoint_option_mean_probability":
        views = views[:2]
    if policy in {"endpoint_option_mean_probability", "three_mean_probability"}:
        scores = {
            letter: sum(float(view["probabilities"][letter]) for view in views) / len(views)
            for letter in LETTERS
        }
        return _argmax(scores, fallback)
    if policy == "three_median_probability":
        scores = {
            letter: median(float(view["probabilities"][letter]) for view in views)
            for letter in LETTERS
        }
        return _argmax(scores, fallback)
    if policy == "three_rrf":
        scores = {
            letter: sum(1.0 / (60.0 + int(view["ranks"][letter])) for view in views)
            for letter in LETTERS
        }
        return _argmax(scores, fallback)
    if policy == "entropy_weighted_probability":
        weights = [_confidence(view) for view in views]
        scores = {
            letter: sum(
                weight * float(view["probabilities"][letter])
                for weight, view in zip(weights, views)
            ) / sum(weights)
            for letter in LETTERS
        }
        return _argmax(scores, fallback)
    if policy == "uncertainty_tiebreak":
        endpoint = str(views[0]["top_answer"])
        option = str(views[1]["top_answer"])
        third = str(views[2]["top_answer"])
        if endpoint == option:
            return endpoint
        if third in {endpoint, option}:
            return third
        return fallback
    raise ValueError(f"Unknown policy: {policy}")


POLICIES = (
    "endpoint_option_mean_probability",
    "three_mean_probability",
    "three_median_probability",
    "three_rrf",
    "entropy_weighted_probability",
    "uncertainty_tiebreak",
)


def _fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "base_features": sorted(os.path.abspath(path) for path in args.base_features),
        "evidence": os.path.abspath(args.evidence_selection),
        "evidence_key": args.evidence_key,
        "view_name": args.view_name,
        "model": args.llm_model,
        "subset": os.path.abspath(args.subset_file) if args.subset_file else None,
        "schema": 1,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--base-features", action="append", required=True)
    parser.add_argument("--evidence-selection", required=True)
    parser.add_argument("--evidence-key", default="final_indices")
    parser.add_argument("--view-name", default="uncertainty")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    rows = apply_subset(load_jsonl(args.input), args.subset_file)
    base = _index(args.base_features, "base features")
    evidence = _index([args.evidence_selection], "evidence selection")
    required = {sample_key(row) for row in rows}
    for label, indexed in (("base features", base), ("evidence", evidence)):
        missing = required - set(indexed)
        if missing:
            raise RuntimeError(f"{label} is missing {len(missing)} rows")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.jsonl"
    fingerprint = _fingerprint(args)
    existing = [] if args.no_resume or not feature_path.exists() else load_jsonl(str(feature_path))
    start = 0
    for index, record in enumerate(existing[: len(rows)]):
        if record.get("sample_key") != sample_key(rows[index]) or record.get("third_view_fingerprint") != fingerprint:
            break
        start += 1
    if len(existing) != start:
        with feature_path.open("w") as handle:
            for record in existing[:start]:
                handle.write(json.dumps(record) + "\n")

    targets = sum(bool(base[sample_key(row)].get("rank_fusion_applied")) for row in rows)
    print(f"Third evidence view: rows={len(rows)} targets={targets} resume={start}")
    model = VLLMModel(
        args.llm_model,
        tp_size=1,
        concurrency=args.concurrency,
        max_frames=args.max_frames,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    begun = time.time()
    with model, feature_path.open("a" if start else "w") as handle:
        for index, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            record = dict(base[key])
            record["third_view_fingerprint"] = fingerprint
            record["third_view_name"] = args.view_name
            if record.get("rank_fusion_applied"):
                frame_indices = _indices(evidence[key], args.evidence_key)
                if len(frame_indices) != args.max_frames or len(set(frame_indices)) != args.max_frames:
                    raise RuntimeError(
                        f"Expected {args.max_frames} unique {args.view_name} frames for {key}; "
                        f"got {len(frame_indices)} values and {len(set(frame_indices))} unique"
                    )
                video_path = os.path.join(args.video_folder, str(row["video_path"]))
                frames = extract_frames_by_indices(video_path, frame_indices)
                prompt = build_longqa_prompt(row["question"], row["mcq_options"], "baseline")
                scores = model.score_choice_letters(
                    frames, [{"role": "user", "content": prompt}]
                )
                record["third_view_indices"] = frame_indices
                record["third_view"] = _scored_view(scores)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"  Third-view progress: {index + 1}/{len(rows)}")

    features = load_jsonl(str(feature_path))
    summary: dict[str, Any] = {
        "rows": len(rows),
        "scored_disagreements": targets,
        "view_name": args.view_name,
        "uses_labels_for_predictions": False,
        "policies": {},
    }
    for policy in POLICIES:
        predictions = []
        changes = 0
        for row, feature in zip(rows, features):
            selected = select_answer(feature, policy)
            changes += int(selected != feature["baseline_answer"])
            predictions.append(build_prediction_row(row, selected, f"third_view_{policy}"))
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
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())


if __name__ == "__main__":
    main()
