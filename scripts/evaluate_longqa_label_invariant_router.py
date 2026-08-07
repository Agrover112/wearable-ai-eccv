#!/usr/bin/env python3
"""Train or apply a candidate-scoring LongQA disagreement router."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "wearable-ai" / "starter_kit"
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_longqa_confidence_router import fit_logistic, fold_for, sigmoid
from longqa_utils import apply_subset, load_jsonl, normalize_answer, sample_key
from run_generate_longqa_object_hints import classify_question
from run_generate_longqa_proofpack import compile_temporal_program_v2


OPERATORS = ("AFTER", "BEFORE", "FIRST", "LAST", "STATE_CHANGE", "MULTI_TIME", "GLOBAL")
QUESTION_TYPES = ("object_detail", "spatial", "state_or_count", "temporal", "global")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument(
        "--candidate", action="append", required=True, metavar="LABEL=JSONL"
    )
    parser.add_argument("--rotation-features")
    parser.add_argument("--proofpack")
    parser.add_argument("--mode", choices=("oof", "apply"), default="oof")
    parser.add_argument("--model-output")
    parser.add_argument("--model-input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=0.3)
    parser.add_argument("--min-switch-delta", type=float, default=0.10)
    parser.add_argument("--min-switch-probability", type=float, default=0.55)
    args = parser.parse_args()
    if args.mode == "oof" and not args.model_output:
        parser.error("--model-output is required in oof mode")
    if args.mode == "apply" and not args.model_input:
        parser.error("--model-input is required in apply mode")
    return args


def parse_candidates(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Candidate must be LABEL=JSONL: {value}")
        label, path = value.split("=", 1)
        if not label or label in result:
            raise ValueError(f"Invalid or duplicate candidate label: {label}")
        result[label] = Path(path)
    return result


def index_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {sample_key(row): row for row in load_jsonl(str(path))}


def answer(row: dict[str, Any]) -> str:
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer"))


def set_answer(row: dict[str, Any], selected: str) -> dict[str, Any]:
    updated = dict(row)
    updated["mcq_answer"] = selected
    updated["mcq_answer_raw"] = selected
    updated["mcq_answer_parsed"] = selected
    return updated


def entropy(probabilities: dict[str, float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in probabilities.values())


def retrieval_statistics(record: dict[str, Any] | None) -> dict[str, float]:
    if not record:
        return {"peak": 0.0, "margin": 0.0, "spread": 0.0, "largest_gap": 0.0}
    selected = record.get("selected", [])
    scores = sorted(
        [float(item["score"]) for item in selected if item.get("score") is not None],
        reverse=True,
    )
    timestamps = sorted(float(item.get("timestamp", 0.0)) for item in selected)
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    return {
        "peak": scores[0] if scores else 0.0,
        "margin": scores[0] - scores[1] if len(scores) > 1 else 0.0,
        "spread": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0,
        "largest_gap": max(gaps, default=0.0),
    }


def feature_names(labels: list[str], use_rotation: bool) -> list[str]:
    names = ["support_fraction", "is_primary", "distinct_fraction"]
    names.extend(f"source_{label}" for label in labels)
    names.extend(
        [
            "support_27b",
            "support_global",
            "support_retrieval",
            "direct_reasoning_agreement",
            "global_retrieval_agreement",
            "support_27b_temporal",
            "support_global_global_question",
            "support_retrieval_temporal",
            "support_global_object_detail",
            "support_retrieval_state_count",
            "retrieval_peak_x_support",
            "retrieval_margin_x_support",
            "retrieval_spread_x_support",
            "retrieval_gap_x_support",
        ]
    )
    if use_rotation:
        names.extend(
            [
                "rotation_probability",
                "rotation_min_probability",
                "rotation_variance",
                "rotation_margin",
                "rotation_entropy",
                "rotation_vote_fraction",
            ]
        )
    return names


def candidate_features(
    row: dict[str, Any],
    candidate: str,
    labels: list[str],
    answers: dict[str, str],
    primary: str,
    rotation: dict[str, Any] | None,
    proofpack: dict[str, Any] | None,
    use_rotation: bool,
) -> list[float]:
    support = {label: float(value == candidate) for label, value in answers.items()}
    count = sum(support.values())
    operator = compile_temporal_program_v2(row.get("question", "")).operator
    qtype = classify_question(row)
    temporal = float(operator != "GLOBAL")
    support_27b = max(support.get("direct27", 0.0), support.get("thinking27", 0.0))
    support_global = support.get("endpoint9", 0.0)
    support_retrieval = support.get("rotation_pivot9", 0.0) or support.get(
        "option_quota27", 0.0
    )
    stats = retrieval_statistics(proofpack)
    values = [count / len(labels), float(candidate == primary), len(set(answers.values())) / len(labels)]
    values.extend(support[label] for label in labels)
    values.extend(
        [
            support_27b,
            support_global,
            support_retrieval,
            float(answers.get("direct27") == answers.get("thinking27") == candidate),
            float(answers.get("endpoint9") == answers.get("rotation_pivot9") == candidate),
            support_27b * temporal,
            support_global * float(operator == "GLOBAL"),
            support_retrieval * temporal,
            support_global * float(qtype == "object_detail"),
            support_retrieval * float(qtype == "state_or_count"),
            stats["peak"] * support_retrieval,
            stats["margin"] * support_retrieval,
            stats["spread"] * support_retrieval,
            stats["largest_gap"] * support_retrieval,
        ]
    )
    if use_rotation:
        probabilities = (rotation or {}).get("averaged_probabilities", {})
        per_rotation = [
            float(item.get("mapped_probabilities", {}).get(candidate, 0.0))
            for item in (rotation or {}).get("rotations", [])
        ]
        probability = float(probabilities.get(candidate, 0.0))
        other = max(
            [float(value) for key, value in probabilities.items() if key != candidate],
            default=0.0,
        )
        vote_counts = (rotation or {}).get("semantic_vote_counts", {})
        values.extend(
            [
                probability,
                min(per_rotation, default=0.0),
                float(np.var(per_rotation)) if per_rotation else 0.0,
                probability - other,
                entropy({key: float(value) for key, value in probabilities.items()}),
                float(vote_counts.get(candidate, 0)) / max(1, len(per_rotation)),
            ]
        )
    return values


def fit_scaled(matrix: np.ndarray, targets: np.ndarray, regularization: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    scaled = np.column_stack([np.ones(len(matrix)), (matrix - mean) / scale])
    return fit_logistic(scaled, targets, regularization), mean, scale


def score_scaled(matrix: np.ndarray, weights: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scaled = np.column_stack([np.ones(len(matrix)), (matrix - mean) / scale])
    return sigmoid(scaled @ weights)


def main() -> None:
    args = parse_args()
    candidate_paths = parse_candidates(args.candidate)
    labels = list(candidate_paths)
    rows = apply_subset(load_jsonl(args.annotations), args.subset_file)
    indexed = {label: index_rows(path) for label, path in candidate_paths.items()}
    primary_rows = index_rows(Path(args.primary))
    rotations = index_rows(Path(args.rotation_features)) if args.rotation_features else {}
    proofpacks = index_rows(Path(args.proofpack)) if args.proofpack else {}
    required = {sample_key(row) for row in rows}
    for label, values in {**indexed, "primary": primary_rows}.items():
        missing = required - set(values)
        if missing:
            raise RuntimeError(f"{label} is missing {len(missing)} subset rows")

    use_rotation = bool(args.rotation_features)
    names = feature_names(labels, use_rotation)
    examples: list[list[float]] = []
    targets: list[float] = []
    owners: list[tuple[int, str]] = []
    groups: list[str] = []
    row_data: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        key = sample_key(row)
        answers = {label: answer(values[key]) for label, values in indexed.items()}
        primary = answer(primary_rows[key])
        proposed = sorted(set(answers.values()))
        row_data.append({"key": key, "answers": answers, "primary": primary, "proposed": proposed})
        if len(proposed) <= 1:
            continue
        truth = normalize_answer(row.get("mcq_answer") or row.get("answer"))
        for candidate in proposed:
            examples.append(
                candidate_features(
                    row, candidate, labels, answers, primary, rotations.get(key), proofpacks.get(key), use_rotation
                )
            )
            targets.append(float(candidate == truth))
            owners.append((row_index, candidate))
            groups.append(str(row.get("video_path", key)))
    matrix = np.asarray(examples, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if matrix.shape[1] != len(names):
        raise RuntimeError("Feature-name mismatch")

    if args.mode == "oof":
        probabilities = np.zeros(len(matrix), dtype=np.float64)
        folds = np.asarray([fold_for(group, args.folds, "label-invariant-router-v1") for group in groups])
        for fold in range(args.folds):
            train = np.flatnonzero(folds != fold)
            test = np.flatnonzero(folds == fold)
            weights, mean, scale = fit_scaled(matrix[train], target[train], args.regularization)
            probabilities[test] = score_scaled(matrix[test], weights, mean, scale)
        weights, mean, scale = fit_scaled(matrix, target, args.regularization)
        model = {
            "schema": 1,
            "candidate_labels": labels,
            "feature_names": names,
            "use_rotation": use_rotation,
            "regularization": args.regularization,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "weights": weights.tolist(),
        }
        Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.model_output).write_text(json.dumps(model, indent=2) + "\n")
        evaluation = "grouped_oof"
    else:
        model = json.loads(Path(args.model_input).read_text())
        if model["candidate_labels"] != labels or model["feature_names"] != names:
            raise RuntimeError("Frozen router model does not match candidate/features")
        probabilities = score_scaled(
            matrix,
            np.asarray(model["weights"]),
            np.asarray(model["mean"]),
            np.asarray(model["scale"]),
        )
        evaluation = "frozen_apply"

    scored: dict[int, dict[str, float]] = defaultdict(dict)
    for probability, (row_index, candidate) in zip(probabilities, owners):
        scored[row_index][candidate] = float(probability)

    output_rows = []
    metrics = Counter()
    by_operator: dict[str, Counter] = defaultdict(Counter)
    for row_index, row in enumerate(rows):
        data = row_data[row_index]
        truth = normalize_answer(row.get("mcq_answer") or row.get("answer"))
        primary = data["primary"]
        selected = primary
        candidate_scores = scored.get(row_index, {})
        if candidate_scores:
            alternative = max(candidate_scores, key=candidate_scores.get)
            if (
                alternative != primary
                and candidate_scores[alternative] >= args.min_switch_probability
                and candidate_scores[alternative] - candidate_scores.get(primary, 0.0) >= args.min_switch_delta
            ):
                selected = alternative
        primary_correct = primary == truth
        selected_correct = selected == truth
        changed = selected != primary
        metrics["primary_correct"] += int(primary_correct)
        metrics["router_correct"] += int(selected_correct)
        metrics["disagreements"] += int(bool(candidate_scores))
        metrics["interventions"] += int(changed)
        metrics["fixes"] += int(changed and not primary_correct and selected_correct)
        metrics["regressions"] += int(changed and primary_correct and not selected_correct)
        metrics["both_wrong_changes"] += int(changed and not primary_correct and not selected_correct)
        if candidate_scores:
            metrics["disagreement_primary_correct"] += int(primary_correct)
            metrics["disagreement_router_correct"] += int(selected_correct)
            metrics["disagreement_oracle_correct"] += int(truth in data["proposed"])
        operator = compile_temporal_program_v2(row.get("question", "")).operator
        by_operator[operator]["rows"] += 1
        by_operator[operator]["primary_correct"] += int(primary_correct)
        by_operator[operator]["router_correct"] += int(selected_correct)
        prediction = set_answer(primary_rows[data["key"]], selected)
        prediction.update(
            {
                "label_invariant_router_evaluation": evaluation,
                "label_invariant_router_primary": primary,
                "label_invariant_router_scores": candidate_scores,
                "label_invariant_router_intervened": changed,
            }
        )
        output_rows.append(prediction)

    total = len(rows)
    summary = dict(metrics)
    summary.update(
        {
            "rows": total,
            "primary_accuracy": metrics["primary_correct"] / total,
            "router_accuracy": metrics["router_correct"] / total,
            "candidate_labels": labels,
            "evaluation": evaluation,
            "uses_answer_letter_features": False,
            "uses_rotation_features": use_rotation,
            "min_switch_delta": args.min_switch_delta,
            "min_switch_probability": args.min_switch_probability,
            "by_operator": {key: dict(value) for key, value in sorted(by_operator.items())},
        }
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    Path(args.summary_output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
