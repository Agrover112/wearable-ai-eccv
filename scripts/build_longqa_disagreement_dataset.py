#!/usr/bin/env python3
"""Build a feature table for pivot/uniform disagreements and run grouped CV."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = ROOT / "baselines/longqa"
sys.path.insert(0, str(STARTER_KIT))

from longqa_utils import (  # noqa: E402
    classify_question_types,
    index_row_aligned_metadata,
    load_jsonl,
    normalize_answer,
    parse_mcq_options,
    sample_key,
)
from run_generate_longqa_proofpack import compile_temporal_program  # noqa: E402


def _by_key(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    return {sample_key(row): row for row in load_jsonl(path)}


def _answer(row: dict[str, Any]) -> str:
    return normalize_answer(row.get("mcq_answer_parsed") or row.get("mcq_answer", ""))


def _selected_stats(pack: dict[str, Any]) -> dict[str, float]:
    selected = pack.get("selected", [])
    scores = [float(item["score"]) for item in selected if item.get("score") is not None]
    times = sorted(float(item.get("timestamp", 0.0)) for item in selected)
    gaps = [right - left for left, right in zip(times, times[1:])]
    sources = Counter(str(item.get("source", "unknown")) for item in selected)
    stats = {
        "pack_score_mean": float(np.mean(scores)) if scores else 0.0,
        "pack_score_std": float(np.std(scores)) if scores else 0.0,
        "pack_score_max": max(scores) if scores else 0.0,
        "pack_time_span": times[-1] - times[0] if len(times) > 1 else 0.0,
        "pack_max_gap": max(gaps) if gaps else 0.0,
    }
    for source, count in sources.items():
        stats[f"pack_source={source}"] = float(count) / max(len(selected), 1)
    return stats


def _feature_dict(
    annotation: dict[str, Any],
    primary_answer: str,
    secondary_answer: str,
    verifier_row: dict[str, Any] | None,
    likelihood_row: dict[str, Any] | None,
    pack: dict[str, Any],
) -> dict[str, float]:
    question = str(annotation.get("question", ""))
    options = parse_mcq_options(annotation.get("mcq_options", ""))
    operator = compile_temporal_program(question).operator
    features: dict[str, float] = {
        "bias": 1.0,
        "question_chars": len(question) / 200.0,
        "question_words": len(question.split()) / 40.0,
        "candidate_same_length": float(
            len(options.get(primary_answer, "")) == len(options.get(secondary_answer, ""))
        ),
        "primary_option_words": len(options.get(primary_answer, "").split()) / 20.0,
        "secondary_option_words": len(options.get(secondary_answer, "").split()) / 20.0,
        f"operator={operator}": 1.0,
        f"primary_letter={primary_answer}": 1.0,
        f"secondary_letter={secondary_answer}": 1.0,
    }
    for tag in classify_question_types(question):
        features[f"question_type={tag}"] = 1.0
    features.update(_selected_stats(pack))

    if verifier_row:
        verifier_answer = _answer(verifier_row)
        features["verifier_present"] = 1.0
        features["verifier_selects_primary"] = float(verifier_answer == primary_answer)
        features["verifier_selects_secondary"] = float(verifier_answer == secondary_answer)
        features["verifier_selects_third"] = float(
            verifier_answer not in {primary_answer, secondary_answer}
        )
    if likelihood_row:
        visual = likelihood_row.get("visual_option_logprobs", {})
        blind = likelihood_row.get("blind_option_logprobs", {})
        if primary_answer in visual and secondary_answer in visual:
            features["likelihood_present"] = 1.0
            features["visual_candidate_delta"] = float(
                visual[secondary_answer] - visual[primary_answer]
            )
            features["visual_candidate_margin"] = abs(features["visual_candidate_delta"])
            if primary_answer in blind and secondary_answer in blind:
                features["blind_candidate_delta"] = float(
                    blind[secondary_answer] - blind[primary_answer]
                )
                features["locked_candidate_delta"] = (
                    features["visual_candidate_delta"]
                    + 0.1 * features["blind_candidate_delta"]
                )
    return features


def _vectorize(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    names = sorted({name for row in rows for name in row["features"]})
    index = {name: idx for idx, name in enumerate(names)}
    matrix = np.zeros((len(rows), len(names)), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for name, value in row["features"].items():
            matrix[row_idx, index[name]] = float(value)
    return matrix, names


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for step in range(1200):
        logits = np.clip(x @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = x.T @ (probabilities - y) / max(len(y), 1)
        penalty = l2 * weights / max(len(y), 1)
        penalty[0] = 0.0
        learning_rate = 0.25 / math.sqrt(1.0 + step / 100.0)
        weights -= learning_rate * (gradient + penalty)
    return weights


def _standardize(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    train_scaled = (train - mean) / std
    test_scaled = (test - mean) / std
    train_scaled[:, 0] = train[:, 0]
    test_scaled[:, 0] = test[:, 0]
    return train_scaled, test_scaled


def _fold(key: str, folds: int, salt: str = "outer") -> int:
    digest = hashlib.sha1(f"{salt}:{key}".encode()).hexdigest()
    return int(digest, 16) % folds


def _choose_l2(x: np.ndarray, y: np.ndarray, keys: list[str]) -> float:
    candidates = (0.01, 0.1, 1.0, 10.0)
    best = candidates[0]
    best_correct = -1
    for l2 in candidates:
        correct = 0
        for fold in range(4):
            train_idx = [idx for idx, key in enumerate(keys) if _fold(key, 4, "inner") != fold]
            test_idx = [idx for idx, key in enumerate(keys) if _fold(key, 4, "inner") == fold]
            if not train_idx or not test_idx:
                continue
            train_x, test_x = _standardize(x[train_idx], x[test_idx])
            weights = _fit_logistic(train_x, y[train_idx], l2)
            predictions = (test_x @ weights > 0.0).astype(np.float64)
            correct += int(np.sum(predictions == y[test_idx]))
        if correct > best_correct:
            best_correct = correct
            best = l2
    return best


def grouped_nested_cv(
    rows: list[dict[str, Any]],
    excluded_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    resolvable = []
    for row in rows:
        if row["target"] not in {"primary", "secondary"}:
            continue
        filtered = dict(row)
        filtered["features"] = {
            name: value
            for name, value in row["features"].items()
            if not name.startswith(excluded_prefixes)
        }
        resolvable.append(filtered)
    x, feature_names = _vectorize(resolvable)
    y = np.array([float(row["target"] == "secondary") for row in resolvable])
    keys = [row["video_path"] for row in resolvable]
    predictions: dict[str, str] = {}
    selected_l2: list[float] = []
    for fold in range(5):
        train_idx = [idx for idx, key in enumerate(keys) if _fold(key, 5) != fold]
        test_idx = [idx for idx, key in enumerate(keys) if _fold(key, 5) == fold]
        if not train_idx or not test_idx:
            continue
        l2 = _choose_l2(x[train_idx], y[train_idx], [keys[idx] for idx in train_idx])
        selected_l2.append(l2)
        train_x, test_x = _standardize(x[train_idx], x[test_idx])
        weights = _fit_logistic(train_x, y[train_idx], l2)
        fold_predictions = test_x @ weights > 0.0
        for idx, prediction in zip(test_idx, fold_predictions):
            predictions[resolvable[idx]["sample_key"]] = (
                "secondary" if prediction else "primary"
            )
    correct_resolvable = sum(
        predictions.get(row["sample_key"]) == row["target"] for row in resolvable
    )
    return {
        "folds": 5,
        "resolvable_rows": len(resolvable),
        "all_disagreements": len(rows),
        "correct_resolvable": correct_resolvable,
        "accuracy_resolvable": correct_resolvable / max(len(resolvable), 1),
        "accuracy_all_disagreements": correct_resolvable / max(len(rows), 1),
        "selected_l2": selected_l2,
        "feature_count": len(feature_names),
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--secondary", required=True)
    parser.add_argument("--primary-proofpack", required=True)
    parser.add_argument("--verifier", default=None)
    parser.add_argument("--likelihood", default=None)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()

    annotations = _by_key(args.annotations)
    primary_rows = load_jsonl(args.primary)
    primary = {sample_key(row): row for row in primary_rows}
    secondary = _by_key(args.secondary)
    verifier = _by_key(args.verifier)
    likelihood = _by_key(args.likelihood)
    proofpack = index_row_aligned_metadata(
        load_jsonl(args.primary_proofpack), primary_rows, "primary proof pack"
    )

    rows: list[dict[str, Any]] = []
    for key, primary_row in primary.items():
        if key not in annotations or key not in secondary:
            continue
        primary_answer = _answer(primary_row)
        secondary_answer = _answer(secondary[key])
        if primary_answer == secondary_answer:
            continue
        gold_answer = normalize_answer(annotations[key].get("mcq_answer", ""))
        target = (
            "primary"
            if primary_answer == gold_answer
            else "secondary"
            if secondary_answer == gold_answer
            else "neither"
        )
        row = {
            "sample_key": key,
            "video_path": annotations[key].get("video_path", ""),
            "question": annotations[key].get("question", ""),
            "mcq_options": annotations[key].get("mcq_options", ""),
            "gold_answer": gold_answer,
            "primary_answer": primary_answer,
            "secondary_answer": secondary_answer,
            "target": target,
            "operator": compile_temporal_program(annotations[key]["question"]).operator,
            "question_types": classify_question_types(annotations[key]["question"]),
            "verifier_answer": _answer(verifier[key]) if key in verifier else "",
            "features": _feature_dict(
                annotations[key],
                primary_answer,
                secondary_answer,
                verifier.get(key),
                likelihood.get(key),
                proofpack[key],
            ),
        }
        rows.append(row)

    cv_ablations = {
        "metadata_only": grouped_nested_cv(
            rows,
            (
                "verifier_",
                "likelihood_",
                "visual_",
                "blind_",
                "locked_",
            ),
        ),
        "metadata_plus_verifier": grouped_nested_cv(
            rows,
            ("likelihood_", "visual_", "blind_", "locked_"),
        ),
        "metadata_plus_likelihood": grouped_nested_cv(rows, ("verifier_",)),
        "all_features": grouped_nested_cv(rows),
    }
    cv = cv_ablations["all_features"]
    target_counts = Counter(row["target"] for row in rows)
    baseline = {
        "primary": target_counts["primary"],
        "secondary": target_counts["secondary"],
        "candidate_oracle": target_counts["primary"] + target_counts["secondary"],
    }
    if verifier:
        baseline["verifier_candidate_restricted"] = sum(
            (
                "primary"
                if row["verifier_answer"] == row["primary_answer"]
                else "secondary"
                if row["verifier_answer"] == row["secondary_answer"]
                else "primary"
            )
            == row["target"]
            for row in rows
        )
    summary = {
        "rows": len(rows),
        "target_counts": dict(target_counts),
        "baselines_correct_on_disagreements": baseline,
        "feature_coverage": {
            "verifier": sum(row["features"].get("verifier_present", 0.0) > 0 for row in rows),
            "likelihood": sum(
                row["features"].get("likelihood_present", 0.0) > 0 for row in rows
            ),
        },
        "grouped_nested_cv": cv,
        "grouped_nested_cv_ablations": cv_ablations,
    }

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as handle:
        for row in rows:
            row["cv_prediction"] = cv["predictions"].get(row["sample_key"])
            row["cv_predictions"] = {
                name: result["predictions"].get(row["sample_key"])
                for name, result in cv_ablations.items()
            }
            handle.write(json.dumps(row) + "\n")
    summary["grouped_nested_cv"].pop("predictions", None)
    for result in summary["grouped_nested_cv_ablations"].values():
        result.pop("predictions", None)
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Feature rows: {output_jsonl}")


if __name__ == "__main__":
    main()
