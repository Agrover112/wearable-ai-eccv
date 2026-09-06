#!/usr/bin/env python3
"""Score complete LongQA options or explicit option clauses on identical frames."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from longqa_utils import (
    apply_subset,
    build_prediction_row,
    compute_diagnostics,
    load_jsonl,
    normalize_answer,
    parse_mcq_options,
    sample_key,
)
from run_generate_longqa_grounded import extract_frames_by_indices
from run_generate_longqa_uncertainty import _index_jsonl, _video_metadata


SUPPORT_MODES = ("whole", "clauses")
POLICIES = ("unrestricted", "conservative", "strict")
SCHEMA_VERSION = 1


def resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def answer(record: dict[str, Any]) -> str:
    return normalize_answer(record.get("mcq_answer_parsed") or record.get("mcq_answer"))


def endpoint_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0 or count <= 0:
        return []
    count = min(total_frames, count)
    if count == 1:
        return [0]
    return sorted(
        {
            round(index * (total_frames - 1) / (count - 1))
            for index in range(count)
        }
    )


def logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def split_claims(text: str) -> list[str]:
    normalized = " ".join(text.split()).strip(" .")
    parts = [part.strip(" ,.") for part in re.split(r"\s*;\s*", normalized) if part.strip()]
    if len(parts) == 1 and re.search(r"\bbefore\b.+\bafter\b", normalized, re.I):
        parts = [
            part.strip(" ,.")
            for part in re.split(r"(?=\bafter\b)", normalized, maxsplit=1, flags=re.I)
            if part.strip(" ,.")
        ]
    if len(parts) == 1 and re.search(r",\s+(?:and|while|whereas)\s+", normalized, re.I):
        parts = [
            part.strip(" ,.")
            for part in re.split(r",\s+(?:and|while|whereas)\s+", normalized, maxsplit=1, flags=re.I)
            if part.strip(" ,.")
        ]
    return parts or [normalized]


def is_compound(row: dict[str, Any], options: dict[str, str]) -> bool:
    question = str(row.get("question", ""))
    if re.search(r"\b(first.+last|before.+after|earlier.+later|and what|and where|and how)\b", question, re.I):
        return True
    return any(len(split_claims(text)) > 1 for text in options.values())


def support_prompt(
    row: dict[str, Any], candidate: str, claims: list[str], timestamps: list[float], mode: str
) -> str:
    timestamp_index = ", ".join(
        f"image {index}={timestamp:.1f}s"
        for index, timestamp in enumerate(timestamps, start=1)
    )
    if mode == "clauses":
        claim_text = "\n".join(
            f"{index}. {claim}" for index, claim in enumerate(claims, start=1)
        )
        evaluation = (
            "Evaluate every listed claim separately. The candidate is SUPPORTED only "
            "if every decisive claim is visibly supported by the correct occurrence.\n\n"
            f"Required claims:\n{claim_text}"
        )
    else:
        evaluation = (
            "Evaluate the complete candidate as one answer. It is SUPPORTED only if "
            "every decisive part is visibly supported by the correct occurrence."
        )
    return f"""Evaluate one candidate answer to a question about a long first-person video.

The images are identical for every candidate and are chronological. A visually similar event at the wrong time does not support the candidate. Repeated nearby images describe one occurrence and must not be counted as independent evidence. Missing evidence is not contradiction.

Question: {row['question']}
Candidate answer: {candidate}

{evaluation}

Image timestamps from the start of the original video:
{timestamp_index}

Choose exactly one verdict:
A. SUPPORTED
B. CONTRADICTED
C. INSUFFICIENT

Answer with only A, B, or C."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl")
    parser.add_argument("--video-folder", default="../egolongqa/val")
    parser.add_argument("--subset-file", required=True)
    parser.add_argument("--candidate-predictions", action="append", required=True)
    parser.add_argument("--candidate-labels", nargs="+", required=True)
    parser.add_argument("--fallback-predictions", required=True)
    parser.add_argument("--support-mode", choices=SUPPORT_MODES, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if len(args.candidate_predictions) != len(args.candidate_labels):
        parser.error("candidate prediction and label counts must match")
    return args


def run_fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "schema": SCHEMA_VERSION,
        "candidates": [os.path.abspath(path) for path in args.candidate_predictions],
        "labels": args.candidate_labels,
        "fallback": os.path.abspath(args.fallback_predictions),
        "support_mode": args.support_mode,
        "model": args.llm_model,
        "max_frames": args.max_frames,
        "subset": os.path.abspath(args.subset_file),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def choose_policy(
    fallback: str, scores: dict[str, dict[str, Any]], policy: str
) -> str:
    valid = {
        letter: float(record["support_margin"])
        for letter, record in scores.items()
        if record.get("support_margin") is not None
    }
    if not valid:
        return fallback
    best = max(valid, key=lambda letter: (valid[letter], letter == fallback))
    if policy == "unrestricted" or best == fallback:
        return best
    current = valid.get(fallback, -math.inf)
    if policy == "conservative":
        if valid[best] >= -0.25 and current <= 0.0 and valid[best] - current >= 1.0:
            return best
    elif valid[best] >= 0.0 and current <= -0.5 and valid[best] - current >= 2.0:
        return best
    return fallback


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    from model import VLLMModel, reset_prompt_token_stats, summarize_prompt_token_stats
    from run_generate_longqa import _print_context_summary

    args = parse_args()
    input_path = resolve(args.input)
    video_folder = resolve(args.video_folder)
    rows = apply_subset(load_jsonl(input_path), args.subset_file)
    candidates = {
        label: _index_jsonl(resolve(path))
        for label, path in zip(args.candidate_labels, args.candidate_predictions)
    }
    fallback = _index_jsonl(resolve(args.fallback_predictions))
    required = {sample_key(row) for row in rows}
    for label, indexed in {**candidates, "fallback": fallback}.items():
        missing = required - set(indexed)
        if missing:
            raise RuntimeError(f"{label} is missing {len(missing)} clause-support rows")

    output_dir = Path(resolve(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "features.jsonl"
    fingerprint = run_fingerprint(args)
    existing = [] if args.no_resume or not features_path.exists() else load_jsonl(str(features_path))
    start = 0
    for index, record in enumerate(existing[: len(rows)]):
        if (
            sample_key(record) != sample_key(rows[index])
            or record.get("clause_support_fingerprint") != fingerprint
        ):
            break
        start += 1
    write_jsonl(features_path, existing[:start])

    model = VLLMModel(
        args.llm_model,
        tp_size=1,
        concurrency=1,
        max_frames=args.max_frames,
        model_type="qwen",
    )
    reset_prompt_token_stats()
    calls = 0
    begun = time.time()
    with model, features_path.open("a") as handle:
        for index, row in enumerate(rows[start:], start=start):
            key = sample_key(row)
            options = parse_mcq_options(row["mcq_options"])
            fallback_letter = answer(fallback[key])
            branch_answers = {
                label: answer(indexed[key]) for label, indexed in candidates.items()
            }
            proposed = {letter for letter in branch_answers.values() if letter in options}
            if fallback_letter in options:
                proposed.add(fallback_letter)
            compound = is_compound(row, options)
            applied = len(proposed) > 1 or compound
            letters = sorted(options) if compound else sorted(proposed)
            scored: dict[str, dict[str, Any]] = {}
            if applied:
                video_path = os.path.join(video_folder, str(row["video_path"]))
                fps, total_frames = _video_metadata(video_path)
                indices = endpoint_indices(total_frames, args.max_frames)
                frames = extract_frames_by_indices(video_path, indices)
                timestamps = [value / max(fps, 1e-6) for value in indices]
                for letter in letters:
                    claims = split_claims(options[letter])
                    error = None
                    verdict_scores = None
                    margin = None
                    try:
                        verdict_scores = model.score_choice_letters(
                            frames,
                            [{"role": "user", "content": support_prompt(
                                row, options[letter], claims, timestamps, args.support_mode
                            )}],
                            letters=("A", "B", "C"),
                        )
                        margin = verdict_scores["A"] - logsumexp(
                            [verdict_scores["B"], verdict_scores["C"]]
                        )
                    except Exception as exception:
                        error = f"{type(exception).__name__}: {exception}"
                    scored[letter] = {
                        "answer_text": options[letter],
                        "claims": claims,
                        "support_margin": margin,
                        "verdict_logprobs": verdict_scores,
                        "error": error,
                    }
                    calls += 1
            record = {
                "sample_key": key,
                "video_path": row["video_path"],
                "question": row["question"],
                "clause_support_schema": SCHEMA_VERSION,
                "clause_support_fingerprint": fingerprint,
                "clause_support_mode": args.support_mode,
                "clause_support_applied": applied,
                "clause_support_compound": compound,
                "clause_support_fallback": fallback_letter,
                "clause_support_candidate_answers": branch_answers,
                "clause_support_scored": scored,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"  Clause support: {index + 1}/{len(rows)} calls={calls}")

    features = load_jsonl(str(features_path))
    outputs: dict[str, list[dict[str, Any]]] = {policy: [] for policy in POLICIES}
    for row, feature in zip(rows, features):
        fallback_letter = str(feature["clause_support_fallback"])
        scores = feature["clause_support_scored"]
        for policy in POLICIES:
            selected = choose_policy(fallback_letter, scores, policy)
            prediction = build_prediction_row(
                row, selected, prompt_variant=f"clause_support_{args.support_mode}_{policy}"
            )
            prediction.update(
                {
                    "clause_support_fingerprint": fingerprint,
                    "clause_support_mode": args.support_mode,
                    "clause_support_policy": policy,
                    "clause_support_applied": feature["clause_support_applied"],
                    "clause_support_fallback": fallback_letter,
                }
            )
            outputs[policy].append(prediction)
    summary = {"support_mode": args.support_mode, "calls": calls, "policies": {}}
    for policy, predictions in outputs.items():
        write_jsonl(output_dir / f"predictions_{policy}.jsonl", predictions)
        diagnostics = compute_diagnostics(
            rows, predictions, run_id=f"clause_support_{args.support_mode}_{policy}"
        )
        (output_dir / f"results_{policy}.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n"
        )
        summary["policies"][policy] = {
            "correct": diagnostics["correct"],
            "total": diagnostics["total"],
            "accuracy": diagnostics["accuracy_raw"],
            "changes": sum(
                answer(prediction) != str(feature["clause_support_fallback"])
                for prediction, feature in zip(predictions, features)
            ),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Runtime seconds: {time.time() - begun:.0f}")
    _print_context_summary(summarize_prompt_token_stats())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
