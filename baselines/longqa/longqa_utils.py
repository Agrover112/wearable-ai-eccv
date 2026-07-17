#!/usr/bin/env python3
"""Shared helpers for EgoLongQA generation and diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Any

PROMPT_VARIANTS = (
    "baseline",
    "evidence_first",
    "option_verify",
    "temporal_anchor",
    "timestamp_grounded",
    "anti_shortcut",
    "combined",
)

RETRIEVAL_QUERY_MODES = (
    "question",
    "question_options",
    "per_option_union",
    "question_temporal_boost",
)

TEMPORAL_CUES = {
    "after",
    "before",
    "first",
    "then",
    "earlier",
    "later",
    "during",
    "while",
    "when",
    "next",
    "previous",
    "prior",
    "following",
    "last",
    "start",
    "end",
    "beginning",
    "finally",
    "eventually",
}

QUESTION_TYPE_PATTERNS = {
    "ocr_named_detail": (
        r"\b(sign|written|word|name|named|label|title|text|plaque|price|cost|"
        r"brand|store|station|artist|album|book|number|tag)\b",
    ),
    "count_quantity": (r"\bhow many\b", r"\bnumber of\b", r"\bcount\b", r"\bhow much\b"),
    "color_appearance": (
        r"\b(colou?r|pattern|shape|decorat|appearance|look|wear|shirt|surface|"
        r"material|texture)\b",
    ),
    "spatial_location": (
        r"\bwhere\b",
        r"\b(left|right|near|next to|behind|front|inside|outside|location|place|area)\b",
    ),
    "fine_object_identity": (
        r"\b(which|what) (item|object|tool|ingredient|product|device|vehicle|"
        r"animal|food|container|building|artwork|species|type)\b",
    ),
    "cross_time_ordering": (
        r"\b(after|before|earlier|later|first|last|then|again|order|between|"
        r"during|previously|eventually|end of)\b",
    ),
}

QUESTION_TYPE_PRIORITY = tuple(QUESTION_TYPE_PATTERNS)

BASELINE_PROMPT_TEMPLATE = (
    "Watch the video and answer the following multiple-choice question.\n\n"
    "Question: {question}\n\n"
    "Options:\n{mcq_options}\n\n"
    "Answer with ONLY the single letter of the correct option (A, B, C, or D). "
    "Do not include any other text."
)

_VARIANT_INSTRUCTIONS = {
    "baseline": "",
    "evidence_first": (
        "You are answering a multiple-choice question about a long first-person "
        "video. Internally identify the relevant time period(s) and visual "
        "evidence before choosing. Pay attention to handled/viewed objects, "
        "attributes such as color/text/shape/quantity/location, and actions "
        "before or after the event mentioned in the question. Return only the "
        "final option letter."
    ),
    "option_verify": (
        "You are answering a multiple-choice question about a long first-person "
        "video. Internally compare each option A, B, C, and D against the "
        "visual evidence. Check object identity, attributes, quantities, "
        "spatial relationships, and actions. Reject options contradicted by "
        "the video. Return only the final option letter."
    ),
    "temporal_anchor": (
        "The question may depend on temporal order. Internally locate the "
        "anchor event mentioned in the question, then inspect what happened "
        "immediately before or after it as required by the wording. Prefer "
        "evidence from the relevant temporal neighborhood over global "
        "impressions. Return only the final option letter."
    ),
    "timestamp_grounded": (
        "You are analyzing a long first-person video whose observations are "
        "shown in chronological order with timestamp labels. Locate all "
        "occurrences relevant to the question and cite the decisive timestamps. "
        "For first/last questions, inspect the full timeline; for before/after "
        "questions, identify the anchor event before examining the requested "
        "direction. Compare all four options against the timestamped visual "
        "evidence, including fine details such as color, text, object identity, "
        "and state changes. Give a concise evidence summary, then end with exactly "
        "`Final answer: X`, where X is A, B, C, or D."
    ),
    "anti_shortcut": (
        "Use visual evidence from the video, not answer priors. Do not choose "
        "based on option letter frequency, option length, or which option "
        "sounds most typical. Return only the final option letter."
    ),
    "combined": (
        "You are answering a multiple-choice question about a long first-person "
        "video. Use video evidence, not option priors. Do not choose based on "
        "option letter frequency, option length, or which option sounds most "
        "typical. Internally locate the relevant time period(s). If the "
        "question contains words like after, before, first, then, earlier, "
        "later, or during, identify the anchor event and reason from the "
        "correct temporal neighborhood. Evaluate each option against visible "
        "evidence, including object identity, color, text, shape, quantity, "
        "location, spatial relationships, and actions. Return only the final "
        "option letter."
    ),
}


def build_longqa_prompt(
    question: object,
    mcq_options: object,
    prompt_variant: str = "baseline",
) -> str:
    """Build a LongQA prompt while preserving baseline text by default."""
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {prompt_variant}")
    instruction = _VARIANT_INSTRUCTIONS[prompt_variant]
    if prompt_variant == "timestamp_grounded":
        return (
            f"{instruction}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{mcq_options}"
        )
    base = BASELINE_PROMPT_TEMPLATE.format(
        question=question,
        mcq_options=mcq_options,
    )
    if not instruction:
        return base
    return f"{instruction}\n\n{base}"


def normalize_answer(raw: object) -> str:
    """Extract A/B/C/D from a raw model output."""
    text = str(raw).strip()
    if not text:
        return ""
    upper = text.upper()
    if len(upper) == 1 and upper in "ABCD":
        return upper

    # Prefer final/answer-like declarations, especially the last such mention.
    final_matches = list(
        re.finditer(
            r"\b(?:final\s+answer|answer|option|choice)\s*[:.]?\s*\(?([A-Da-d])\)?\b",
            text,
            re.IGNORECASE,
        )
    )
    if final_matches:
        return final_matches[-1].group(1).upper()

    leading = re.match(r"^\s*\(?([A-Da-d])\)?\s*[\.\:\)]?\s*$", text)
    if leading:
        return leading.group(1).upper()

    # Lowercase a-d are common words in explanations; only accept them when
    # an answer declaration or a single-letter response matched above.
    standalone = list(re.finditer(r"\b([A-D])\b", text))
    if standalone:
        return standalone[-1].group(1).upper()

    return upper[0] if upper[:1] in "ABCD" else ""


def parse_mcq_options(mcq_options: object) -> dict[str, str]:
    """Parse MCQ option text into a letter->option mapping."""
    text = str(mcq_options)
    matches = list(re.finditer(r"(?ms)(?:^|\s)([A-D])\.\s*", text))
    if not matches:
        return {}
    parsed: dict[str, str] = {}
    for i, match in enumerate(matches):
        letter = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parsed[letter] = text[start:end].strip()
    return parsed


def has_temporal_cue(question: object) -> bool:
    tokens = re.findall(r"[A-Za-z]+", str(question).lower())
    return any(token in TEMPORAL_CUES for token in tokens)


def classify_question_types(question: object) -> list[str]:
    """Return deterministic, overlapping question-type tags."""
    text = str(question)
    return [
        name
        for name, patterns in QUESTION_TYPE_PATTERNS.items()
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    ]


def primary_question_type(question: object) -> str:
    """Return the first matching tag for mutually exclusive reporting."""
    tags = set(classify_question_types(question))
    return next((name for name in QUESTION_TYPE_PRIORITY if name in tags), "other")


def sample_key(row: dict[str, Any]) -> str:
    """Stable key for matching annotations/predictions/subsets."""
    return str(row.get("id") or f"{row.get('video_path', '')}||{row.get('question', '')}")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_subset_keys(path: str | None) -> set[str] | None:
    if not path:
        return None
    with open(path, "r") as f:
        data = json.load(f)
    raw_items = data.get("samples", data) if isinstance(data, dict) else data
    keys: set[str] = set()
    for item in raw_items:
        if isinstance(item, dict):
            keys.add(str(item.get("key") or item.get("sample_key") or sample_key(item)))
        else:
            keys.add(str(item))
    return keys


def apply_subset(rows: list[dict[str, Any]], subset_file: str | None) -> list[dict[str, Any]]:
    keys = load_subset_keys(subset_file)
    if keys is None:
        return rows
    return [row for row in rows if sample_key(row) in keys]


def query_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def build_prediction_row(
    row: dict[str, Any],
    raw_response: object,
    prompt_variant: str = "baseline",
) -> dict[str, Any]:
    pred = dict(row)
    raw = str(raw_response)
    parsed = normalize_answer(raw)
    # Preserve backward compatibility: existing evaluator reads mcq_answer.
    pred["mcq_answer"] = raw
    pred["mcq_answer_raw"] = raw
    pred["mcq_answer_parsed"] = parsed
    pred["prompt_variant"] = prompt_variant
    return pred


def compute_diagnostics(
    golden: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    run_id: str | None = None,
) -> dict[str, Any]:
    n = min(len(golden), len(preds))
    golden = golden[:n]
    preds = preds[:n]
    correct = 0
    letter_stats: dict[str, dict[str, int]] = {
        letter: {"correct": 0, "total": 0} for letter in "ABCD"
    }
    category_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    pred_dist = {letter: 0 for letter in "ABCD"}
    gold_dist = {letter: 0 for letter in "ABCD"}
    temporal_stats = {"correct": 0, "total": 0}
    non_temporal_stats = {"correct": 0, "total": 0}
    non_c = {"correct": 0, "total": 0}
    c_gold = {"correct": 0, "total": 0}
    non_shortest_gold = {"correct": 0, "total": 0}
    always_c_correct = 0
    shortest_correct = 0
    per_row = []

    for idx, (gold, pred) in enumerate(zip(golden, preds)):
        gold_answer = normalize_answer(gold.get("mcq_answer", ""))
        pred_answer = normalize_answer(
            pred.get("mcq_answer_parsed") or pred.get("mcq_answer", "")
        )
        ok = pred_answer == gold_answer
        correct += int(ok)
        if gold_answer in gold_dist:
            gold_dist[gold_answer] += 1
            letter_stats[gold_answer]["total"] += 1
            letter_stats[gold_answer]["correct"] += int(ok)
        if pred_answer in pred_dist:
            pred_dist[pred_answer] += 1
        if gold_answer == "C":
            always_c_correct += 1
            c_gold["total"] += 1
            c_gold["correct"] += int(ok)
        else:
            non_c["total"] += 1
            non_c["correct"] += int(ok)

        options = parse_mcq_options(gold.get("mcq_options", ""))
        shortest_letter = ""
        if options:
            shortest_letter = min(options, key=lambda letter: len(options[letter]))
            shortest_correct += int(shortest_letter == gold_answer)
            if shortest_letter != gold_answer:
                non_shortest_gold["total"] += 1
                non_shortest_gold["correct"] += int(ok)

        temporal = has_temporal_cue(gold.get("question", ""))
        target = temporal_stats if temporal else non_temporal_stats
        target["total"] += 1
        target["correct"] += int(ok)

        category = str(gold.get("category", ""))
        category_stats[category]["total"] += 1
        category_stats[category]["correct"] += int(ok)
        per_row.append(
            {
                "index": idx,
                "video_path": gold.get("video_path", ""),
                "question": gold.get("question", ""),
                "category": category,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "correct": ok,
                "temporal": temporal,
            }
        )

    def acc(stats: dict[str, int]) -> float:
        return round(stats["correct"] / stats["total"], 4) if stats["total"] else 0.0

    letter_accuracy = {
        letter: {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": acc(stats),
        }
        for letter, stats in letter_stats.items()
    }
    macro_letter_values = [
        item["accuracy"] for item in letter_accuracy.values() if item["total"]
    ]
    category_accuracy = {
        cat: round(stats["correct"] / stats["total"], 4)
        for cat, stats in sorted(category_stats.items())
    }
    category_count = {cat: stats["total"] for cat, stats in sorted(category_stats.items())}
    accuracy = round(correct / n, 4) if n else 0.0
    prompt_variants = sorted(
        {str(pred.get("prompt_variant", "")) for pred in preds if pred.get("prompt_variant")}
    )
    return {
        "run_id": run_id,
        "prompt_variant": prompt_variants[0] if len(prompt_variants) == 1 else prompt_variants,
        "accuracy_raw": accuracy,
        "accuracy": accuracy,
        "correct": correct,
        "total": n,
        "always_c_accuracy": round(always_c_correct / n, 4) if n else 0.0,
        "margin_over_always_c": round(accuracy - (always_c_correct / n), 4) if n else 0.0,
        "always_shortest_accuracy": round(shortest_correct / n, 4) if n else 0.0,
        "macro_letter_accuracy": round(sum(macro_letter_values) / len(macro_letter_values), 4)
        if macro_letter_values
        else 0.0,
        "per_letter_accuracy": letter_accuracy,
        "answer_distribution_predicted": pred_dist,
        "answer_distribution_gold": gold_dist,
        "non_c_accuracy": acc(non_c),
        "c_gold_accuracy": acc(c_gold),
        "non_shortest_gold_accuracy": acc(non_shortest_gold),
        "temporal_question_accuracy": acc(temporal_stats),
        "non_temporal_question_accuracy": acc(non_temporal_stats),
        "category_accuracy": category_accuracy,
        "category_count": category_count,
        "per_row": per_row,
    }
