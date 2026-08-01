#!/usr/bin/env python3
"""Measure how much of each LongQA retrieval query reaches a text encoder."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import median

from transformers import AutoConfig, AutoProcessor


OPTION_PATTERN = re.compile(r"(?:^|\s)([A-D])\.\s*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--model", default="google/siglip2-so400m-patch14-384"
    )
    parser.add_argument("--output")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def retrieval_query(row: dict) -> str:
    return (
        "Find video evidence needed to answer this multiple-choice question.\n"
        f"Question: {row['question']}\n"
        f"Options:\n{row['mcq_options']}"
    )


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[round((len(values) - 1) * fraction)]


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    config = AutoConfig.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    tokenizer = processor.tokenizer
    text_config = getattr(config, "text_config", config)
    token_limit = int(text_config.max_position_embeddings)

    lengths: list[int] = []
    question_coverage = Counter()
    option_coverage = {letter: Counter() for letter in "ABCD"}
    records = []
    for row in load_jsonl(args.input):
        query = retrieval_query(row)

        def token_count(text: str) -> int:
            encoded = tokenizer(text, add_special_tokens=True)
            return len(encoded["input_ids"])

        query_tokens = token_count(query)
        lengths.append(query_tokens)
        question_end = query.index(row["question"]) + len(row["question"])
        question_status = (
            "full" if token_count(query[:question_end]) <= token_limit else "partial"
        )
        question_coverage[question_status] += 1

        options_text = row["mcq_options"]
        options_start = query.index(options_text)
        matches = list(OPTION_PATTERN.finditer(options_text))
        row_options = {}
        for index, match in enumerate(matches):
            letter = match.group(1)
            start = options_start + match.start()
            end = options_start + (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(options_text)
            )
            if token_count(query[:end]) <= token_limit:
                status = "full"
            elif token_count(query[:start]) < token_limit:
                status = "partial"
            else:
                status = "absent"
            option_coverage[letter][status] += 1
            row_options[letter] = status
        records.append(
            {
                "video_path": row.get("video_path"),
                "query_tokens": query_tokens,
                "token_limit": token_limit,
                "truncated": query_tokens > token_limit,
                "question": question_status,
                "options": row_options,
            }
        )

    total = len(records)
    truncated = sum(record["truncated"] for record in records)
    summary = {
        "model": args.model,
        "text_position_limit": token_limit,
        "rows": total,
        "truncated_rows": truncated,
        "truncated_fraction": round(truncated / total, 6),
        "query_tokens": {
            "min": min(lengths),
            "median": median(lengths),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "question_coverage": dict(question_coverage),
        "option_coverage": {
            letter: dict(counts) for letter, counts in option_coverage.items()
        },
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"summary": summary, "records": records}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
