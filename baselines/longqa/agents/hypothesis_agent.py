"""Text-only option-to-hypothesis compiler."""

from __future__ import annotations

from typing import Any

from agents.protocol import HYPOTHESIS_SCHEMA, validate_hypothesis_report

PROMPT_VERSION = "hypothesis_v3"


def build_hypothesis_prompt(row: dict[str, Any], options: dict[str, str]) -> str:
    option_text = "\n".join(f"{letter}. {options[letter]}" for letter in "ABCD")
    return f"""You are the hypothesis compiler in a video-QA system.

Turn every answer option into a concrete claim that separate evidence roles can
test. Do not choose an answer, rank options, or use answer-letter priors.
For temporal questions, identify the anchor event, the event or detail to test,
and the required temporal relation. For non-temporal questions use relation
IDENTITY, COUNT, STATE_CHANGE, or NONE as appropriate. Each visual check should
name a directly observable fact. Keep every string to one short sentence, use
at most two visual checks per option, and do not restate the question or option.

Question:
{row["question"]}

Options:
{option_text}

Return only the requested JSON object."""


def run_hypothesis_agent(
    model: Any,
    row: dict[str, Any],
    options: dict[str, str],
    max_new_tokens: int,
) -> dict[str, Any]:
    report = model.generate_json(
        [],
        [{"role": "user", "content": build_hypothesis_prompt(row, options)}],
        HYPOTHESIS_SCHEMA,
        "egolongqa_hypotheses",
        max_new_tokens,
        retry_max_new_tokens=max_new_tokens * 2,
    )
    return validate_hypothesis_report(report)
