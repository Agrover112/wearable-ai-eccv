"""Text-only organizer that resolves specialist evidence."""

from __future__ import annotations

import json
from typing import Any

from agents.protocol import ORGANIZER_SCHEMA, validate_organizer_report

PROMPT_VERSION = "organizer_v3"


def build_organizer_prompt(
    row: dict[str, Any],
    options: dict[str, str],
    hypotheses: dict[str, Any],
    temporal_report: dict[str, Any],
    visual_report: dict[str, Any],
) -> str:
    option_text = "\n".join(f"{letter}. {options[letter]}" for letter in "ABCD")
    packet = {
        "hypotheses": hypotheses,
        "temporal_specialist": temporal_report,
        "visual_specialist": visual_report,
    }
    return f"""You are the organizer in a video-QA system.

Choose one option using only the structured reports below. Assess all four
options. Treat cited, direct observations as stronger than confidence numbers.
The temporal and visual reports are correlated role-conditioned views from the
same model and images, not independent votes. Agreement is corroboration but
must not be double-counted. A contradiction must describe evidence incompatible
with the full option; UNKNOWN is not a contradiction. For multi-part options,
require support for all decisive parts. Use the question's temporal operator
exactly. Do not use answer-letter frequency, option length, or world-knowledge
plausibility as evidence. Keep each option reason to one short sentence and the
final rationale under 50 words.

Question:
{row["question"]}

Options:
{option_text}

Evidence packet:
{json.dumps(packet, sort_keys=True)}

Return only the requested JSON object."""


def run_organizer_agent(
    model: Any,
    row: dict[str, Any],
    options: dict[str, str],
    hypotheses: dict[str, Any],
    temporal_report: dict[str, Any],
    visual_report: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    report = model.generate_json(
        [],
        [
            {
                "role": "user",
                "content": build_organizer_prompt(
                    row,
                    options,
                    hypotheses,
                    temporal_report,
                    visual_report,
                ),
            }
        ],
        ORGANIZER_SCHEMA,
        "egolongqa_organizer",
        max_new_tokens,
        retry_max_new_tokens=max_new_tokens * 2,
    )
    return validate_organizer_report(report)
