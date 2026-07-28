"""JSON contracts shared by the EgoLongQA agents."""

from __future__ import annotations

from typing import Any

OPTION_LETTERS = ("A", "B", "C", "D")
VERDICT_STATUSES = ("SUPPORTED", "CONTRADICTED", "UNKNOWN")

HYPOTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question_type": {
            "type": "string",
            "enum": [
                "TEMPORAL_ORDER",
                "STATE_CHANGE",
                "OBJECT_ACTION",
                "OCR_DETAIL",
                "COUNT",
                "GLOBAL_INTENT",
                "OTHER",
            ],
        },
        "hypotheses": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "option": {"type": "string", "enum": list(OPTION_LETTERS)},
                    "claim": {"type": "string"},
                    "anchor_event": {"type": "string"},
                    "target_event": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": [
                            "BEFORE",
                            "AFTER",
                            "FIRST",
                            "LAST",
                            "REVISIT",
                            "STATE_CHANGE",
                            "COUNT",
                            "IDENTITY",
                            "NONE",
                        ],
                    },
                    "visual_checks": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "option",
                    "claim",
                    "anchor_event",
                    "target_event",
                    "relation",
                    "visual_checks",
                ],
            },
        },
    },
    "required": ["question_type", "hypotheses"],
}

SPECIALIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "role": {"type": "string", "enum": ["temporal", "visual"]},
        "summary": {"type": "string"},
        "verdicts": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "option": {"type": "string", "enum": list(OPTION_LETTERS)},
                    "status": {
                        "type": "string",
                        "enum": list(VERDICT_STATUSES),
                    },
                    "confidence": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "evidence": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "frame_id": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 64,
                                },
                                "observation": {"type": "string"},
                            },
                            "required": ["frame_id", "observation"],
                        },
                    },
                    "missing_evidence": {"type": "string"},
                },
                "required": [
                    "option",
                    "status",
                    "confidence",
                    "evidence",
                    "missing_evidence",
                ],
            },
        },
    },
    "required": ["role", "summary", "verdicts"],
}

ORGANIZER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "option_assessments": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "option": {"type": "string", "enum": list(OPTION_LETTERS)},
                    "support_score": {
                        "type": "integer",
                        "minimum": -100,
                        "maximum": 100,
                    },
                    "reason": {"type": "string"},
                },
                "required": ["option", "support_score", "reason"],
            },
        },
        "selected_option": {"type": "string", "enum": list(OPTION_LETTERS)},
        "rationale": {"type": "string"},
    },
    "required": ["option_assessments", "selected_option", "rationale"],
}


def normalize_option_records(
    report: dict[str, Any],
    field: str,
) -> dict[str, Any]:
    """Require one record per option and store them in stable A-D order."""
    records = report[field]
    by_option = {str(record["option"]): record for record in records}
    if set(by_option) != set(OPTION_LETTERS) or len(records) != len(by_option):
        raise RuntimeError(f"{field} must contain each option A-D exactly once")
    report[field] = [by_option[letter] for letter in OPTION_LETTERS]
    return report


def validate_hypothesis_report(report: dict[str, Any]) -> dict[str, Any]:
    return normalize_option_records(report, "hypotheses")


def validate_specialist_report(
    report: dict[str, Any],
    role: str,
    frame_count: int,
) -> dict[str, Any]:
    if report["role"] != role:
        raise RuntimeError(f"Expected {role} report, received {report['role']}")
    normalize_option_records(report, "verdicts")
    for verdict in report["verdicts"]:
        for citation in verdict["evidence"]:
            if int(citation["frame_id"]) > frame_count:
                raise RuntimeError(
                    f"{role} cited frame {citation['frame_id']} but only "
                    f"{frame_count} frames were supplied"
                )
    return report


def validate_organizer_report(report: dict[str, Any]) -> dict[str, Any]:
    return normalize_option_records(report, "option_assessments")
