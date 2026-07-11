import numpy as np

from run_generate_longqa_openqa import build_openqa_prompt, select_nearest_option
from longqa_utils import classify_question_types, primary_question_type


def test_openqa_prompt_does_not_include_options():
    prompt = build_openqa_prompt("What did I pick up?")
    assert "What did I pick up?" in prompt
    assert "Options:" not in prompt
    assert "A." not in prompt


def test_select_nearest_option_returns_scores_and_margin():
    answer = np.array([1.0, 0.0], dtype=np.float32)
    options = np.array(
        [[0.0, 1.0], [0.8, 0.2], [1.0, 0.0], [-1.0, 0.0]],
        dtype=np.float32,
    )

    letter, scores, margin = select_nearest_option(
        answer,
        options,
        ["A", "B", "C", "D"],
    )

    assert letter == "C"
    assert scores["C"] == 1.0
    assert margin == 0.2


def test_question_type_classification_is_overlapping_and_deterministic():
    question = "What name was written on the sign after I entered the store?"
    assert classify_question_types(question) == [
        "ocr_named_detail",
        "cross_time_ordering",
    ]
    assert primary_question_type(question) == "ocr_named_detail"
