import pytest

from longqa_utils import index_row_aligned_metadata
from run_generate_longqa_verifier import (
    build_pairwise_verifier_prompt,
    build_verifier_frame_indices,
    build_verifier_prompt,
    choose_candidate_from_scores,
    choose_candidate_from_votes,
    parse_pairwise_choice,
    rotate_mcq_options,
)


def test_verifier_pack_is_unique_chronological_and_capped():
    selected = [
        {
            "frame_index": idx * 10,
            "timestamp": float(idx),
            "score": float(idx),
            "source": "pivot" if idx < 4 else "anchor",
        }
        for idx in range(64)
    ]
    frames, meta = build_verifier_frame_indices(
        selected,
        total_frames=1000,
        max_frames=64,
        proofpack_quota=32,
    )
    assert len(frames) == 64
    assert len(frames) == len(set(frames))
    assert frames == sorted(frames)
    assert {0, 10, 20, 30}.issubset(frames)
    assert meta["final_frames"] == 64


def test_verifier_prompt_does_not_assume_either_candidate_is_correct():
    row = {
        "question": "What happened after payment?",
        "mcq_options": "A. Sat B. Left C. Ordered D. Ate",
    }
    prompt = build_verifier_prompt(row, "B", "D")
    assert "proposed options B and D" in prompt
    assert "Do not assume either proposed option is correct" in prompt
    assert "Answer with ONLY the single letter" in prompt


def test_support_contradiction_prompt_requests_all_three_checks():
    row = {
        "question": "What happened after payment?",
        "mcq_options": "A. Sat B. Left C. Ordered D. Ate",
    }
    prompt = build_verifier_prompt(row, "B", "D", "support_contradiction")
    assert "supporting evidence" in prompt
    assert "contradictory evidence" in prompt
    assert "temporal order" in prompt


def test_qwen35_verifier_prompt_uses_json_answer_contract():
    row = {
        "question": "What happened after payment?",
        "mcq_options": "A. Sat B. Left C. Ordered D. Ate",
    }
    prompt = build_verifier_prompt(
        row,
        "B",
        "D",
        answer_prompt_variant="qwen3_5",
    )
    assert 'JSON object in exactly this format: {"answer":"C"}' in prompt
    assert "Return only the final option letter" not in prompt


def test_pairwise_prompt_exposes_only_candidate_semantics():
    row = {
        "question": "What happened after payment?",
        "mcq_options": "A. Sat B. Left C. Ordered D. Ate",
    }
    prompt = build_pairwise_verifier_prompt(row, "B", "D")
    assert "Candidate 1: Left" in prompt
    assert "Candidate 2: Ate" in prompt
    assert "must select one of the two" in prompt
    assert "A. Sat" not in prompt
    assert parse_pairwise_choice("2") == 2
    assert parse_pairwise_choice("Candidate 1") == 1


def test_option_rotations_cover_each_display_position_once():
    row = {
        "question": "What happened?",
        "mcq_options": "A. Alpha B. Beta C. Gamma D. Delta",
    }
    mappings = [rotate_mcq_options(row, offset)[1] for offset in range(4)]
    for original in "ABCD":
        assert {displayed for mapping in mappings for displayed, value in mapping.items() if value == original} == set("ABCD")


def test_vote_aggregation_is_candidate_restricted_and_ties_fall_back():
    answer, counts = choose_candidate_from_votes(["C", "D", "D", "A"], "B", "C")
    assert answer == "C"
    assert counts == {"A": 1, "B": 0, "C": 1, "D": 2}
    answer, _ = choose_candidate_from_votes(["A", "D"], "B", "C")
    assert answer == "B"


def test_candidate_likelihood_restricts_choice_after_score_fusion():
    answer, scores = choose_candidate_from_scores(
        "B",
        "D",
        {"A": 10.0, "B": 1.0, "C": 9.0, "D": 2.0},
        {"A": 10.0, "B": 3.0, "C": 9.0, "D": 2.0},
        {"A": 0.0, "B": -1.0, "C": 0.0, "D": -2.0},
        -0.1,
    )
    assert answer == "B"
    assert scores["A"] > scores["B"]


def test_row_aligned_proofpacks_keep_questions_from_same_video_distinct():
    references = [
        {"video_path": "same.mp4", "question": "First question?"},
        {"video_path": "same.mp4", "question": "Second question?"},
    ]
    metadata = [
        {"video_path": "same.mp4", "selected": [1]},
        {"video_path": "same.mp4", "selected": [2]},
    ]
    indexed = index_row_aligned_metadata(metadata, references, "proof pack")
    assert indexed["same.mp4||First question?"]["selected"] == [1]
    assert indexed["same.mp4||Second question?"]["selected"] == [2]
    with pytest.raises(RuntimeError, match="video mismatch"):
        index_row_aligned_metadata(
            [{"video_path": "wrong.mp4"}, metadata[1]], references, "proof pack"
        )
