import sys
from unittest.mock import patch

from run_generate_longqa_verifier import (
    build_pairwise_verifier_prompt,
    build_verifier_frame_indices,
    build_verifier_prompt,
    parse_args,
    parse_pairwise_choice,
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


def test_verifier_defaults_to_qwen35_9b():
    argv = [
        "run_generate_longqa_verifier.py",
        "--primary-predictions",
        "primary.jsonl",
        "--secondary-predictions",
        "secondary.jsonl",
        "--primary-proofpack",
        "proofpack.jsonl",
        "--output",
        "predictions.jsonl",
    ]
    with patch.object(sys, "argv", argv):
        args = parse_args()

    assert args.llm_model == "Qwen/Qwen3.5-9B"
