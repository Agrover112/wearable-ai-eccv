import numpy as np

from run_generate_longqa_grounded import CandidateFrame
from run_generate_longqa_proofpack import (
    TemporalProgram,
    build_option_hypotheses,
    build_structured_evidence_prompt,
    compile_temporal_program,
    select_eventlet_hybrid,
    select_option_contrastive_eventlets,
    select_temporal_pivot_pack,
)


def _candidates(count=24):
    return [
        CandidateFrame(index=idx * 10, timestamp=float(idx * 5), image=None)
        for idx in range(count)
    ]


def _features(count=24, dimensions=8):
    values = np.arange(1, count * dimensions + 1, dtype=np.float32).reshape(
        count, dimensions
    )
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_compile_temporal_program_extracts_leading_pivot():
    program = compile_temporal_program(
        "After paying for coffee, what activity did I perform next?"
    )
    assert program.operator == "AFTER"
    assert program.direction == "forward"
    assert program.pivot == "paying for coffee"


def test_compile_temporal_program_extracts_embedded_before_pivot():
    program = compile_temporal_program("What did I pick up before entering the store?")
    assert program.operator == "BEFORE"
    assert program.direction == "backward"
    assert program.pivot == "entering the store"


def test_first_questions_preserve_multiple_events():
    program = compile_temporal_program("Which item did I encounter first and see later?")
    assert program.operator == "FIRST"
    assert program.direction == "multi_event"


def test_eventlet_hybrid_is_chronological_unique_and_capped():
    candidates = _candidates()
    scores = [float(idx) for idx in range(len(candidates))]
    selected, _ = select_eventlet_hybrid(
        candidates,
        scores,
        _features(),
        event_centers=4,
        eventlet_radius=1,
        anchor_k=8,
        boundary_k=4,
        final_max_frames=16,
        temporal_nms_seconds=5.0,
    )
    indices = [frame.candidate.index for frame in selected]
    timestamps = [frame.candidate.timestamp for frame in selected]
    assert len(indices) == 16
    assert len(indices) == len(set(indices))
    assert timestamps == sorted(timestamps)


def test_option_contrastive_eventlets_balance_options_and_cap_frames():
    candidates = _candidates()
    count = len(candidates)
    option_scores = {
        "option_A": [-(idx - 3) ** 2 for idx in range(count)],
        "option_B": [-(idx - 8) ** 2 for idx in range(count)],
        "option_C": [-(idx - 14) ** 2 for idx in range(count)],
        "option_D": [-(idx - 20) ** 2 for idx in range(count)],
    }
    selected, meta = select_option_contrastive_eventlets(
        candidates,
        option_scores,
        _features(),
        centers_per_option=2,
        eventlet_radius=1,
        anchor_k=8,
        final_max_frames=24,
        temporal_nms_seconds=5.0,
    )
    assert set(meta["option_centers"]) == set(option_scores)
    assert all(len(centers) == 2 for centers in meta["option_centers"].values())
    assert len(selected) <= 24
    assert len({frame.candidate.index for frame in selected}) == len(selected)


def test_temporal_pivot_after_constrains_target_centers_forward():
    candidates = _candidates()
    count = len(candidates)
    pivot_scores = [-(idx - 10) ** 2 for idx in range(count)]
    target_scores = [float(count - idx) for idx in range(count)]
    selected, meta = select_temporal_pivot_pack(
        candidates,
        pivot_scores,
        target_scores,
        _features(),
        TemporalProgram("AFTER", "payment", "forward", "next activity"),
        pivot_centers=1,
        target_centers=3,
        eventlet_radius=1,
        anchor_k=4,
        bridge_k=4,
        final_max_frames=20,
        temporal_nms_seconds=5.0,
    )
    assert meta["pivot_centers"] == [100]
    assert all(index > 100 for index in meta["target_centers"])
    assert len(selected) <= 20


def test_temporal_pivot_before_constrains_target_centers_backward():
    candidates = _candidates()
    count = len(candidates)
    pivot_scores = [-(idx - 10) ** 2 for idx in range(count)]
    target_scores = [float(idx) for idx in range(count)]
    _, meta = select_temporal_pivot_pack(
        candidates,
        pivot_scores,
        target_scores,
        _features(),
        TemporalProgram("BEFORE", "entry", "backward", "previous activity"),
        pivot_centers=1,
        target_centers=3,
        eventlet_radius=1,
        anchor_k=4,
        bridge_k=4,
        final_max_frames=20,
        temporal_nms_seconds=5.0,
    )
    assert all(index < 100 for index in meta["target_centers"])


def test_temporal_pivot_uniform_coverage_fill_avoids_boundary_frames():
    candidates = _candidates()
    count = len(candidates)
    pivot_scores = [-(idx - 10) ** 2 for idx in range(count)]
    target_scores = [-(idx - 18) ** 2 for idx in range(count)]
    selected, meta = select_temporal_pivot_pack(
        candidates,
        pivot_scores,
        target_scores,
        _features(),
        TemporalProgram("AFTER", "payment", "forward", "next activity"),
        pivot_centers=1,
        target_centers=2,
        eventlet_radius=1,
        anchor_k=4,
        bridge_k=2,
        final_max_frames=20,
        temporal_nms_seconds=5.0,
        fill_mode="uniform_coverage",
    )
    sources = {frame.source for frame in selected}
    assert len(selected) == 20
    assert "coverage_fill" in sources
    assert "semantic_boundary" not in sources
    assert meta["fill_mode"] == "uniform_coverage"


def test_option_hypotheses_are_separate_queries():
    row = {
        "question": "What happened next?",
        "mcq_options": "A. Sat down B. Left the room C. Paid D. Ate",
    }
    queries = build_option_hypotheses(row)
    assert [query.label for query in queries] == [
        "option_A",
        "option_B",
        "option_C",
        "option_D",
    ]
    assert "Hypothesis A: Sat down" in queries[0].text


def test_structured_prompt_preserves_direct_mcq_and_evidence_roles():
    row = {
        "question": "What happened after payment?",
        "mcq_options": "A. Sat B. Left C. Ordered D. Ate",
    }
    selected = [
        {"source": "anchor", "timestamp": 10.0},
        {"source": "pivot", "timestamp": 20.0},
        {"source": "directional_target", "timestamp": 30.0},
    ]
    prompt = build_structured_evidence_prompt(
        row,
        selected,
        {"operator": "AFTER", "direction": "forward", "pivot": "payment"},
    )
    assert "images 2@20.0s" in prompt
    assert "Temporal operator: AFTER" in prompt
    assert "Question: What happened after payment?" in prompt
    assert "Answer with ONLY the single letter" in prompt
