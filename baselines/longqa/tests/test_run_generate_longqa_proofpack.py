import numpy as np

from run_generate_longqa_grounded import CandidateFrame
from run_generate_longqa_proofpack import (
    TemporalProgram,
    baseline_uniform_indices,
    build_multi_event_queries,
    build_option_hypotheses,
    build_structured_evidence_prompt,
    compile_temporal_program,
    select_adaq_pack,
    select_eventlet_hybrid,
    select_focus_pack,
    select_mixed_resolution_pack,
    select_option_contrastive_eventlets,
    select_multi_event_pack,
    select_qca_pack,
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


def test_baseline_uniform_indices_match_single_interval_sampling():
    assert baseline_uniform_indices(101, 4) == [0, 25, 50, 75]
    assert baseline_uniform_indices(3, 8) == [0, 1]


def test_multi_event_queries_split_temporally_distinct_clauses():
    queries = build_multi_event_queries(
        {
            "question": (
                "I picked up the red cup, then walked to the sink. "
                "Later I placed it beside the plate. What happened?"
            )
        }
    )
    assert len(queries) == 3
    assert [query.label for query in queries] == ["event_1", "event_2", "event_3"]


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


def test_qca_pack_allocates_exact_chronological_budget():
    candidates = _candidates()
    selected, meta = select_qca_pack(
        candidates,
        [float(index) for index in range(len(candidates))],
        _features(),
        budget=12,
        num_segments=4,
        alpha=0.5,
        beta=0.5,
        temperature=0.5,
        relevance_threshold=0.7,
    )
    assert len(selected) == 12
    assert sum(meta["qca_quotas"]) == 12
    assert [frame.candidate.timestamp for frame in selected] == sorted(
        frame.candidate.timestamp for frame in selected
    )


def test_multi_event_pack_is_unique_chronological_and_capped():
    candidates = _candidates()
    count = len(candidates)
    event_scores = {
        "event_1": [-(index - 4) ** 2 for index in range(count)],
        "event_2": [-(index - 18) ** 2 for index in range(count)],
    }
    selected, meta = select_multi_event_pack(
        candidates,
        event_scores,
        [0.0] * count,
        centers_per_event=1,
        eventlet_radius=1,
        anchor_k=6,
        bridge_k=4,
        final_max_frames=16,
        temporal_nms_seconds=5.0,
    )
    indices = [frame.candidate.index for frame in selected]
    assert len(indices) == 16
    assert len(indices) == len(set(indices))
    assert indices == sorted(indices)
    assert set(meta["event_centers"]) == {"event_1", "event_2"}


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


def test_adaq_pack_is_reproducible_and_exact_budget():
    candidates = _candidates()
    scores = [-(index - 9) ** 2 / 100 for index in range(len(candidates))]
    first, first_meta = select_adaq_pack(
        candidates, scores, 12, 0.5, 0.95, np.random.default_rng(42)
    )
    second, _ = select_adaq_pack(
        candidates, scores, 12, 0.5, 0.95, np.random.default_rng(42)
    )
    assert len(first) == 12
    assert [item.candidate.index for item in first] == [
        item.candidate.index for item in second
    ]
    assert first_meta["adaq_tau"] >= 0.01


def test_focus_pack_allocates_exact_unique_budget():
    candidates = _candidates()
    scores = [-(index - 15) ** 2 / 100 for index in range(len(candidates))]
    selected, meta = select_focus_pack(
        candidates,
        scores,
        budget=12,
        num_arms=6,
        zoom_ratio=0.5,
        extra_samples_per_arm=2,
        top_ratio=0.2,
        temperature=0.06,
        rng=np.random.default_rng(42),
    )
    indices = [item.candidate.index for item in selected]
    assert len(indices) == 12
    assert len(indices) == len(set(indices))
    assert len(meta["focus_selected_arms"]) == 4


def test_mixed_resolution_assigns_requested_tiers():
    candidates = _candidates()
    selected, meta = select_mixed_resolution_pack(
        candidates,
        [float(index) for index in range(len(candidates))],
        high_frames=2,
        medium_frames=3,
        low_frames=7,
        high_pixels=451584,
        medium_pixels=200704,
        low_pixels=50176,
        temperature=0.1,
        rng=np.random.default_rng(42),
    )
    sources = [item.source for item in selected]
    assert len(selected) == 12
    assert sources.count("qframe_high") == 2
    assert sources.count("qframe_medium") == 3
    assert sources.count("qframe_low") == 7
    assert len(meta["frame_max_pixels"]) == 12


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
