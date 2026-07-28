import argparse

from run_generate_longqa_tcot import (
    build_final_pack,
    expand_candidate_neighborhoods,
    has_final_answer_marker,
    inference_fingerprint,
    normalize_local_ids,
    selection_fingerprint,
    selector_schema,
    split_positions,
)


def _args(**overrides):
    values = {
        "mode": "dynamic_segment",
        "llm_model": "Qwen/Qwen3-VL-8B-Instruct",
        "max_frames": 64,
        "candidate_frames": 256,
        "segments": 4,
        "max_selected_per_segment": 6,
        "selector_max_pixels": 50176,
        "neighborhood_radius": 1,
        "selected_quota": 64,
        "uniform_quota": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_segments_are_contiguous_and_cover_every_candidate_once():
    segments = split_positions(10, 4)
    assert [position for segment in segments for position in segment] == list(range(10))
    assert all(right[0] == left[-1] + 1 for left, right in zip(segments, segments[1:]))


def test_selector_ids_are_bounded_deduplicated_and_sorted():
    assert normalize_local_ids([4, "2", 4, 0, 8, "bad", 3], 6, 3) == [2, 3, 4]


def test_selector_schema_allows_irrelevant_segments_to_select_nothing():
    schema = selector_schema(64, 6)
    assert schema["properties"]["frame_ids"]["minItems"] == 0


def test_answer_cot_requires_an_explicit_final_answer_marker():
    assert has_final_answer_marker("Evidence first.\nFinal Answer: C")
    assert not has_final_answer_marker("Option C looks correct, but reasoning continues")


def test_candidate_neighborhood_expansion_uses_candidate_sequence_positions():
    candidates = [0, 10, 20, 30, 40, 50]
    assert expand_candidate_neighborhoods(candidates, [20, 50], 1) == [10, 20, 30, 40, 50]


def test_coverage_pack_preserves_selected_quota_and_adds_distinct_uniform_frames():
    candidates = list(range(0, 2560, 10))
    seeds = candidates[20:100:4]
    final_indices, metadata = build_final_pack(
        candidates,
        seeds,
        total_frames=2560,
        neighborhood_radius=1,
        selected_quota=48,
        uniform_quota=16,
        max_frames=64,
    )
    assert len(final_indices) == 64
    assert len(metadata["selected_pack_indices"]) == 48
    assert len(metadata["uniform_added_indices"]) == 16
    assert not set(metadata["selected_pack_indices"]) & set(
        metadata["uniform_added_indices"]
    )


def test_dynamic_coverage_variants_share_selector_cache_but_not_inference_identity():
    selected_only = _args(selected_quota=64, uniform_quota=0)
    coverage = _args(selected_quota=48, uniform_quota=16)
    assert selection_fingerprint(selected_only) == selection_fingerprint(coverage)
    assert inference_fingerprint(selected_only) != inference_fingerprint(coverage)
