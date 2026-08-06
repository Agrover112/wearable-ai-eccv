from run_generate_longqa_hierarchical_pivot import (
    choose_distinct_centers,
    endpoint_uniform_indices,
    fill_pack,
    relation_allowed,
    representative_positions,
    split_positions,
)


def test_endpoint_uniform_indices_include_both_video_ends():
    assert endpoint_uniform_indices(101, 5) == [0, 25, 50, 75, 100]


def test_coarse_windows_cover_every_candidate_once():
    windows = split_positions(17, 4)
    assert [position for window in windows for position in window] == list(range(17))
    assert representative_positions(windows[0], 3)[0] == 0


def test_distinct_centers_respect_gap_when_possible():
    scores = [0.0, 5.0, 4.0, 0.0, 3.0, 0.0]
    assert choose_distinct_centers(list(range(6)), scores, 3, 2) == [1, 2, 4]


def test_temporal_relation_filter_is_occurrence_aware():
    assert relation_allowed(8, [2, 10], "forward")
    assert relation_allowed(2, [4, 10], "backward")
    assert not relation_allowed(1, [2, 10], "forward")


def test_pack_preserves_priority_and_endpoints():
    candidates = endpoint_uniform_indices(101, 11)
    final, global_indices = fill_pack(candidates, [5, 6, 4], 101, 3, 2, 5)
    assert set([40, 50, 60]).issubset(final)
    assert global_indices == [0, 100]
