from argparse import Namespace

from run_generate_longqa_dense_bursts import (
    DEFAULT_BURST_OFFSETS_SECONDS,
    GLOBAL_ANCHOR_COUNT,
    build_dense_burst_frame_plan,
    dense_burst_fingerprint,
    rank_event_centers,
)


def _parent_record(selected, selection_meta=None):
    return {
        "selected": selected,
        "selection_meta": selection_meta or {},
        "proofpack_fingerprint": "parent123",
    }


def _selected(frame_index, score, source="event_center"):
    return {"frame_index": frame_index, "score": score, "source": source}


def test_dense_bursts_clip_to_video_boundaries_with_native_fps():
    parent = _parent_record(
        [_selected(1, 0.9), _selected(98, 0.8)],
        {"event_centers": [1, 98]},
    )
    plan = build_dense_burst_frame_plan(
        total_frames=100,
        fps=15.0,
        parent_record=parent,
        max_frames=64,
    )
    indices = [frame.frame_index for frame in plan]
    bursts = [frame for frame in plan if frame.source == "event_burst"]
    assert all(0 <= index < 100 for index in indices)
    assert 0 in indices and 99 in indices
    assert any(frame.center_frame_index == 1 and frame.offset_seconds == 0.33 for frame in bursts)
    assert any(frame.center_frame_index == 98 and frame.offset_seconds == -0.33 for frame in bursts)


def test_dense_burst_plan_is_unique_chronological_and_exact_at_budget():
    parent = _parent_record(
        [_selected(index, 1.0 - rank / 10) for rank, index in enumerate(range(100, 900, 100))],
        {"event_centers": list(range(100, 900, 100))},
    )
    plan = build_dense_burst_frame_plan(1000, 15.0, parent)
    indices = [frame.frame_index for frame in plan]
    assert len(plan) == 64
    assert len(indices) == len(set(indices))
    assert indices == sorted(indices)
    assert sum(frame.source == "global_anchor" for frame in plan) == GLOBAL_ANCHOR_COUNT


def test_dense_burst_fill_keeps_an_exact_budget_after_anchor_burst_collisions():
    parent = _parent_record(
        [_selected(0, 0.9), _selected(999, 0.8)],
        {"event_centers": [0, 999]},
    )
    plan = build_dense_burst_frame_plan(1000, 15.0, parent)
    assert len(plan) == 64
    assert sum(frame.source == "global_anchor" for frame in plan) == GLOBAL_ANCHOR_COUNT
    assert any(frame.source == "coverage_fill" for frame in plan)


def test_rank_event_centers_uses_parent_scores_then_deduplicates_indices():
    parent = _parent_record(
        [
            _selected(20, 0.3),
            _selected(80, 0.9),
            _selected(40, 0.7),
        ],
        {"event_centers": [20, 80, 40, 80]},
    )
    ranked = rank_event_centers(parent)
    assert [center.frame_index for center in ranked[:3]] == [80, 40, 20]
    assert ranked[0].source == "parent_event_center"


def test_dense_burst_fingerprint_tracks_sampling_configuration():
    args = Namespace(
        parent_proofpack="runs/parent/proofpack.jsonl",
        event_centers=8,
        burst_offsets_seconds=list(DEFAULT_BURST_OFFSETS_SECONDS),
        max_frames=64,
    )
    original = dense_burst_fingerprint(args)
    args.burst_offsets_seconds = [-1.0, 1.0]
    changed_offsets = dense_burst_fingerprint(args)
    args.burst_offsets_seconds = list(DEFAULT_BURST_OFFSETS_SECONDS)
    args.event_centers = 4
    changed_centers = dense_burst_fingerprint(args)
    assert original != changed_offsets
    assert original != changed_centers
