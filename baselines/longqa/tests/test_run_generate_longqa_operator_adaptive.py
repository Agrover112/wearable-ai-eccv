from run_generate_longqa_operator_adaptive import (
    EDGE_ANCHOR_COUNT,
    GLOBAL_ANCHOR_COUNT,
    MAX_OPERATOR_ADAPTIVE_FRAMES,
    _frame_metadata_record,
    build_operator_adaptive_frame_plan,
    compile_temporal_program,
    operator_adaptive_fingerprint,
    parse_args,
)


def _parent_record():
    selected = [
        {"frame_index": 700, "score": 0.99, "source": "pivot"},
        {"frame_index": 1600, "score": 0.95, "source": "directional_target"},
        {"frame_index": 2600, "score": 0.90, "source": "event_center"},
        {"frame_index": 3600, "score": 0.85, "source": "event_center"},
        {"frame_index": 4600, "score": 0.80, "source": "event_center"},
        {"frame_index": 5600, "score": 0.75, "source": "event_center"},
    ]
    return {
        "selected": selected,
        "selection_meta": {
            "pivot_centers": [700],
            "target_centers": [1600],
            "event_centers": [1600, 2600, 3600, 4600, 5600],
        },
        "proofpack_fingerprint": "parent123",
    }


def _assert_valid_plan(plan, total_frames=9000, budget=64):
    indices = [frame.frame_index for frame in plan]
    assert len(plan) == budget
    assert len(plan) <= MAX_OPERATOR_ADAPTIVE_FRAMES
    assert len(indices) == len(set(indices))
    assert indices == sorted(indices)
    assert all(0 <= frame_index < total_frames for frame_index in indices)


def test_global_policy_is_uniform64_without_parent_evidence():
    plan = build_operator_adaptive_frame_plan(
        total_frames=9000,
        fps=15.0,
        parent_record=_parent_record(),
        program=compile_temporal_program("What was visible in the room?"),
    )
    _assert_valid_plan(plan)
    assert {frame.source for frame in plan} == {"uniform_global"}
    assert {frame.policy for frame in plan} == {"uniform64"}


def test_after_policy_retains_anchors_and_uses_native_fps_directional_bursts():
    plan = build_operator_adaptive_frame_plan(
        total_frames=9000,
        fps=15.0,
        parent_record=_parent_record(),
        program=compile_temporal_program("After paying, what did I do next?"),
    )
    _assert_valid_plan(plan)
    assert sum(frame.is_global_anchor for frame in plan) == GLOBAL_ANCHOR_COUNT
    assert any(
        frame.source == "after_pivot_burst"
        and frame.center_frame_index == 700
        and frame.offset_seconds == 1.0
        and frame.frame_index == 715
        for frame in plan
    )
    assert any(
        frame.source == "after_target_burst"
        and frame.center_frame_index == 1600
        and frame.offset_seconds == 4.0
        and frame.frame_index == 1660
        for frame in plan
    )


def test_before_policy_uses_backward_target_bursts():
    plan = build_operator_adaptive_frame_plan(
        total_frames=9000,
        fps=15.0,
        parent_record=_parent_record(),
        program=compile_temporal_program("What happened before paying?"),
    )
    _assert_valid_plan(plan)
    assert sum(frame.is_global_anchor for frame in plan) == GLOBAL_ANCHOR_COUNT
    assert any(
        frame.source == "before_target_burst"
        and frame.center_frame_index == 1600
        and frame.offset_seconds == -4.0
        and frame.frame_index == 1540
        for frame in plan
    )


def test_first_policy_biases_neighborhoods_toward_earliest_relevant_centers():
    plan = build_operator_adaptive_frame_plan(
        9000,
        15.0,
        _parent_record(),
        compile_temporal_program("Which item did I encounter first?"),
    )
    _assert_valid_plan(plan)
    assert sum(frame.is_global_anchor for frame in plan) == EDGE_ANCHOR_COUNT
    assert any(
        frame.source == "first_event_neighborhood" and frame.center_frame_index == 700
        for frame in plan
    )


def test_last_policy_biases_neighborhoods_toward_latest_relevant_centers():
    plan = build_operator_adaptive_frame_plan(
        9000,
        15.0,
        _parent_record(),
        compile_temporal_program("Which item did I encounter last?"),
    )
    _assert_valid_plan(plan)
    assert sum(frame.is_global_anchor for frame in plan) == EDGE_ANCHOR_COUNT
    assert any(
        frame.source == "last_event_neighborhood" and frame.center_frame_index == 5600
        for frame in plan
    )


def test_state_change_policy_pairs_pre_and_post_bursts():
    plan = build_operator_adaptive_frame_plan(
        total_frames=9000,
        fps=15.0,
        parent_record=_parent_record(),
        program=compile_temporal_program("How did the object change?"),
    )
    _assert_valid_plan(plan)
    assert sum(frame.is_global_anchor for frame in plan) == GLOBAL_ANCHOR_COUNT
    assert any(frame.source == "state_change_pre_burst" for frame in plan)
    assert any(frame.source == "state_change_post_burst" for frame in plan)


def test_small_budget_is_bounded_and_keeps_required_anchor_positions():
    plan = build_operator_adaptive_frame_plan(
        total_frames=9000,
        fps=15.0,
        parent_record=_parent_record(),
        program=compile_temporal_program("After paying, what did I do next?"),
        max_frames=32,
    )
    _assert_valid_plan(plan, budget=32)
    assert sum(frame.is_global_anchor for frame in plan) == GLOBAL_ANCHOR_COUNT


def test_defaults_and_fingerprint_track_operator_adaptive_configuration():
    args = parse_args(
        [
            "--parent-proofpack",
            "runs/parent/proofpack.jsonl",
            "--frame-metadata-output",
            "frames.jsonl",
            "--output",
            "predictions.jsonl",
        ]
    )
    assert args.max_frames == MAX_OPERATOR_ADAPTIVE_FRAMES
    assert args.llm_model == "Qwen/Qwen3.5-9B"
    assert args.backend == "vllm"
    original = operator_adaptive_fingerprint(args)
    args.max_frames = 32
    assert operator_adaptive_fingerprint(args) != original


def test_frame_metadata_records_operator_policy_source_and_offsets():
    parent = _parent_record()
    program = compile_temporal_program("After paying, what did I do next?")
    plan = build_operator_adaptive_frame_plan(9000, 15.0, parent, program)
    record = _frame_metadata_record(
        0,
        {"id": "sample", "video_path": "video.mp4"},
        parent,
        "runs/parent/proofpack.jsonl",
        "adaptive123",
        15.0,
        9000,
        program,
        plan,
    )
    burst = next(frame for frame in record["selected"] if frame["source"] == "after_target_burst")
    assert record["operator"] == "AFTER"
    assert record["policy"] == "after_directional_bursts"
    assert burst["center_frame_index"] == 1600
    assert burst["offset_seconds"] is not None
