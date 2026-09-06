from run_generate_longqa_proofpack import TemporalProgram
from run_generate_longqa_verified_bursts import (
    build_final_indices,
    choose_coarse_proposals,
    relation_eligible,
    temporal_burst_indices,
)
from run_answer_longqa_selection import build_vertical_detail_panel


def test_after_relation_uses_windows_after_verified_anchor():
    program = TemporalProgram("AFTER", "opened the cabinet", "forward", "item")
    assert relation_eligible(program, 3, 8) == {4, 5, 6, 7}


def test_unrestricted_target_survives_wrong_anchor_filter():
    program = TemporalProgram("AFTER", "opened the cabinet", "forward", "item")
    scores = {
        "anchor": [0.1, 0.2, 0.3, 0.95, 0.1, 0.1],
        # The strongest target is before the proposed anchor; it must remain for recall.
        "target": [0.1, 0.9, 0.2, 0.1, 0.7, 0.3],
    }
    proposals = choose_coarse_proposals(scores, program, 5, 0.5, 0.1)
    selected = {(item["role"], item["window"], item["reason"]) for item in proposals}
    assert ("target", 1, "best_unrestricted_target") in selected
    assert ("target", 4, "best_relation_compatible_target") in selected


def test_first_last_operator_preserves_both_qualified_extremes():
    program = TemporalProgram("MULTI_TIME", "", "multi_event", "first and last")
    scores = {"target": [0.1, 0.86, 0.2, 0.84, 0.3]}
    proposals = choose_coarse_proposals(scores, program, 5, 0.5, 0.05)
    selected = {item["window"] for item in proposals if item["role"] == "target"}
    assert {1, 3}.issubset(selected)


def test_burst_is_one_second_scale_not_candidate_spacing():
    indices = temporal_burst_indices(
        center=3000,
        fps=30.0,
        total_frames=10000,
        count=5,
        span_seconds=4.0,
    )
    assert indices == [2940, 2970, 3000, 3030, 3060]


def test_final_pack_has_exact_budget_and_endpoints():
    selected, bursts, global_indices = build_final_indices(
        centers=[1000, 2000, 3000, 4000, 5000, 6000],
        fps=30.0,
        total_frames=9001,
        burst_frames=5,
        burst_span_seconds=4.0,
        global_frames=32,
        final_frames=64,
    )
    assert len(selected) == 64
    assert len(bursts) == 30
    assert len(global_indices) == 32
    assert selected[0] == 0
    assert selected[-1] == 9000


def test_detail_panel_preserves_full_view_and_two_enlarged_halves():
    from PIL import Image

    source = Image.new("RGB", (1280, 720), (40, 80, 120))
    panel = build_vertical_detail_panel(source, 672)
    assert panel.size == (672, 672)
