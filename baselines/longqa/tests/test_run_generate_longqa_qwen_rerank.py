from run_generate_longqa_grounded import CandidateFrame
from run_generate_longqa_proofpack import TemporalProgram
from run_generate_longqa_qwen_rerank import (
    build_final_frame_indices,
    build_siglip_shortlist,
    select_reranked_centers,
)


def _candidates(count=128):
    return [CandidateFrame(index=index * 10, timestamp=index * 5.0, image=None) for index in range(count)]


def test_siglip_shortlist_preserves_all_option_queries():
    candidates = _candidates()
    components = {"target": [0.0] * 128}
    for option_index, letter in enumerate("ABCD"):
        scores = [0.0] * 128
        scores[10 + option_index * 20] = 1.0
        components[f"option_{letter}"] = scores
    pivot = [0.0] * 128
    pivot[5] = 1.0
    selected = build_siglip_shortlist(candidates, components, pivot, 24, 8.0)
    assert len(selected) == 24
    assert 5 in selected
    assert {10, 30, 50, 70}.issubset(selected)


def test_temporal_reranking_keeps_option_centers_after_anchor():
    candidates = _candidates(32)
    centers = list(range(4, 28))
    option_scores = {}
    for option_index, letter in enumerate("ABCD"):
        option_scores[f"option_{letter}"] = [
            0.9 if center == 16 + option_index * 2 else 0.1 for center in centers
        ]
    pivot_scores = [0.95 if center == 12 else 0.05 for center in centers]
    program = TemporalProgram("AFTER", "anchor", "forward", "target")
    selected, metadata = select_reranked_centers(
        centers,
        candidates,
        option_scores,
        pivot_scores,
        program,
        "temporal",
        final_centers=8,
        contrast_weight=0.5,
        nms_seconds=8.0,
    )
    assert len(selected) == 8
    assert metadata["anchor_center"] == 12
    for values in metadata["option_centers"].values():
        assert values
        assert all(center > 12 for center in values)


def test_final_pack_has_exact_budget_and_video_endpoints():
    candidates = _candidates()
    selected, retrieval, global_indices = build_final_frame_indices(
        candidates,
        centers=[8, 20, 32, 44, 56, 68, 80, 92],
        total_video_frames=2000,
        window_radius=1,
        global_frames=40,
        final_frames=64,
    )
    assert len(selected) == 64
    assert len(retrieval) == 24
    assert len(global_indices) == 40
    assert selected[0] == 0
    assert selected[-1] == 1999
