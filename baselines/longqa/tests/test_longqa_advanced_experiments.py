from run_generate_longqa_likelihood import calibrated_choice
from run_generate_longqa_narrative_gate import parse_numbered_captions


def test_blind_correction_can_change_visual_prior_choice():
    answer, scores = calibrated_choice(
        {"A": -0.5, "B": -0.6, "C": -2.0, "D": -3.0},
        {"A": -0.1, "B": -1.0, "C": -1.5, "D": -2.0},
        0.5,
    )
    assert answer == "B"
    assert scores["B"] > scores["A"]


def test_numbered_caption_parser_preserves_image_alignment():
    captions = parse_numbered_captions("1: pays at till\n3: exits shop\n2: takes receipt", 3)
    assert captions == ["pays at till", "takes receipt", "exits shop"]
