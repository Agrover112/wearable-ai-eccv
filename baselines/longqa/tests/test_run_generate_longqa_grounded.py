import numpy as np
import torch
from types import SimpleNamespace

from run_generate_longqa_grounded import (
    CandidateFrame,
    QWEN_FRAME_RETRIEVAL_INSTRUCTION,
    TextImageGrounder,
    _pooled_features,
    load_grounder_feature_cache,
    save_grounder_feature_cache,
    select_per_option_union_frames,
)


class FakeQwenEmbedder:
    def __init__(self):
        self.calls = []

    def encode(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return np.asarray([[1.0, 0.0] for _ in inputs], dtype=np.float32)


def make_qwen_grounder():
    grounder = TextImageGrounder.__new__(TextImageGrounder)
    grounder.is_qwen_vl_embedding = True
    grounder.batch_size = 4
    grounder.model = FakeQwenEmbedder()
    return grounder


def test_qwen_grounder_encodes_images_as_normalized_multimodal_inputs():
    grounder = make_qwen_grounder()
    frames = [CandidateFrame(index=0, timestamp=0.0, image="frame")]

    features = grounder.encode_images(frames)

    inputs, kwargs = grounder.model.calls[0]
    assert inputs == [{"image": "frame"}]
    assert kwargs["normalize_embeddings"] is True
    np.testing.assert_allclose(features, [[1.0, 0.0]])


def test_qwen_grounder_applies_frame_retrieval_instruction_to_text():
    grounder = make_qwen_grounder()

    grounder.encode_texts(["paying for coffee"])

    inputs, kwargs = grounder.model.calls[0]
    assert inputs == ["paying for coffee"]
    assert kwargs["prompt"] == QWEN_FRAME_RETRIEVAL_INSTRUCTION


def test_pooled_features_accepts_legacy_tensor_output():
    features = torch.randn(2, 4)
    assert _pooled_features(features, "image") is features


def test_pooled_features_extracts_transformers_model_output():
    features = torch.randn(2, 4)
    output = SimpleNamespace(pooler_output=features)
    assert _pooled_features(output, "image") is features


def test_pooled_features_rejects_unknown_output():
    try:
        _pooled_features(SimpleNamespace(), "text")
    except TypeError as exc:
        assert "unsupported output" in str(exc)
    else:
        raise AssertionError("Expected unsupported grounder output to raise TypeError")


def test_per_option_union_deduplicates_anchors_by_source_frame_index():
    candidates = [
        CandidateFrame(index=position * 10, timestamp=float(position), image=None)
        for position in range(12)
    ]
    query_scores = {
        "option_A": [float(position) for position in range(12)],
        "option_B": [float(12 - position) for position in range(12)],
        "option_C": [float(position % 3) for position in range(12)],
        "option_D": [float(position % 5) for position in range(12)],
    }

    selected, _ = select_per_option_union_frames(
        candidates,
        query_scores,
        top_k=8,
        top_k_per_option=2,
        anchor_k=8,
        window_radius=0,
        final_max_frames=8,
        temporal_nms_seconds=0.0,
        temporal_nms_candidates=None,
    )

    frame_indices = [frame.candidate.index for frame in selected]
    assert len(frame_indices) <= 8
    assert len(frame_indices) == len(set(frame_indices))


def test_grounder_feature_cache_round_trip(tmp_path):
    candidates = [
        CandidateFrame(index=10, timestamp=1.0, image=None),
        CandidateFrame(index=20, timestamp=2.0, image=None),
    ]
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cache_path = str(tmp_path / "video.npz")

    save_grounder_feature_cache(
        cache_path,
        "google/siglip2-so400m-patch14-384",
        128,
        candidates,
        features,
    )
    loaded = load_grounder_feature_cache(
        cache_path,
        "google/siglip2-so400m-patch14-384",
        128,
    )

    assert loaded is not None
    loaded_candidates, loaded_features = loaded
    assert [frame.index for frame in loaded_candidates] == [10, 20]
    assert [frame.timestamp for frame in loaded_candidates] == [1.0, 2.0]
    np.testing.assert_allclose(loaded_features, features)


def test_grounder_feature_cache_rejects_model_mismatch(tmp_path):
    cache_path = str(tmp_path / "video.npz")
    save_grounder_feature_cache(
        cache_path,
        "old-model",
        128,
        [CandidateFrame(index=10, timestamp=1.0, image=None)],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    assert load_grounder_feature_cache(cache_path, "new-model", 128) is None


def test_grounder_feature_cache_rejects_changed_video(tmp_path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"original")
    cache_path = str(tmp_path / "video.npz")
    save_grounder_feature_cache(
        cache_path,
        "model",
        1,
        [CandidateFrame(index=0, timestamp=0.0, image=None)],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        video_path=str(video_path),
    )
    assert load_grounder_feature_cache(
        cache_path,
        "model",
        1,
        video_path=str(video_path),
    ) is not None

    video_path.write_bytes(b"changed-size")
    assert load_grounder_feature_cache(
        cache_path,
        "model",
        1,
        video_path=str(video_path),
    ) is None
