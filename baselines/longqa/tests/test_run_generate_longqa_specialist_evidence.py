import numpy as np

from run_generate_longqa_specialist_evidence import (
    cluster_reid_observations,
    normalize_label,
    summarize_tracks,
)


def test_normalize_label_is_stable():
    assert normalize_label(" Red_Coffee   Mug ") == "red coffee mug"


def test_reid_clustering_separates_dissimilar_instances():
    observations = [
        {"label": "cup", "timestamp": 1.0, "frame_index": 10, "score": 0.9},
        {"label": "cup", "timestamp": 2.0, "frame_index": 20, "score": 0.8},
        {"label": "cup", "timestamp": 3.0, "frame_index": 30, "score": 0.7},
    ]
    features = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    clusters = cluster_reid_observations(observations, features, 0.8)
    assert len(clusters) == 2
    indices, events = summarize_tracks(clusters, 2)
    assert indices == [10, 20, 30]
    assert {event["track_id"] for event in events} == {"T01", "T02"}
