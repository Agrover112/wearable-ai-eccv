import unittest

from PIL import Image

from run_generate_longqa_object_hints import (
    build_evidence_panel,
    build_object_track_indices,
    choose_base_frames,
    choose_detection_indices,
    classify_question,
    crop_detection,
    extract_object_concepts,
)


class ObjectHintTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "question": "After picking up the cup, which object did I touch?",
            "mcq_options": "A. red kettle\nB. coffee cup\nC. wooden spoon",
        }
        self.selected = [
            {
                "frame_index": index * 10,
                "timestamp": float(index),
                "score": index / 10,
                "source": "anchor" if index % 2 else "directional_target",
            }
            for index in range(10)
        ]

    def test_extract_object_concepts_is_deterministic_and_option_aware(self):
        concepts = extract_object_concepts(self.row, limit=8)
        self.assertEqual(concepts, extract_object_concepts(self.row, limit=8))
        self.assertLessEqual(len(concepts), 8)
        self.assertTrue(any("kettle" in concept for concept in concepts))
        self.assertNotIn("after", concepts)

    def test_question_only_concepts_exclude_distractor_objects(self):
        concepts = extract_object_concepts(self.row, source="question")
        self.assertTrue(any("cup" in concept for concept in concepts))
        self.assertFalse(any("kettle" in concept for concept in concepts))

    def test_detection_indices_combine_focused_and_uniform(self):
        indices = choose_detection_indices(
            self.selected, total_frames=100, proofpack_count=4, uniform_count=4
        )
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue({0, 20, 40, 60}.issubset(indices))
        self.assertLessEqual(len(indices), 8)

    def test_required_detail_frames_survive_base_reduction(self):
        kept = choose_base_frames(self.selected, count=6, required_indices={10, 30})
        kept_indices = {int(item["frame_index"]) for item in kept}
        self.assertEqual(len(kept), 6)
        self.assertTrue({10, 30}.issubset(kept_indices))

    def test_question_routes(self):
        self.assertEqual(classify_question(self.row), "object_detail")
        self.assertEqual(
            classify_question(
                {"question": "How many times did I move the cup?", "mcq_options": ""}
            ),
            "state_or_count",
        )
        self.assertEqual(
            classify_question(
                {"question": "After opening the door, where did I go?", "mcq_options": ""}
            ),
            "spatial",
        )
        self.assertEqual(
            classify_question(
                {"question": "What did I do after lunch?", "mcq_options": ""}
            ),
            "temporal",
        )

    def test_track_neighbours_stay_inside_video(self):
        detection_frames = [
            {
                "frame_index": 1,
                "timestamp": 0.1,
                "detections": [
                    {"label": "cup", "score": 0.9, "box": [1, 1, 20, 20]}
                ],
            },
            {
                "frame_index": 98,
                "timestamp": 9.8,
                "detections": [
                    {"label": "cup", "score": 0.8, "box": [2, 2, 22, 22]}
                ],
            },
        ]
        indices, events = build_object_track_indices(
            detection_frames,
            self.selected,
            fps=10.0,
            total_frames=100,
            max_frames=12,
            event_budget=6,
        )
        self.assertTrue(events)
        self.assertTrue(all(0 <= index < 100 for index in indices))
        self.assertLessEqual(len(indices), 12)

    def test_crop_and_panel_are_valid_images(self):
        image = Image.new("RGB", (640, 480), (100, 110, 120))
        detection = {
            "label": "cup",
            "score": 0.9,
            "box": [200, 120, 300, 300],
        }
        crop = crop_detection(image, detection, timestamp=4.2)
        panel = build_evidence_panel(image, [detection], timestamp=4.2, size=672)
        self.assertGreater(crop.width, 0)
        self.assertGreater(crop.height, 0)
        self.assertEqual(panel.size, (672, 672))


if __name__ == "__main__":
    unittest.main()
