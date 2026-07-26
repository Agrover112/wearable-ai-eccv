import sys
from types import SimpleNamespace
from unittest.mock import patch

from run_generate_longqa_native_video import (
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL,
    DEFAULT_PROXY_HEIGHT,
    DEFAULT_PROXY_WIDTH,
    DEFAULT_VIDEO_FPS,
    NativeVideoVLLMModel,
    build_native_video_payload,
    native_video_fingerprint,
    parse_args,
    prepare_video_proxies,
    resume_position,
)


def test_native_video_payload_uses_one_file_video_url_and_disables_thinking():
    payload = build_native_video_payload(
        DEFAULT_MODEL,
        "file:///scratch/videos/example.mp4",
        "Which option is correct?",
        16,
        0.5,
    )

    content = payload["messages"][0]["content"]
    assert content[0] == {
        "type": "video_url",
        "video_url": {"url": "file:///scratch/videos/example.mp4"},
    }
    assert content[1] == {"type": "text", "text": "Which option is correct?"}
    assert payload["mm_processor_kwargs"] == {
        "fps": 0.5,
        "do_sample_frames": True,
    }
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "image_url" not in str(payload)


def test_native_server_configures_fps_sampling_and_unbounded_video_frames():
    model = NativeVideoVLLMModel(
        DEFAULT_MODEL,
        "/scratch/videos",
        video_fps=0.5,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
    )
    model._port = 12345
    args = model.server_args()

    assert args[args.index("--allowed-local-media-path") + 1] == "/scratch/videos"
    assert args[args.index("--limit-mm-per-prompt") + 1] == '{"video": 1}'
    processor = args[args.index("--mm-processor-kwargs") + 1]
    assert processor == '{"fps":0.5,"do_sample_frames":true}'
    assert args[args.index("--media-io-kwargs") + 1] == '{"video":{"num_frames":-1}}'
    assert args[args.index("--max-model-len") + 1] == "131072"
    assert args[args.index("--gdn-prefill-backend") + 1] == "triton"
    assert args[args.index("--mm-processor-cache-gb") + 1] == "0"


def test_native_video_defaults_target_qwen35_and_quarter_fps():
    with patch.object(sys, "argv", ["run_generate_longqa_native_video.py"]):
        args = parse_args()

    assert args.llm_model == DEFAULT_MODEL
    assert args.video_fps == DEFAULT_VIDEO_FPS
    assert args.max_model_len == DEFAULT_MAX_MODEL_LEN
    assert args.concurrency == 1
    assert args.proxy_width == DEFAULT_PROXY_WIDTH
    assert args.proxy_height == DEFAULT_PROXY_HEIGHT


def test_fingerprint_changes_when_fps_changes():
    base = SimpleNamespace(
        input="/scratch/input.jsonl",
        subset_file="/scratch/subset.json",
        video_folder="/scratch/videos",
        allowed_local_media_path="/scratch/videos",
        llm_model=DEFAULT_MODEL,
        video_fps=0.25,
        proxy_width=480,
        proxy_height=256,
        proxy_codec="libx264",
        proxy_crf=18,
        proxy_preset="veryfast",
        max_model_len=131072,
        prompt_variant="baseline",
        max_new_tokens=16,
    )
    changed = SimpleNamespace(**{**vars(base), "video_fps": 0.5})

    assert native_video_fingerprint(base) != native_video_fingerprint(changed)


def test_proxy_preprocessing_streams_fps_scale_and_reuses_cache(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    args = SimpleNamespace(
        video_fps=0.25,
        proxy_width=480,
        proxy_height=256,
        proxy_codec="libx264",
        proxy_crf=18,
        proxy_preset="veryfast",
        proxy_cache_dir=str(tmp_path / "cache"),
        proxy_workers=1,
        require_proxies=False,
    )

    def write_proxy(command, check):
        assert check
        assert command[command.index("-vf") + 1] == (
            "fps=0.25,scale=480:256:flags=lanczos"
        )
        assert command[command.index("-c:v") + 1] == "libx264"
        with open(command[-1], "wb") as handle:
            handle.write(b"proxy")

    with patch("subprocess.run", side_effect=write_proxy) as run:
        paths = prepare_video_proxies([str(source)], args)
        assert run.call_count == 1

    with patch("subprocess.run") as run:
        assert prepare_video_proxies([str(source)], args) == paths
        run.assert_not_called()


def test_resume_requires_matching_row_schema_answer_and_native_fingerprint():
    rows = [
        {"video_path": "first.mp4", "question": "First?"},
        {"video_path": "second.mp4", "question": "Second?"},
    ]
    predictions = [
        {
            "video_path": "first.mp4",
            "question": "First?",
            "mcq_answer": "A",
            "native_video_fingerprint": "abc123",
        },
        {
            "video_path": "second.mp4",
            "question": "Second?",
            "mcq_answer": "B",
            "native_video_fingerprint": "other",
        },
    ]

    assert resume_position(rows, predictions, "abc123") == 1
