"""Video sampling behaviour: stride, looping, limits, timestamps."""

from __future__ import annotations

import pytest

from anpr_edge.config import load_config
from anpr_edge.video_source import VideoSource


def build(env, tmp_path, sample_video, **overrides) -> VideoSource:
    # One pass over the clip unless a test asks for looping; MAX_FRAMES=0 means
    # "no limit" so the loop test can keep the clip going.
    settings = {
        "REALTIME_FPS": "0",
        "LOOP_VIDEO": "false",
        "MAX_FRAMES": "0",
    }
    settings.update(overrides)
    env(VIDEO_PATH=str(sample_video), **settings)
    return VideoSource(load_config().frames)


def test_stride_controls_how_many_frames_are_returned(env, sample_video):
    frames = list(build(env, None, sample_video, FRAME_STRIDE="5").frames())
    # 10 frame clip, every 5th frame -> indices 0 and 5
    assert [info.index for info in frames] == [0, 5]


def test_stride_one_returns_every_frame(env, sample_video):
    frames = list(build(env, None, sample_video, FRAME_STRIDE="1").frames())
    assert [info.index for info in frames] == list(range(10))


def test_max_frames_caps_processing(env, sample_video):
    frames = list(build(env, None, sample_video, MAX_FRAMES="3").frames())
    assert len(frames) == 3


def test_max_frames_zero_means_no_limit(env, sample_video):
    frames = list(build(env, None, sample_video, MAX_FRAMES="0").frames())
    assert len(frames) == 10


def test_frames_carry_metadata_and_bgr_images(env, sample_video):
    source = build(env, None, sample_video)
    infos = list(source.frames())
    info = infos[0]
    assert info.image.ndim == 3 and info.image.shape[2] == 3  # BGR
    assert info.index == 0
    assert info.sampled_index == 0
    assert infos[-1].sampled_index == len(infos) - 1


def test_downscale_preserves_aspect_ratio(env, sample_video):
    source = build(env, None, sample_video, FRAME_MAX_WIDTH="32")
    info = next(iter(source.frames()))
    assert info.image.shape[1] == 32
    assert info.image.shape[0] < 48  # height shrinks with the width


def test_missing_file_raises_a_clear_error(env, tmp_path, sample_video):
    import dataclasses

    env(VIDEO_PATH=str(sample_video), REALTIME_FPS="0")
    frames = dataclasses.replace(load_config().frames, path=tmp_path / "missing.mp4")
    with pytest.raises(FileNotFoundError):
        VideoSource(frames).open()


def test_loop_restarts_the_clip(env, sample_video):
    source = build(env, None, sample_video, LOOP_VIDEO="true", MAX_FRAMES="14")
    # Frame indices keep counting across loops, so downstream stats stay unique.
    assert [info.index for info in source.frames()] == list(range(14))
    assert [info.sampled_index for info in source.frames()] == list(range(14))


def test_published_timestamps_are_iso_utc_and_ordered(env, sample_video):
    source = build(env, None, sample_video)
    stamps = [source.timestamp_for(info) for info in source.frames()]
    assert all(stamp.endswith("Z") for stamp in stamps)
    assert stamps == sorted(stamps)


def test_video_timestamps_advance_with_the_index(env, sample_video):
    source = build(env, None, sample_video)
    infos = list(source.frames())
    assert infos[0].video_time_seconds == pytest.approx(0.0, abs=0.2)
    assert infos[-1].video_time_seconds > infos[0].video_time_seconds
