"""Video input handling.

The container treats a file as a stand-in for a live camera feed: frames are
read sequentially, optionally looped forever, and optionally throttled to
simulate a real frame rate. Read failures are logged and recovered from rather
than propagated, so a truncated clip degrades instead of killing the process.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from .config import FrameConfig
from .logging_setup import get_logger, log_event

__all__ = ["FrameInfo", "VideoSource"]

_LOG = get_logger("anpr_edge.video")


@dataclass(frozen=True, slots=True)
class FrameInfo:
    """A decoded frame plus the bookkeeping the pipeline needs."""

    image: np.ndarray
    index: int
    sampled_index: int
    video_time_seconds: float
    wall_time: float


class VideoSource:
    """Iterator over sampled frames of ``FrameConfig.path``."""

    def __init__(self, config: FrameConfig) -> None:
        self._config = config
        self._capture: cv2.VideoCapture | None = None
        self._index = 0
        self._loops = 0
        self._fps = 0.0
        self._width = 0
        self._height = 0
        self._frame_count = 0
        self._video_epoch = self._resolve_epoch()

    # ------------------------------------------------------------------ setup
    def _resolve_epoch(self) -> datetime:
        raw = self._config.video_start_time
        if raw:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"VIDEO_START_TIME is not a valid ISO 8601 timestamp: {raw!r}") from exc
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        return datetime.now(timezone.utc)

    def open(self) -> None:
        """Open the file and read its metadata."""
        path = Path(self._config.path)
        if not path.is_file():
            raise FileNotFoundError(f"VIDEO_PATH does not exist: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"OpenCV could not open the video file: {path}")
        self._capture = capture
        self._fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        log_event(
            _LOG,
            20,
            "video_opened",
            path=str(path),
            size_bytes=path.stat().st_size,
            fps=round(self._fps, 3),
            width=self._width,
            height=self._height,
            frame_count=self._frame_count,
            stride=self._config.stride,
            loop=self._config.loop,
        )

    # ---------------------------------------------------------------- iterate
    def frames(self) -> Iterator[FrameInfo]:
        """Yield every ``stride``-th frame, looping when configured to."""
        if self._capture is None:
            self.open()
        assert self._capture is not None

        next_due = time.monotonic()
        frame_period = 1.0 / self._config.realtime_fps if self._config.realtime_fps > 0 else 0.0
        emitted = 0

        while True:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                if self._config.loop:
                    self._reopen()
                    continue
                log_event(
                    _LOG,
                    20,
                    "video_eof",
                    frames_decoded=self._index,
                    frames_sampled=emitted,
                )
                return

            index = self._index
            self._index += 1

            if index % self._config.stride:
                continue
            if self._config.max_width and frame.shape[1] > self._config.max_width:
                scale = self._config.max_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (self._config.max_width, max(1, round(frame.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            if frame_period:
                now = time.monotonic()
                if now < next_due:
                    time.sleep(next_due - now)
                next_due = max(next_due + frame_period, time.monotonic())

            yield FrameInfo(
                image=frame,
                index=index,
                sampled_index=emitted,
                video_time_seconds=(index / self._fps) if self._fps > 0 else float(index),
                wall_time=time.time(),
            )
            emitted += 1
            if self._config.max_frames and emitted >= self._config.max_frames:
                log_event(_LOG, 20, "max_frames_reached", frames_sampled=emitted)
                return

    def _reopen(self) -> None:
        """Rewind (or reopen) the capture after EOF."""
        self._loops += 1
        if self._capture is not None:
            self._capture.release()
        self.open()
        log_event(_LOG, 20, "video_looped", loop=self._loops, path=self._config.path)

    def timestamp_for(self, info: FrameInfo) -> str:
        """Return the ISO 8601 UTC timestamp to publish for this frame."""
        if self._config.timestamp_source == "video":
            moment = self._video_epoch + timedelta(seconds=info.video_time_seconds)
        else:
            moment = datetime.fromtimestamp(info.wall_time, tz=timezone.utc)
        return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    # ------------------------------------------------------------- life cycle
    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            log_event(_LOG, 20, "video_closed", loops=self._loops, frames_decoded=self._index)

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
