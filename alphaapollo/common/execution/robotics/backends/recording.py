# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Optional per-step episode recording for in-process simulator backends.

The agent loop only sees one observation per tool call, but a ``vla_act``
call can spend dozens of simulator steps; the backend is the only layer
that observes every step. When a backend is constructed with a
``record_dir``, it streams a dense side-channel record of the episode to
disk — nothing in the recording enters the agent transcript or the tool
contracts.

Each episode lands in its own ``episode-NNN`` directory:

- ``episode.json`` — the task/environment metadata captured at reset, plus
  the terminal summary (steps, success, termination reason) once known.
- ``trace.jsonl`` — one line per simulator step: the executed low-level
  action vector, the simulator reward, the termination flags, and the
  proprioceptive state after the step. Step indices match the backend's
  cumulative ``steps_used``, so a transcript's per-tool-call step counts
  segment this trace into per-turn ranges.
- ``episode.mp4`` — every rendered frame of the episode at the simulator
  control rate. Frames are appended to the encoder as they arrive, so
  memory use is bounded regardless of episode length.

Recording is strictly opt-in. Reset-time setup failures remain fail-loud, while
a mid-episode recorder failure is written to the episode summary and does not
replace the simulator's authoritative transition.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = ["EpisodeRecorder", "VideoSink", "VideoSinkFactory", "open_mp4_sink"]


@runtime_checkable
class VideoSink(Protocol):
    """Incremental video encoder: one ``append`` per frame, ``close`` to flush."""

    def append(self, frame: Any) -> None: ...

    def close(self) -> None: ...


VideoSinkFactory = Callable[[Path, int], VideoSink]


class _FFmpegSink:
    """Stream RGB frames to H.264 via imageio's ffmpeg writer."""

    def __init__(self, path: Path, fps: int) -> None:
        try:
            import imageio.v2 as iio
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "episode video recording requires the optional robotics dependencies "
                "(imageio with the ffmpeg plugin)"
            ) from exc
        self._np = np
        self._writer = iio.get_writer(str(path), fps=fps)

    def append(self, frame: Any) -> None:
        self._writer.append_data(self._np.asarray(frame, dtype=self._np.uint8))

    def close(self) -> None:
        self._writer.close()


def open_mp4_sink(path: Path, fps: int) -> VideoSink:
    """Default sink factory: an ffmpeg-backed incremental mp4 writer."""

    return _FFmpegSink(path, fps)


def _json_number(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("recorded values must be finite")
    return value


def _json_ready(value: Any) -> Any:
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return _json_number(value)


class EpisodeRecorder:
    """Stream one episode at a time into ``root/episode-NNN`` directories."""

    def __init__(
        self,
        root: str | Path,
        *,
        fps: int = 20,
        video_sink_factory: VideoSinkFactory = open_mp4_sink,
        clock: Callable[[], str] | None = None,
    ) -> None:
        if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
            raise ValueError("recorder fps must be a positive integer")
        if not callable(video_sink_factory):
            raise TypeError("video_sink_factory must be callable")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self._root = Path(root)
        self._fps = fps
        self._video_sink_factory = video_sink_factory
        self._clock = clock
        self._episode_dir: Path | None = None
        self._trace_file: Any = None
        self._sink: VideoSink | None = None
        self._frames = 0
        self._metadata: dict[str, Any] = {}
        self._steps = 0

    @property
    def active(self) -> bool:
        return self._episode_dir is not None

    @property
    def episode_dir(self) -> Path | None:
        return self._episode_dir

    def start(self, metadata: Mapping[str, Any]) -> Path:
        """Open the next episode directory and persist the reset metadata."""

        if self.active:
            self.finish(reason="superseded_by_reset")
        # Validate the metadata before touching disk: a failure here must not
        # leave a half-started episode behind.
        prepared = {"fps": self._fps, **_json_ready(metadata)}
        if self._clock is not None:
            prepared["started_at"] = self._clock()
        json.dumps(prepared)
        self._root.mkdir(parents=True, exist_ok=True)
        # mkdir itself is the collision test: concurrent recorders sharing one
        # root would otherwise race the exists() check.
        index = sum(1 for entry in self._root.glob("episode-*") if entry.is_dir())
        while True:
            episode_dir = self._root / f"episode-{index:03d}"
            try:
                episode_dir.mkdir()
                break
            except FileExistsError:
                index += 1
        try:
            trace_file = (episode_dir / "trace.jsonl").open("w", encoding="utf-8")
        except BaseException:
            try:
                episode_dir.rmdir()
            except OSError:
                pass
            raise
        # Commit recorder state only once every fallible step has succeeded.
        self._episode_dir = episode_dir
        self._metadata = prepared
        self._trace_file = trace_file
        self._sink = None
        self._frames = 0
        self._steps = 0
        try:
            self._write_episode_json()
        except BaseException:
            self.abort(reason="start_failed")
            raise
        return episode_dir

    def record_step(
        self,
        *,
        step: int,
        action: Sequence[float] | None,
        reward: float | None,
        terminated: bool,
        truncated: bool,
        success: bool | None,
        proprio: Mapping[str, Any] | None = None,
        frame: Any = None,
    ) -> None:
        """Append one simulator step to the trace; ``step`` is cumulative.

        ``action`` is ``None`` only for the reset frame (step 0), which
        anchors the video and the trace before the first commanded action.
        """

        if not self.active:
            raise RuntimeError("record_step requires an active episode (call start first)")
        record: dict[str, Any] = {
            "step": step,
            "action": None if action is None else [float(value) for value in action],
            "reward": None if reward is None else float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": success if success is None else bool(success),
        }
        if proprio:
            record["proprio"] = _json_ready(proprio)
        if self._clock is not None:
            record["timestamp"] = self._clock()
        self._trace_file.write(json.dumps(record) + "\n")
        self._trace_file.flush()
        # The trace line is authoritative for step accounting. Count it as
        # soon as it is flushed so a later video failure cannot make the
        # summary disagree with trace.jsonl.
        if action is not None:
            self._steps += 1
        if frame is not None:
            # Stream to the encoder immediately: buffering whole episodes in
            # RAM would OOM exactly the long runs recording is meant to debug.
            if self._sink is None:
                assert self._episode_dir is not None
                self._sink = self._video_sink_factory(self._episode_dir / "episode.mp4", self._fps)
            self._sink.append(frame)
            self._frames += 1

    def finish(self, *, reason: str) -> None:
        """Close the episode: flush the video and write the terminal summary.

        A sink failure still deactivates the recorder and records an honest
        summary — a recorder stuck "active" with its trace file gone would
        corrupt the next episode — and then re-raises.
        """

        if not self.active:
            return
        episode_dir = self._episode_dir
        assert episode_dir is not None
        trace_file, self._trace_file = self._trace_file, None
        if trace_file is not None:
            trace_file.close()
        sink, self._sink = self._sink, None
        frames, self._frames = self._frames, 0
        steps, self._steps = self._steps, 0
        metadata, self._metadata = self._metadata, {}
        self._episode_dir = None
        summary: dict[str, Any] = {
            "reason": reason,
            "steps_recorded": steps,
            "frames_recorded": frames,
        }
        if self._clock is not None:
            summary["finished_at"] = self._clock()
        try:
            if sink is not None:
                sink.close()
        except BaseException as exc:
            summary["video_error"] = f"{type(exc).__name__}: {exc}"
            metadata["summary"] = summary
            self._write_metadata(episode_dir, metadata)
            raise
        metadata["summary"] = summary
        self._write_metadata(episode_dir, metadata)

    def abort(self, *, reason: str = "aborted", error: BaseException | None = None) -> None:
        """Close an in-progress episode after a reset or recorder failure.

        Abort is deliberately best-effort: the original reset/recording error
        must remain the exception the caller sees, while trace and video
        resources must not leak into the next episode.
        """

        if not self.active:
            return
        episode_dir = self._episode_dir
        assert episode_dir is not None
        trace_file, self._trace_file = self._trace_file, None
        sink, self._sink = self._sink, None
        frames, self._frames = self._frames, 0
        steps, self._steps = self._steps, 0
        metadata, self._metadata = self._metadata, {}
        self._episode_dir = None
        if trace_file is not None:
            try:
                trace_file.close()
            except Exception:
                pass
        if sink is not None:
            try:
                sink.close()
            except Exception:
                pass
        summary: dict[str, Any] = {
            "reason": reason,
            "steps_recorded": steps,
            "frames_recorded": frames,
            "aborted": True,
        }
        if error is not None:
            summary["recorder_error"] = f"{type(error).__name__}: {error}"
        metadata["summary"] = summary
        try:
            self._write_metadata(episode_dir, metadata)
        except Exception:
            pass

    def _write_metadata(self, episode_dir: Path, metadata: Mapping[str, Any]) -> None:
        path = episode_dir / "episode.json"
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def _write_episode_json(self) -> None:
        assert self._episode_dir is not None
        self._write_metadata(self._episode_dir, self._metadata)
