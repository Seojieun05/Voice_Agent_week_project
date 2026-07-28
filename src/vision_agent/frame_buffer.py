"""Per-session in-memory ring buffer of recent JPEG frames.

The VLM fallback needs one recent, sharp frame per chat question. Frames are
kept only in memory, never written to disk or logs, and old sessions are
evicted least-recently-updated so a long-running server stays bounded.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass(frozen=True, slots=True)
class BufferedFrame:
    """One JPEG frame kept for potential VLM use."""

    sequence_id: int
    received_at_ms: int
    jpeg_bytes: bytes


class FrameRingBuffer:
    """Bounded, thread-safe ring buffer of recent JPEG frames per session."""

    def __init__(self, *, max_frames: int = 5, max_sessions: int = 32) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        self._max_frames = max_frames
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, deque[BufferedFrame]] = OrderedDict()
        self._lock = threading.Lock()

    def append(
        self,
        session_id: str,
        jpeg_bytes: bytes,
        *,
        sequence_id: int,
        received_at_ms: int | None = None,
    ) -> None:
        frame = BufferedFrame(
            sequence_id=sequence_id,
            received_at_ms=received_at_ms if received_at_ms is not None else _now_ms(),
            jpeg_bytes=jpeg_bytes,
        )
        with self._lock:
            frames = self._sessions.get(session_id)
            if frames is None:
                frames = deque(maxlen=self._max_frames)
                self._sessions[session_id] = frames
            else:
                self._sessions.move_to_end(session_id)
            frames.append(frame)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def recent(
        self,
        session_id: str,
        *,
        max_age_ms: int | None = None,
        now_ms: int | None = None,
    ) -> list[BufferedFrame]:
        """Return buffered frames oldest-first, optionally dropping stale ones."""
        with self._lock:
            frames = self._sessions.get(session_id)
            if frames is None:
                return []
            snapshot = list(frames)
        if max_age_ms is None:
            return snapshot
        reference_ms = now_ms if now_ms is not None else _now_ms()
        return [frame for frame in snapshot if reference_ms - frame.received_at_ms <= max_age_ms]
