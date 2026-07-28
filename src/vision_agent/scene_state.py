"""Per-session storage of the latest structured scene analysis results.

The WebSocket vision pipeline produces ``analysis_events`` and ``narrations``
for every processed frame. The chat API needs the most recent of those results
to ground Grok answers, so this module keeps a small, bounded, thread-safe
snapshot per session without coupling the vision pipeline to the chat AI.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass(frozen=True, slots=True)
class SceneSnapshot:
    """Immutable view of the latest analysis results for one session."""

    session_id: str
    recent_events: tuple[dict[str, object], ...]
    latest_narrations: tuple[str, ...]
    updated_at_ms: int | None

    @property
    def has_analysis(self) -> bool:
        return self.updated_at_ms is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "recent_events": [dict(event) for event in self.recent_events],
            "latest_narrations": list(self.latest_narrations),
            "updated_at_ms": self.updated_at_ms,
        }


class _SessionState:
    __slots__ = ("recent_events", "latest_narrations", "updated_at_ms")

    def __init__(self, max_recent_events: int, max_narrations: int) -> None:
        self.recent_events: deque[dict[str, object]] = deque(maxlen=max_recent_events)
        self.latest_narrations: deque[str] = deque(maxlen=max_narrations)
        self.updated_at_ms: int | None = None


class SceneStateStore:
    """Bounded, thread-safe store of the latest scene state per session.

    Sessions are evicted least-recently-updated once ``max_sessions`` is
    exceeded, which keeps memory bounded for a long-running server.
    """

    def __init__(
        self,
        *,
        max_recent_events: int = 20,
        max_narrations: int = 5,
        max_sessions: int = 256,
    ) -> None:
        if max_recent_events < 1:
            raise ValueError("max_recent_events must be at least 1")
        if max_narrations < 1:
            raise ValueError("max_narrations must be at least 1")
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        self._max_recent_events = max_recent_events
        self._max_narrations = max_narrations
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, _SessionState] = OrderedDict()
        self._lock = threading.Lock()

    def register(self, session_id: str) -> None:
        """Create an empty state for a session so it is known before frames arrive."""
        with self._lock:
            self._touch(session_id)

    def update(
        self,
        session_id: str,
        *,
        analysis_events: Sequence[Mapping[str, object]] = (),
        narrations: Sequence[str] = (),
        updated_at_ms: int | None = None,
    ) -> None:
        """Record the newest analysis results for a session."""
        with self._lock:
            state = self._touch(session_id)
            for event in analysis_events:
                if isinstance(event, Mapping):
                    state.recent_events.append(dict(event))
            for narration in narrations:
                message = str(narration).strip()
                if message:
                    state.latest_narrations.append(message)
            state.updated_at_ms = updated_at_ms if updated_at_ms is not None else _now_ms()

    def snapshot(self, session_id: str) -> SceneSnapshot | None:
        """Return the latest state for a session, or None if it is unknown."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            return SceneSnapshot(
                session_id=session_id,
                recent_events=tuple(dict(event) for event in state.recent_events),
                latest_narrations=tuple(state.latest_narrations),
                updated_at_ms=state.updated_at_ms,
            )

    def known(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def _touch(self, session_id: str) -> _SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            state = _SessionState(self._max_recent_events, self._max_narrations)
            self._sessions[session_id] = state
        else:
            self._sessions.move_to_end(session_id)
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
        return state
