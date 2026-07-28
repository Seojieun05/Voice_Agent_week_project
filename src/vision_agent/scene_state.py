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
    visible_objects: tuple[dict[str, object], ...]
    recent_events: tuple[dict[str, object], ...]
    latest_narrations: tuple[str, ...]
    updated_at_ms: int | None
    # Internal scene-quality signal (best detection confidence); deliberately
    # excluded from to_dict() so the chat model never sees raw numbers.
    scene_confidence: float | None = None

    @property
    def has_analysis(self) -> bool:
        return self.updated_at_ms is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "visible_objects": [dict(item) for item in self.visible_objects],
            "recent_events": [dict(event) for event in self.recent_events],
            "latest_narrations": list(self.latest_narrations),
            "updated_at_ms": self.updated_at_ms,
        }


class _SessionState:
    __slots__ = (
        "visible_objects",
        "recent_events",
        "latest_narrations",
        "updated_at_ms",
        "scene_confidence",
    )

    def __init__(self, max_recent_events: int, max_narrations: int) -> None:
        self.visible_objects: tuple[dict[str, object], ...] = ()
        self.scene_confidence: float | None = None
        # Events and narrations are stored with the frame time they arrived
        # at so snapshots can drop entries the camera has since moved past.
        self.recent_events: deque[tuple[int, dict[str, object]]] = deque(maxlen=max_recent_events)
        self.latest_narrations: deque[tuple[int, str]] = deque(maxlen=max_narrations)
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
        event_ttl_s: float = 10.0,
        narration_ttl_s: float = 15.0,
    ) -> None:
        if max_recent_events < 1:
            raise ValueError("max_recent_events must be at least 1")
        if max_narrations < 1:
            raise ValueError("max_narrations must be at least 1")
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if event_ttl_s <= 0.0:
            raise ValueError("event_ttl_s must be positive")
        if narration_ttl_s <= 0.0:
            raise ValueError("narration_ttl_s must be positive")
        self._max_recent_events = max_recent_events
        self._max_narrations = max_narrations
        self._max_sessions = max_sessions
        self._event_ttl_ms = int(event_ttl_s * 1000.0)
        self._narration_ttl_ms = int(narration_ttl_s * 1000.0)
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
        visible_objects: Sequence[Mapping[str, object]] | None = None,
        scene_confidence: float | None = None,
        updated_at_ms: int | None = None,
    ) -> None:
        """Record the newest analysis results for a session.

        ``analysis_events`` and ``narrations`` accumulate (bounded); a
        non-None ``visible_objects`` replaces the previous frame's view,
        because it describes what is visible right now.
        """
        with self._lock:
            state = self._touch(session_id)
            resolved_at_ms = updated_at_ms if updated_at_ms is not None else _now_ms()
            for event in analysis_events:
                if isinstance(event, Mapping):
                    state.recent_events.append((resolved_at_ms, dict(event)))
            for narration in narrations:
                message = str(narration).strip()
                if message:
                    state.latest_narrations.append((resolved_at_ms, message))
            if visible_objects is not None:
                state.visible_objects = tuple(
                    dict(item) for item in visible_objects if isinstance(item, Mapping)
                )
                state.scene_confidence = scene_confidence
            state.updated_at_ms = resolved_at_ms

    def snapshot(self, session_id: str) -> SceneSnapshot | None:
        """Return the latest state for a session, or None if it is unknown.

        Events and narrations older than their TTL — measured against the
        newest frame, not the wall clock — are dropped so a camera that moved
        on stops reporting objects it saw earlier. Kept events gain a
        ``seconds_ago`` field so the chat AI can tell past from present.
        """
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            reference_ms = state.updated_at_ms
            recent_events: list[dict[str, object]] = []
            latest_narrations: list[str] = []
            if reference_ms is not None:
                for added_at_ms, event in state.recent_events:
                    age_ms = reference_ms - added_at_ms
                    if age_ms > self._event_ttl_ms:
                        continue
                    aged = dict(event)
                    aged["seconds_ago"] = max(0, round(age_ms / 1000.0))
                    recent_events.append(aged)
                latest_narrations = [
                    message
                    for added_at_ms, message in state.latest_narrations
                    if reference_ms - added_at_ms <= self._narration_ttl_ms
                ]
            return SceneSnapshot(
                session_id=session_id,
                visible_objects=tuple(dict(item) for item in state.visible_objects),
                recent_events=tuple(recent_events),
                latest_narrations=tuple(latest_narrations),
                updated_at_ms=state.updated_at_ms,
                scene_confidence=state.scene_confidence,
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
