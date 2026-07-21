from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SignalState(str, Enum):
    """Conservative traffic-signal color result."""

    RED = "RED"
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class Detection:
    """One post-processed detection whose track ID is the raw tracker value."""

    frame_index: int
    timestamp_s: float
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SceneEvent:
    """A stable change whose object key is independent of the raw tracker ID."""

    event_type: str
    object_key: str
    class_name: str
    timestamp_s: float
    message: str
    previous_state: SignalState | None = None
    current_state: SignalState | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.previous_state is None:
            payload.pop("previous_state")
        else:
            payload["previous_state"] = self.previous_state.value
        if self.current_state is None:
            payload.pop("current_state")
        else:
            payload["current_state"] = self.current_state.value
        return payload


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Structured output shared by every object-specific analyzer."""

    object_type: str
    stable_id: str
    state: str | None
    confidence: float
    attributes: dict[str, object] = field(default_factory=dict)
    is_uncertain: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if isinstance(self.state, Enum):
            payload["state"] = self.state.value
        return payload


@dataclass(frozen=True, slots=True)
class AnalysisEvent:
    """A deduplicatable scene event derived from an :class:`AnalysisResult`."""

    event_type: str
    object_type: str
    stable_id: str
    timestamp_s: float
    previous_state: str | None = None
    current_state: str | None = None
    confidence: float = 0.0
    attributes: dict[str, object] = field(default_factory=dict)
    is_uncertain: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field_name in ("previous_state", "current_state"):
            value = getattr(self, field_name)
            if value is None:
                payload.pop(field_name)
            elif isinstance(value, Enum):
                payload[field_name] = value.value
        return payload
