from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .event_manager import (
    DESCRIPTION_CONFIRMED,
    OBJECT_APPEARED,
    OBJECT_APPROACHING,
    OBJECT_DISAPPEARED,
    OBJECT_STATE_CHANGED,
    SCREEN_CHANGED,
    TEXT_CONFIRMED,
)
from .object_types import KIOSK_OBJECT_TYPES, SIGNAL_OBJECT_TYPES
from .types import AnalysisEvent

_UNKNOWN_STATES = {"", "UNKNOWN", "UNCERTAIN", "NONE"}
_STATE_LABELS = {
    "GREEN": "초록색",
    "RED": "빨간색",
    "YELLOW": "노란색",
}
_OBJECT_LABELS = {
    "pedestrian_signal": "보행자 신호",
    "vehicle_traffic_light": "차량 신호",
    "traffic_light": "신호등",
    "bus": "버스",
    "kiosk": "키오스크",
    "self_service_kiosk": "무인 키오스크",
    "touchscreen_kiosk": "터치스크린 키오스크",
    "sign": "표지판",
    "stop_sign": "표지판",
    "display": "전광판",
    "screen": "화면",
    "monitor": "화면",
    "tv": "화면",
    "ticket_machine": "발권기",
    "reverse_vending_machine": "빈 용기 회수기",
    "bus_route_display": "버스 노선 표시기",
    "unknown_panel": "알 수 없는 조작 패널",
    "person": "사람",
    "car": "자동차",
    "vehicle": "차량",
}


@dataclass(frozen=True, slots=True)
class Narration:
    """One deterministic utterance candidate; it does not perform TTS."""

    message: str
    priority: int
    event: AnalysisEvent


def _normalized_state(value: object) -> str | None:
    if value is None:
        return None
    raw_value = getattr(value, "value", value)
    normalized = str(raw_value).strip().upper()
    return None if normalized in _UNKNOWN_STATES else normalized


def _object_label(object_type: str) -> str:
    normalized = object_type.strip().lower().replace("-", "_").replace(" ", "_")
    return _OBJECT_LABELS.get(normalized, f"{normalized} 객체")


def _with_subject_particle(label: str) -> str:
    last_character = label[-1]
    hangul_offset = ord(last_character) - 0xAC00
    has_final_consonant = 0 <= hangul_offset <= 0xD7A3 - 0xAC00 and hangul_offset % 28 != 0
    return f"{label}{'이' if has_final_consonant else '가'}"


def _string_attribute(event: AnalysisEvent, name: str) -> str | None:
    value = event.attributes.get(name)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _visible_options(event: AnalysisEvent) -> list[str]:
    raw_options = event.attributes.get("visible_options")
    if not isinstance(raw_options, (list, tuple)):
        return []
    return [str(option).strip() for option in raw_options if str(option).strip()]


class NarrationPolicy:
    """Choose concise Korean template messages from structured scene events."""

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.5,
        duplicate_cooldown_s: float = 5.0,
        max_messages_per_batch: int = 1,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        if duplicate_cooldown_s < 0.0:
            raise ValueError("duplicate_cooldown_s must be non-negative")
        if max_messages_per_batch < 1:
            raise ValueError("max_messages_per_batch must be at least 1")

        self.minimum_confidence = minimum_confidence
        self.duplicate_cooldown_s = duplicate_cooldown_s
        self.max_messages_per_batch = max_messages_per_batch
        self._last_narrated_at: dict[tuple[object, ...], float] = {}

    @staticmethod
    def priority_for(event: AnalysisEvent) -> int:
        object_type = event.object_type.strip().lower().replace(" ", "_")
        if event.event_type == OBJECT_STATE_CHANGED and object_type in SIGNAL_OBJECT_TYPES:
            return 1
        if event.event_type == OBJECT_APPROACHING and object_type in {
            "bus",
            "car",
            "vehicle",
        }:
            return 2
        if event.event_type == TEXT_CONFIRMED and object_type == "bus":
            return 3
        if event.event_type == SCREEN_CHANGED and object_type in KIOSK_OBJECT_TYPES:
            return 4
        if event.event_type == TEXT_CONFIRMED:
            return 5
        if event.event_type == DESCRIPTION_CONFIRMED:
            return 6
        if event.event_type in {OBJECT_APPEARED, OBJECT_DISAPPEARED}:
            return 7
        if event.event_type == OBJECT_STATE_CHANGED:
            return 7
        return 100

    def message_for(self, event: AnalysisEvent) -> str | None:
        """Return a pure template result without changing duplicate history."""
        if event.is_uncertain:
            return None
        if event.confidence < self.minimum_confidence:
            return None

        object_type = event.object_type.strip().lower().replace(" ", "_")
        if event.event_type == OBJECT_STATE_CHANGED:
            previous_state = _normalized_state(event.previous_state)
            current_state = _normalized_state(event.current_state)
            if previous_state is None or current_state is None or previous_state == current_state:
                return None
            state_label = _STATE_LABELS.get(current_state)
            if state_label is None:
                return None
            if object_type == "pedestrian_signal":
                return f"보행자 신호가 {state_label}으로 바뀌었습니다."
            if object_type == "vehicle_traffic_light":
                return f"차량 신호가 {state_label}으로 바뀌었습니다."
            if object_type == "traffic_light":
                return f"신호등 표시가 {state_label}으로 바뀌었습니다."
            return f"{_object_label(object_type)} 상태가 {state_label}으로 바뀌었습니다."

        if event.event_type == OBJECT_APPROACHING:
            if object_type == "bus":
                route_number = _string_attribute(event, "route_number")
                if route_number is not None:
                    return f"{route_number}번 버스가 들어오고 있습니다."
                return "버스가 접근하고 있습니다."
            if object_type in {"car", "vehicle"}:
                return f"{_with_subject_particle(_object_label(object_type))} 접근하고 있습니다."
            return None

        if event.event_type == TEXT_CONFIRMED:
            if object_type == "bus":
                route_number = _string_attribute(event, "route_number")
                return f"{route_number}번 버스입니다." if route_number is not None else None
            text = _string_attribute(event, "text")
            if text is None:
                return None
            return f"{_object_label(object_type)}에 {text}라고 표시되어 있습니다."

        if event.event_type == SCREEN_CHANGED:
            if object_type not in KIOSK_OBJECT_TYPES:
                return None
            object_label = _object_label(object_type)
            options = _visible_options(event)
            if len(options) == 2:
                return f"{options[0]}와 {options[1]} 중 하나를 선택하는 화면입니다."
            if options:
                return f"{object_label} 화면에 {', '.join(options)} 선택지가 있습니다."
            return f"{object_label} 화면이 바뀌었습니다."

        if event.event_type == DESCRIPTION_CONFIRMED:
            description = _string_attribute(event, "description")
            return description

        if event.event_type == OBJECT_APPEARED:
            return f"{_with_subject_particle(_object_label(object_type))} 감지되었습니다."
        if event.event_type == OBJECT_DISAPPEARED:
            return f"{_with_subject_particle(_object_label(object_type))} 화면에서 사라졌습니다."
        return None

    @staticmethod
    def _deduplication_key(event: AnalysisEvent, message: str) -> tuple[object, ...]:
        semantic_identity = (
            event.attributes.get("screen_fingerprint")
            if event.event_type == SCREEN_CHANGED
            else None
        )
        return (
            event.event_type,
            event.object_type,
            event.stable_id,
            _normalized_state(event.previous_state),
            _normalized_state(event.current_state),
            semantic_identity,
            message,
        )

    def _is_recent_duplicate(self, event: AnalysisEvent, message: str) -> bool:
        key = self._deduplication_key(event, message)
        previous_timestamp = self._last_narrated_at.get(key)
        if previous_timestamp is None:
            return False
        return event.timestamp_s - previous_timestamp < self.duplicate_cooldown_s

    def select(self, events: Sequence[AnalysisEvent]) -> list[Narration]:
        """Select at most the configured number of highest-priority messages."""
        candidates: list[tuple[int, int, Narration]] = []
        for index, event in enumerate(events):
            message = self.message_for(event)
            if message is None or self._is_recent_duplicate(event, message):
                continue
            priority = self.priority_for(event)
            if priority >= 100:
                continue
            candidates.append((priority, index, Narration(message, priority, event)))

        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
        selected = [candidate[2] for candidate in candidates[: self.max_messages_per_batch]]
        for narration in selected:
            key = self._deduplication_key(narration.event, narration.message)
            self._last_narrated_at[key] = narration.event.timestamp_s
        return selected

    def narrate(self, events: AnalysisEvent | Sequence[AnalysisEvent]) -> list[str]:
        """Return deterministic messages for one event or one simultaneous batch."""
        batch = [events] if isinstance(events, AnalysisEvent) else events
        return [narration.message for narration in self.select(batch)]
