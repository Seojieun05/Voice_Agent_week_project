from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum

from .types import AnalysisEvent, AnalysisResult, SceneEvent

OBJECT_APPEARED = "OBJECT_APPEARED"
OBJECT_DISAPPEARED = "OBJECT_DISAPPEARED"
OBJECT_STATE_CHANGED = "OBJECT_STATE_CHANGED"
TEXT_CONFIRMED = "TEXT_CONFIRMED"
SCREEN_CHANGED = "SCREEN_CHANGED"
OBJECT_APPROACHING = "OBJECT_APPROACHING"
DESCRIPTION_CONFIRMED = "DESCRIPTION_CONFIRMED"

_UNKNOWN_STATES = {"", "UNKNOWN", "UNCERTAIN", "NONE"}
_PENDING_APPEARED = "PENDING_APPEARED"
_DEFAULT_SIGNAL_DETECTION_CONFIDENCE = 0.2
_LEGACY_EVENT_TYPES = {
    "appeared": OBJECT_APPEARED,
    "disappeared": OBJECT_DISAPPEARED,
    "signal_changed": OBJECT_STATE_CHANGED,
}


def _normalize_stable_id(value: str) -> str:
    """Return the bare stable ID used by analyzer results and legacy events."""
    normalized = value.strip()
    if ":" in normalized:
        normalized = normalized.rsplit(":", 1)[-1]
    return normalized


def _normalize_object_type(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def _normalize_state(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    normalized = str(value).strip().upper()
    return None if normalized in _UNKNOWN_STATES else normalized


def _confidence(value: object, fallback: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = fallback
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        normalized = fallback
    return normalized if math.isfinite(normalized) and 0.0 <= normalized <= 1.0 else 0.0


class SceneEventManager:
    """Turn structured object analyses and legacy presence events into scene events.

    In standalone mode, the first result for an object produces ``OBJECT_APPEARED``
    and its first missing update produces ``OBJECT_DISAPPEARED``. A pipeline that
    already owns presence/state stabilization can disable both derivations and pass
    its existing :class:`SceneEvent` values through ``scene_events`` instead.
    """

    def __init__(
        self,
        *,
        auto_presence: bool = True,
        derive_state_changes: bool = True,
        derive_domain_events: bool = True,
        minimum_approach_confidence: float = 0.5,
        minimum_presence_confidence: float = 0.5,
        minimum_domain_confidence: float = 0.5,
    ) -> None:
        for name, value in (
            ("minimum_approach_confidence", minimum_approach_confidence),
            ("minimum_presence_confidence", minimum_presence_confidence),
            ("minimum_domain_confidence", minimum_domain_confidence),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        self.auto_presence = auto_presence
        self.derive_state_changes = derive_state_changes
        self.derive_domain_events = derive_domain_events
        self.minimum_approach_confidence = minimum_approach_confidence
        self.minimum_presence_confidence = minimum_presence_confidence
        self.minimum_domain_confidence = minimum_domain_confidence
        self._active_results: dict[str, AnalysisResult] = {}
        self._known_states: dict[str, str] = {}
        self._presence_states: dict[str, str] = {}
        self._domain_values: dict[tuple[str, str], object] = {}
        self._emitted_signatures: set[tuple[object, ...]] = set()

    @staticmethod
    def _freeze(value: object) -> object:
        if isinstance(value, Mapping):
            return tuple(
                sorted(
                    (str(key), SceneEventManager._freeze(item))
                    for key, item in value.items()
                )
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return tuple(SceneEventManager._freeze(item) for item in value)
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    @staticmethod
    def _signature(event: AnalysisEvent) -> tuple[object, ...]:
        return (
            event.event_type,
            event.stable_id,
            event.previous_state,
            event.current_state,
            event.timestamp_s,
            SceneEventManager._freeze(event.attributes),
        )

    def _append_once(self, events: list[AnalysisEvent], event: AnalysisEvent) -> bool:
        signature = self._signature(event)
        if signature in self._emitted_signatures:
            return False
        self._emitted_signatures.add(signature)
        events.append(event)
        return True

    @staticmethod
    def _event_from_result(
        event_type: str,
        result: AnalysisResult,
        timestamp_s: float,
        *,
        previous_state: str | None = None,
        current_state: str | None = None,
        confidence: float | None = None,
        is_uncertain: bool | None = None,
    ) -> AnalysisEvent:
        return AnalysisEvent(
            event_type=event_type,
            object_type=_normalize_object_type(result.object_type),
            stable_id=_normalize_stable_id(result.stable_id),
            timestamp_s=timestamp_s,
            previous_state=previous_state,
            current_state=current_state,
            confidence=result.confidence if confidence is None else confidence,
            attributes=dict(result.attributes),
            is_uncertain=result.is_uncertain if is_uncertain is None else is_uncertain,
        )

    @staticmethod
    def _detector_confidence(result: AnalysisResult | None) -> float | None:
        if result is None:
            return None
        raw_confidence = result.attributes.get("detection_confidence")
        if raw_confidence is None:
            return None
        return _confidence(raw_confidence, 0.0)

    @classmethod
    def _presence_confidence(cls, result: AnalysisResult | None) -> float:
        detection_confidence = cls._detector_confidence(result)
        if detection_confidence is None:
            # Injected/custom analyzers may not expose detector evidence. Preserve
            # the legacy stabilized-presence contract rather than inheriting OCR
            # or state-analysis uncertainty from result.confidence.
            return 1.0
        return detection_confidence

    @classmethod
    def _combined_evidence_confidence(
        cls,
        result: AnalysisResult,
        raw_confidence: object,
    ) -> float:
        evidence_confidence = _confidence(raw_confidence, 0.0)
        detection_confidence = cls._detector_confidence(result)
        if detection_confidence is None:
            return evidence_confidence
        return min(evidence_confidence, detection_confidence)

    @classmethod
    def _state_confidence(cls, result: AnalysisResult) -> float:
        object_type = _normalize_object_type(result.object_type)
        if object_type in {"bus", "car", "vehicle"}:
            return cls._combined_evidence_confidence(
                result,
                result.attributes.get("motion_confidence", result.confidence),
            )
        if object_type in {"traffic_light", "pedestrian_signal"}:
            # TrafficLightAnalyzer already rejects detector evidence below its
            # dedicated small-object threshold before producing a known state.
            detection_confidence = cls._detector_confidence(result)
            minimum_detection_confidence = _confidence(
                result.attributes.get("minimum_detection_confidence"),
                _DEFAULT_SIGNAL_DETECTION_CONFIDENCE,
            )
            if (
                detection_confidence is not None
                and detection_confidence < minimum_detection_confidence
            ):
                return 0.0
            return _confidence(result.confidence, 0.0)
        return cls._combined_evidence_confidence(result, result.confidence)

    def _forget_object(self, stable_id: str, *, remove_deduplication: bool = False) -> None:
        self._active_results.pop(stable_id, None)
        self._known_states.pop(stable_id, None)
        self._domain_values = {
            key: value for key, value in self._domain_values.items() if key[0] != stable_id
        }
        if remove_deduplication:
            self._presence_states.pop(stable_id, None)
            self._emitted_signatures = {
                signature
                for signature in self._emitted_signatures
                if len(signature) < 2 or signature[1] != stable_id
            }

    def reset(self, stable_id: str | None = None) -> None:
        """Forget one retired object, or all event history when omitted."""
        if stable_id is None:
            self._active_results.clear()
            self._known_states.clear()
            self._presence_states.clear()
            self._domain_values.clear()
            self._emitted_signatures.clear()
            return
        self._forget_object(
            _normalize_stable_id(stable_id),
            remove_deduplication=True,
        )

    def _derive_domain_specific_events(
        self,
        result: AnalysisResult,
        timestamp_s: float,
        *,
        observed_state: str | None,
        events: list[AnalysisEvent],
    ) -> None:
        if not self.derive_domain_events:
            return

        stable_id = _normalize_stable_id(result.stable_id)
        object_type = _normalize_object_type(result.object_type)
        if (
            object_type == "kiosk"
            and result.attributes.get("screen_is_confirmed", not result.is_uncertain) is True
        ):
            raw_fingerprint = result.attributes.get("screen_fingerprint")
            screen_fingerprint = (
                str(raw_fingerprint).strip()
                if raw_fingerprint is not None and str(raw_fingerprint).strip()
                else (
                    observed_state,
                    self._freeze(result.attributes.get("visible_options")),
                    self._freeze(result.attributes.get("visible_text")),
                )
            )
            value_key = (stable_id, "screen")
            if self._domain_values.get(value_key) != screen_fingerprint:
                raw_confidence = result.attributes.get(
                    "screen_confidence",
                    result.confidence,
                )
                screen_confidence = self._combined_evidence_confidence(
                    result,
                    raw_confidence,
                )
                if screen_confidence >= self.minimum_domain_confidence:
                    appended = self._append_once(
                        events,
                        self._event_from_result(
                            SCREEN_CHANGED,
                            result,
                            timestamp_s,
                            confidence=screen_confidence,
                            is_uncertain=False,
                        ),
                    )
                    if appended:
                        self._domain_values[value_key] = screen_fingerprint

        if result.is_uncertain:
            return

        approaching_key = (stable_id, "approaching")
        motion_confidence = self._combined_evidence_confidence(
            result,
            result.attributes.get("motion_confidence", result.confidence),
        )
        if (
            object_type in {"bus", "car", "vehicle"}
            and observed_state == "APPROACHING"
            and motion_confidence >= self.minimum_approach_confidence
            and self._domain_values.get(approaching_key) is not True
        ):
            appended = self._append_once(
                events,
                self._event_from_result(
                    OBJECT_APPROACHING,
                    result,
                    timestamp_s,
                    confidence=motion_confidence,
                ),
            )
            if appended:
                self._domain_values[approaching_key] = True
        elif (
            observed_state is not None
            and observed_state != "APPROACHING"
            and motion_confidence >= self.minimum_approach_confidence
        ):
            self._domain_values.pop(approaching_key, None)

        attribute_name = "route_number" if object_type == "bus" else "text"
        raw_text = result.attributes.get(attribute_name)
        confirmed_text = str(raw_text).strip() if raw_text is not None else ""
        text_is_known = confirmed_text.upper() not in _UNKNOWN_STATES
        if confirmed_text and text_is_known:
            value_key = (stable_id, attribute_name)
            if self._domain_values.get(value_key) != confirmed_text:
                event_confidence = result.confidence
                if object_type == "bus":
                    event_confidence = self._combined_evidence_confidence(
                        result,
                        result.attributes.get("route_confidence", result.confidence),
                    )
                else:
                    event_confidence = self._combined_evidence_confidence(
                        result,
                        event_confidence,
                    )
                if event_confidence >= self.minimum_domain_confidence:
                    appended = self._append_once(
                        events,
                        self._event_from_result(
                            TEXT_CONFIRMED,
                            result,
                            timestamp_s,
                            confidence=event_confidence,
                        ),
                    )
                    if appended:
                        self._domain_values[value_key] = confirmed_text

        description = result.attributes.get("description")
        confirmed_description = str(description).strip() if description is not None else ""
        if confirmed_description:
            value_key = (stable_id, "description")
            if self._domain_values.get(value_key) != confirmed_description:
                description_confidence = self._combined_evidence_confidence(
                    result,
                    result.confidence,
                )
                if description_confidence >= self.minimum_domain_confidence:
                    appended = self._append_once(
                        events,
                        self._event_from_result(
                            DESCRIPTION_CONFIRMED,
                            result,
                            timestamp_s,
                            confidence=description_confidence,
                        ),
                    )
                    if appended:
                        self._domain_values[value_key] = confirmed_description

    def _apply_legacy_event(
        self,
        scene_event: SceneEvent,
        result_by_id: dict[str, AnalysisResult],
        events: list[AnalysisEvent],
    ) -> None:
        event_type = _LEGACY_EVENT_TYPES.get(scene_event.event_type.lower())
        if event_type is None:
            return

        stable_id = _normalize_stable_id(scene_event.object_key)
        matching_result = result_by_id.get(stable_id) or self._active_results.get(stable_id)
        object_type = _normalize_object_type(
            matching_result.object_type if matching_result is not None else scene_event.class_name
        )
        confidence = self._state_confidence(matching_result) if matching_result else 1.0
        attributes = dict(matching_result.attributes) if matching_result is not None else {}
        is_uncertain = matching_result.is_uncertain if matching_result is not None else False

        if event_type in {OBJECT_APPEARED, OBJECT_DISAPPEARED}:
            # Presence is a state machine, so a buggy/retried legacy producer cannot
            # cause repeated appearance or disappearance announcements. The legacy
            # engine has already stabilized presence; an analyzer's uncertain OCR or
            # state result must not make that independent evidence uncertain.
            presence_state = self._presence_states.get(stable_id)
            if presence_state == event_type:
                return
            if event_type == OBJECT_APPEARED:
                presence_confidence = self._presence_confidence(matching_result)
                if presence_confidence < self.minimum_presence_confidence:
                    self._presence_states[stable_id] = _PENDING_APPEARED
                    return
            else:
                if presence_state == _PENDING_APPEARED:
                    self._presence_states[stable_id] = OBJECT_DISAPPEARED
                    self._forget_object(stable_id)
                    return
                presence_confidence = 1.0
            translated = AnalysisEvent(
                event_type=event_type,
                object_type=object_type,
                stable_id=stable_id,
                timestamp_s=scene_event.timestamp_s,
                confidence=presence_confidence,
                attributes=attributes,
                is_uncertain=False,
            )
            if self._append_once(events, translated):
                self._presence_states[stable_id] = event_type
            if event_type == OBJECT_DISAPPEARED:
                self._forget_object(stable_id)
            return

        previous_state = _normalize_state(scene_event.previous_state)
        current_state = _normalize_state(scene_event.current_state)
        if previous_state is None or current_state is None or previous_state == current_state:
            return
        if confidence < self.minimum_domain_confidence:
            return

        # The existing event engine is authoritative for pipeline timing. Accept
        # its explicit previous state even when this manager has no prior frame.
        if self._known_states.get(stable_id) == current_state:
            return
        translated = AnalysisEvent(
            event_type=OBJECT_STATE_CHANGED,
            object_type=object_type,
            stable_id=stable_id,
            timestamp_s=scene_event.timestamp_s,
            previous_state=previous_state,
            current_state=current_state,
            confidence=confidence,
            attributes=attributes,
            is_uncertain=is_uncertain,
        )
        if self._append_once(events, translated):
            self._known_states[stable_id] = current_state

    def update(
        self,
        results: Sequence[AnalysisResult],
        timestamp_s: float,
        *,
        scene_events: Sequence[SceneEvent] = (),
    ) -> list[AnalysisEvent]:
        """Process one complete scene update and return only new events."""
        events: list[AnalysisEvent] = []
        result_by_id = {_normalize_stable_id(result.stable_id): result for result in results}

        # Legacy events go first because their timestamps and explicit transition
        # states are canonical in the current YOLO pipeline.
        for scene_event in scene_events:
            self._apply_legacy_event(scene_event, result_by_id, events)

        current_ids = set(result_by_id)
        previous_active_ids = set(self._active_results)

        for stable_id, result in result_by_id.items():
            presence_state = self._presence_states.get(stable_id)
            if (
                presence_state != OBJECT_APPEARED
                and (self.auto_presence or presence_state == _PENDING_APPEARED)
            ):
                presence_confidence = self._presence_confidence(result)
                if presence_confidence >= self.minimum_presence_confidence:
                    appeared = self._event_from_result(
                        OBJECT_APPEARED,
                        result,
                        timestamp_s,
                        confidence=presence_confidence,
                        is_uncertain=False,
                    )
                    if self._append_once(events, appeared):
                        self._presence_states[stable_id] = OBJECT_APPEARED
                elif presence_state != OBJECT_APPEARED:
                    self._presence_states[stable_id] = _PENDING_APPEARED

            observed_state = None if result.is_uncertain else _normalize_state(result.state)
            previous_state = self._known_states.get(stable_id)
            state_confidence = self._state_confidence(result)
            if (
                self.derive_state_changes
                and observed_state is not None
                and state_confidence >= self.minimum_domain_confidence
            ):
                if previous_state is not None and observed_state != previous_state:
                    changed = self._event_from_result(
                        OBJECT_STATE_CHANGED,
                        result,
                        timestamp_s,
                        previous_state=previous_state,
                        current_state=observed_state,
                    )
                    self._append_once(events, changed)
                self._known_states[stable_id] = observed_state

            self._derive_domain_specific_events(
                result,
                timestamp_s,
                observed_state=observed_state,
                events=events,
            )

            self._active_results[stable_id] = result

        if self.auto_presence:
            for stable_id in sorted(previous_active_ids - current_ids):
                previous_result = self._active_results.pop(stable_id)
                presence_state = self._presence_states.get(stable_id)
                if presence_state == _PENDING_APPEARED:
                    self._presence_states[stable_id] = OBJECT_DISAPPEARED
                elif presence_state != OBJECT_DISAPPEARED:
                    disappeared = self._event_from_result(
                        OBJECT_DISAPPEARED,
                        previous_result,
                        timestamp_s,
                    )
                    if self._append_once(events, disappeared):
                        self._presence_states[stable_id] = OBJECT_DISAPPEARED
                self._forget_object(stable_id)

        return events
