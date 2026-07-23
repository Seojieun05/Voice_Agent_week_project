from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .types import Detection, SceneEvent, SignalState


@dataclass(slots=True)
class _ObjectState:
    stable_id: int
    last_detection: Detection
    seen_streak: int = 0
    missed_streak: int = 0
    announced: bool = False
    confirmed_signal_state: SignalState | None = None
    candidate_signal_state: SignalState | None = None
    candidate_signal_streak: int = 0


@dataclass(frozen=True, slots=True)
class FrameEventUpdate:
    """Events and stable keys aligned with the input detection order."""

    events: tuple[SceneEvent, ...]
    object_keys: tuple[str, ...]
    retired_object_keys: tuple[str, ...] = ()


def _class_key(detection: Detection) -> tuple[int, str]:
    return detection.class_id, detection.class_name


def _raw_track_key(detection: Detection) -> tuple[int, str, int] | None:
    if detection.track_id is None:
        return None
    return detection.class_id, detection.class_name, detection.track_id


def _intersection_over_union(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second

    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection_area = intersection_width * intersection_height

    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area if union_area > 0.0 else 0.0


class StableObjectEventEngine:
    """Stabilize object presence and optional externally classified signal states."""

    def __init__(
        self,
        *,
        min_seen_frames: int = 3,
        max_missed_frames: int = 8,
        reconnect_iou_threshold: float = 0.3,
        max_reconnect_frames: int = 3,
        min_signal_state_frames: int = 3,
    ) -> None:
        if min_seen_frames < 1:
            raise ValueError("min_seen_frames must be at least 1")
        if max_missed_frames < 1:
            raise ValueError("max_missed_frames must be at least 1")
        if not 0.0 < reconnect_iou_threshold <= 1.0:
            raise ValueError("reconnect_iou_threshold must be greater than 0 and at most 1")
        if max_reconnect_frames < 0:
            raise ValueError("max_reconnect_frames must be at least 0")
        if min_signal_state_frames < 1:
            raise ValueError("min_signal_state_frames must be at least 1")

        self.min_seen_frames = min_seen_frames
        self.max_missed_frames = max_missed_frames
        self.reconnect_iou_threshold = reconnect_iou_threshold
        self.max_reconnect_frames = max_reconnect_frames
        self.min_signal_state_frames = min_signal_state_frames
        self._states: dict[int, _ObjectState] = {}
        self._raw_track_to_stable: dict[tuple[int, str, int], int] = {}
        self._next_stable_id = 1

    def reset(self) -> None:
        """Forget all tracked objects and restart stable-ID allocation."""
        self._states.clear()
        self._raw_track_to_stable.clear()
        self._next_stable_id = 1

    @staticmethod
    def _object_key(state: _ObjectState) -> str:
        return f"{state.last_detection.class_name}:stable-{state.stable_id}"

    def _create_state(self, detection: Detection) -> int:
        stable_id = self._next_stable_id
        self._next_stable_id += 1
        self._states[stable_id] = _ObjectState(
            stable_id=stable_id,
            last_detection=detection,
        )
        return stable_id

    def _delete_state(self, stable_id: int) -> None:
        del self._states[stable_id]
        self._raw_track_to_stable = {
            raw_key: mapped_stable_id
            for raw_key, mapped_stable_id in self._raw_track_to_stable.items()
            if mapped_stable_id != stable_id
        }

    def _assign_detections(self, detections: list[Detection]) -> dict[int, int]:
        """Return a one-to-one mapping of detection index to internal stable ID."""

        assignments: dict[int, int] = {}
        assigned_stable_ids: set[int] = set()

        # A tracker-provided ID is the strongest signal while its state is active.
        for detection_index, detection in enumerate(detections):
            raw_key = _raw_track_key(detection)
            if raw_key is None:
                continue

            stable_id = self._raw_track_to_stable.get(raw_key)
            if stable_id is None:
                continue
            if stable_id not in self._states:
                del self._raw_track_to_stable[raw_key]
                continue
            if stable_id in assigned_stable_ids:
                continue

            assignments[detection_index] = stable_id
            assigned_stable_ids.add(stable_id)

        # Match the remaining detections globally by descending IoU. Sorting the
        # tie-breakers makes the greedy result independent of dictionary ordering.
        spatial_candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            if detection_index in assignments:
                continue

            for stable_id, state in self._states.items():
                if stable_id in assigned_stable_ids:
                    continue
                if state.missed_streak > self.max_reconnect_frames:
                    continue
                if _class_key(detection) != _class_key(state.last_detection):
                    continue

                iou = _intersection_over_union(detection.xyxy, state.last_detection.xyxy)
                if iou >= self.reconnect_iou_threshold:
                    spatial_candidates.append((iou, detection_index, stable_id))

        spatial_candidates.sort(key=lambda candidate: (-candidate[0], candidate[2], candidate[1]))
        for _iou, detection_index, stable_id in spatial_candidates:
            if detection_index in assignments or stable_id in assigned_stable_ids:
                continue
            assignments[detection_index] = stable_id
            assigned_stable_ids.add(stable_id)

        for detection_index, detection in enumerate(detections):
            if detection_index in assignments:
                continue
            stable_id = self._create_state(detection)
            assignments[detection_index] = stable_id
            assigned_stable_ids.add(stable_id)

        return assignments

    @staticmethod
    def _reset_signal_candidate(state: _ObjectState) -> None:
        state.candidate_signal_state = None
        state.candidate_signal_streak = 0

    def _update_signal_state(
        self,
        state: _ObjectState,
        observed_state: SignalState | None,
        timestamp_s: float,
    ) -> SceneEvent | None:
        if not state.announced or observed_state in (None, SignalState.UNKNOWN):
            self._reset_signal_candidate(state)
            return None

        if observed_state is state.confirmed_signal_state:
            self._reset_signal_candidate(state)
            return None

        if observed_state is state.candidate_signal_state:
            state.candidate_signal_streak += 1
        else:
            state.candidate_signal_state = observed_state
            state.candidate_signal_streak = 1

        if state.candidate_signal_streak < self.min_signal_state_frames:
            return None

        previous_state = state.confirmed_signal_state
        state.confirmed_signal_state = observed_state
        self._reset_signal_candidate(state)
        if previous_state is None:
            return None

        state_labels = {
            SignalState.GREEN: "초록색",
            SignalState.RED: "빨간색",
            SignalState.YELLOW: "노란색",
        }
        class_name = state.last_detection.class_name
        return SceneEvent(
            event_type="signal_changed",
            object_key=self._object_key(state),
            class_name=class_name,
            timestamp_s=timestamp_s,
            message=(
                f"신호등 표시가 {state_labels[previous_state]}에서 "
                f"{state_labels[observed_state]}으로 바뀌었습니다."
            ),
            previous_state=previous_state,
            current_state=observed_state,
        )

    def update_frame(
        self,
        detections: list[Detection],
        timestamp_s: float,
        *,
        signal_states: Sequence[SignalState | None] | None = None,
    ) -> FrameEventUpdate:
        if signal_states is None:
            normalized_signal_states: Sequence[SignalState | None] = (None,) * len(detections)
        elif len(signal_states) != len(detections):
            raise ValueError("signal_states must have the same length as detections")
        else:
            normalized_signal_states = signal_states

        events: list[SceneEvent] = []
        retired_object_keys: list[str] = []
        assignments = self._assign_detections(detections)
        seen_stable_ids = set(assignments.values())

        for detection_index, detection in enumerate(detections):
            stable_id = assignments[detection_index]
            state = self._states[stable_id]
            state.seen_streak += 1
            state.missed_streak = 0
            state.last_detection = detection

            raw_key = _raw_track_key(detection)
            mapped_stable_id = (
                self._raw_track_to_stable.get(raw_key) if raw_key is not None else None
            )
            if raw_key is not None and (
                mapped_stable_id is None
                or mapped_stable_id == stable_id
                or mapped_stable_id not in self._states
            ):
                self._raw_track_to_stable[raw_key] = stable_id

            if not state.announced and state.seen_streak >= self.min_seen_frames:
                state.announced = True
                events.append(
                    SceneEvent(
                        event_type="appeared",
                        object_key=self._object_key(state),
                        class_name=detection.class_name,
                        timestamp_s=timestamp_s,
                        message=f"새로운 {detection.class_name} 객체가 감지되었습니다.",
                    )
                )

            signal_event = self._update_signal_state(
                state,
                normalized_signal_states[detection_index],
                timestamp_s,
            )
            if signal_event is not None:
                events.append(signal_event)

        object_keys = tuple(
            self._object_key(self._states[assignments[index]]) for index in range(len(detections))
        )

        for stable_id, state in list(self._states.items()):
            if stable_id in seen_stable_ids:
                continue

            state.missed_streak += 1
            state.seen_streak = 0
            self._reset_signal_candidate(state)
            if state.announced and state.missed_streak >= self.max_missed_frames:
                class_name = state.last_detection.class_name
                retired_object_keys.append(self._object_key(state))
                events.append(
                    SceneEvent(
                        event_type="disappeared",
                        object_key=self._object_key(state),
                        class_name=class_name,
                        timestamp_s=timestamp_s,
                        message=f"{class_name} 객체가 화면에서 사라졌습니다.",
                    )
                )
                self._delete_state(stable_id)
            elif not state.announced and state.missed_streak >= self.max_missed_frames:
                retired_object_keys.append(self._object_key(state))
                self._delete_state(stable_id)

        return FrameEventUpdate(
            events=tuple(events),
            object_keys=object_keys,
            retired_object_keys=tuple(retired_object_keys),
        )

    def update(
        self,
        detections: list[Detection],
        timestamp_s: float,
        *,
        signal_states: Sequence[SignalState | None] | None = None,
    ) -> list[SceneEvent]:
        """Return events while preserving the original list-returning API."""
        return list(
            self.update_frame(
                detections,
                timestamp_s,
                signal_states=signal_states,
            ).events
        )
