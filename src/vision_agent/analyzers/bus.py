from __future__ import annotations

import math
import re
import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from ..ocr import OcrEngine, OcrResult
from ..signals import ImageArray, SignalStateResult
from ..types import AnalysisResult, Detection


class BusMotionState(str, Enum):
    """Conservative motion states inferred from one stable bus track."""

    APPROACHING = "APPROACHING"
    STOPPED = "STOPPED"
    RECEDING = "RECEDING"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class _GeometryObservation:
    frame_index: int
    area: float
    center_x: float
    center_y: float
    width: float
    height: float
    detection_confidence: float


@dataclass(frozen=True, slots=True)
class _MotionEvidence:
    state: BusMotionState
    confidence: float
    area_change_ratio: float | None = None
    center_x_change_ratio: float | None = None
    center_y_change_ratio: float | None = None
    center_displacement_ratio: float | None = None
    increasing_consistency: float | None = None
    decreasing_consistency: float | None = None


@dataclass(slots=True)
class _BusHistory:
    geometry: deque[_GeometryObservation] = field(default_factory=deque)
    last_frame_index: int | None = None
    unreliable_motion_frames: int = 0
    confirmed_motion: BusMotionState | None = None
    motion_candidate: BusMotionState | None = None
    motion_candidate_frames: int = 0
    motion_candidate_confidences: list[float] = field(default_factory=list)
    motion_confirmed_frames: int = 0
    route_candidate: str | None = None
    route_candidate_frames: int = 0
    route_candidate_confidences: list[float] = field(default_factory=list)
    confirmed_route: str | None = None
    confirmed_route_confidence: float = 0.0
    route_observed_frames: int = 0
    last_route_ocr_frame: int | None = None
    last_route_observed: str | None = None
    last_route_observed_confidence: float = 0.0


_ROUTE_TOKEN = re.compile(
    r"(?<![A-Z0-9])(?:[A-Z]{1,2}\d{1,5}(?:-\d{1,2})?|\d{1,5}(?:-\d{1,2})?)(?![A-Z0-9])",
    flags=re.IGNORECASE,
)


def _positive_integer(name: str, value: int, *, minimum: int = 1) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")


def _unit_interval(name: str, value: float, *, positive: bool = False) -> None:
    lower_is_valid = value > 0.0 if positive else value >= 0.0
    if not math.isfinite(value) or not lower_is_valid or value > 1.0:
        bound = "0 < value <= 1" if positive else "0 <= value <= 1"
        raise ValueError(f"{name} must be finite and satisfy {bound}")


class BusAnalyzer:
    """Stabilize bus motion and route-number OCR for each stable object.

    Motion uses a short, consecutive history of bounding-box area and center
    position. Route OCR is optional and injected; this analyzer never performs a
    network call or invents characters that the OCR engine did not return.
    """

    def __init__(
        self,
        ocr_engine: OcrEngine | None = None,
        *,
        motion_window_frames: int = 3,
        minimum_motion_confirmed_frames: int = 2,
        minimum_ocr_confirmed_frames: int = 3,
        minimum_detection_confidence: float = 0.5,
        minimum_motion_confidence: float = 0.5,
        minimum_ocr_confidence: float = 0.75,
        minimum_area_change_ratio: float = 0.1,
        maximum_stopped_area_change_ratio: float = 0.03,
        maximum_stopped_center_change_ratio: float = 0.08,
        maximum_directional_center_shift_ratio: float = 1.25,
        center_direction_tolerance_ratio: float = 0.12,
        minimum_route_confidence_margin: float = 0.05,
        minimum_direction_consistency: float = 0.65,
        area_jitter_tolerance_ratio: float = 0.02,
        maximum_motion_frame_gap: int = 0,
        route_ocr_interval_frames: int = 1,
        route_ocr_requires_relevant_motion: bool = False,
        route_roi: tuple[float, float, float, float] = (0.08, 0.0, 0.92, 0.55),
    ) -> None:
        _positive_integer("motion_window_frames", motion_window_frames, minimum=2)
        _positive_integer("minimum_motion_confirmed_frames", minimum_motion_confirmed_frames)
        _positive_integer("minimum_ocr_confirmed_frames", minimum_ocr_confirmed_frames)
        _unit_interval("minimum_detection_confidence", minimum_detection_confidence)
        _unit_interval("minimum_motion_confidence", minimum_motion_confidence)
        _unit_interval("minimum_ocr_confidence", minimum_ocr_confidence)
        _unit_interval("minimum_area_change_ratio", minimum_area_change_ratio, positive=True)
        _unit_interval("maximum_stopped_area_change_ratio", maximum_stopped_area_change_ratio)
        _unit_interval("minimum_route_confidence_margin", minimum_route_confidence_margin)
        _unit_interval(
            "minimum_direction_consistency", minimum_direction_consistency, positive=True
        )
        _unit_interval("area_jitter_tolerance_ratio", area_jitter_tolerance_ratio)
        _positive_integer("maximum_motion_frame_gap", maximum_motion_frame_gap, minimum=0)
        _positive_integer("route_ocr_interval_frames", route_ocr_interval_frames)
        if not isinstance(route_ocr_requires_relevant_motion, bool):
            raise ValueError("route_ocr_requires_relevant_motion must be a boolean")
        if maximum_stopped_area_change_ratio >= minimum_area_change_ratio:
            raise ValueError(
                "maximum_stopped_area_change_ratio must be less than minimum_area_change_ratio"
            )
        for name, value in (
            ("maximum_stopped_center_change_ratio", maximum_stopped_center_change_ratio),
            ("maximum_directional_center_shift_ratio", maximum_directional_center_shift_ratio),
            ("center_direction_tolerance_ratio", center_direction_tolerance_ratio),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if len(route_roi) != 4 or not all(math.isfinite(value) for value in route_roi):
            raise ValueError("route_roi must contain four finite normalized values")
        x_start, y_start, x_end, y_end = route_roi
        if not 0.0 <= x_start < x_end <= 1.0 or not 0.0 <= y_start < y_end <= 1.0:
            raise ValueError("route_roi must satisfy 0 <= start < end <= 1")

        self.ocr_engine = ocr_engine
        self.motion_window_frames = motion_window_frames
        self.minimum_motion_confirmed_frames = minimum_motion_confirmed_frames
        self.minimum_ocr_confirmed_frames = minimum_ocr_confirmed_frames
        self.minimum_detection_confidence = minimum_detection_confidence
        self.minimum_motion_confidence = minimum_motion_confidence
        self.minimum_ocr_confidence = minimum_ocr_confidence
        self.minimum_area_change_ratio = minimum_area_change_ratio
        self.maximum_stopped_area_change_ratio = maximum_stopped_area_change_ratio
        self.maximum_stopped_center_change_ratio = maximum_stopped_center_change_ratio
        self.maximum_directional_center_shift_ratio = maximum_directional_center_shift_ratio
        self.center_direction_tolerance_ratio = center_direction_tolerance_ratio
        self.minimum_route_confidence_margin = minimum_route_confidence_margin
        self.minimum_direction_consistency = minimum_direction_consistency
        self.area_jitter_tolerance_ratio = area_jitter_tolerance_ratio
        self.maximum_motion_frame_gap = maximum_motion_frame_gap
        self.route_ocr_interval_frames = route_ocr_interval_frames
        self.route_ocr_requires_relevant_motion = route_ocr_requires_relevant_motion
        self.route_roi = route_roi
        self._history_by_stable_id: dict[str, _BusHistory] = {}

    @staticmethod
    def _geometry(detection: Detection) -> _GeometryObservation | None:
        try:
            left, top, right, bottom = (float(value) for value in detection.xyxy)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (left, top, right, bottom)):
            return None
        width = right - left
        height = bottom - top
        if width <= 0.0 or height <= 0.0:
            return None
        return _GeometryObservation(
            frame_index=detection.frame_index,
            area=width * height,
            center_x=(left + right) / 2.0,
            center_y=(top + bottom) / 2.0,
            width=width,
            height=height,
            detection_confidence=float(detection.confidence),
        )

    @staticmethod
    def _break_candidates(history: _BusHistory) -> None:
        history.motion_candidate = None
        history.motion_candidate_frames = 0
        history.motion_candidate_confidences.clear()
        history.motion_confirmed_frames = 0
        history.route_candidate = None
        history.route_candidate_frames = 0
        history.route_candidate_confidences.clear()
        history.route_observed_frames = 0
        history.last_route_ocr_frame = None
        history.last_route_observed = None
        history.last_route_observed_confidence = 0.0

    def _start_frame(self, history: _BusHistory, frame_index: int) -> bool:
        if history.last_frame_index is None:
            frame_gap = 0
            consecutive = True
        else:
            frame_gap = frame_index - history.last_frame_index - 1
            consecutive = frame_gap == 0
        gap_is_tolerated = 0 <= frame_gap <= self.maximum_motion_frame_gap
        if not consecutive and not gap_is_tolerated:
            history.geometry.clear()
            self._break_candidates(history)
            history.confirmed_motion = None
            history.unreliable_motion_frames = 0
        history.last_frame_index = frame_index
        return consecutive

    def _motion_evidence(self, history: _BusHistory) -> _MotionEvidence:
        observations = tuple(history.geometry)
        if len(observations) < self.motion_window_frames:
            return _MotionEvidence(BusMotionState.UNKNOWN, 0.0)

        first = observations[0]
        endpoint_size = max(1, len(observations) // 3)
        first_group = observations[:endpoint_size]
        last_group = observations[-endpoint_size:]
        mean_width = sum(item.width for item in observations) / len(observations)
        mean_height = sum(item.height for item in observations) / len(observations)
        first_area = statistics.median(item.area for item in first_group)
        last_area = statistics.median(item.area for item in last_group)
        if mean_width <= 0.0 or mean_height <= 0.0 or first_area <= 0.0:
            return _MotionEvidence(BusMotionState.UNKNOWN, 0.0)

        area_change = (last_area - first_area) / first_area
        first_center_x = statistics.median(item.center_x for item in first_group)
        last_center_x = statistics.median(item.center_x for item in last_group)
        first_center_y = statistics.median(item.center_y for item in first_group)
        last_center_y = statistics.median(item.center_y for item in last_group)
        center_x_change = (last_center_x - first_center_x) / mean_width
        center_y_change = (last_center_y - first_center_y) / mean_height
        center_displacement = math.hypot(center_x_change, center_y_change)
        pair_area_changes = [
            (current.area - previous.area) / previous.area
            for previous, current in zip(observations, observations[1:])
        ]
        pair_count = len(pair_area_changes)
        increasing_consistency = (
            sum(change >= -self.area_jitter_tolerance_ratio for change in pair_area_changes)
            / pair_count
        )
        decreasing_consistency = (
            sum(change <= self.area_jitter_tolerance_ratio for change in pair_area_changes)
            / pair_count
        )
        increasing = increasing_consistency >= self.minimum_direction_consistency
        decreasing = decreasing_consistency >= self.minimum_direction_consistency
        mean_detection_confidence = sum(item.detection_confidence for item in observations) / len(
            observations
        )

        state = BusMotionState.UNKNOWN
        strength = 0.0
        center_is_plausible = center_displacement <= self.maximum_directional_center_shift_ratio
        if (
            increasing
            and area_change >= self.minimum_area_change_ratio
            and center_is_plausible
            and center_y_change >= -self.center_direction_tolerance_ratio
        ):
            state = BusMotionState.APPROACHING
            strength = min(
                1.0,
                area_change / (2.0 * self.minimum_area_change_ratio),
            )
        elif (
            decreasing
            and area_change <= -self.minimum_area_change_ratio
            and center_is_plausible
            and center_y_change <= self.center_direction_tolerance_ratio
        ):
            state = BusMotionState.RECEDING
            strength = min(
                1.0,
                abs(area_change) / (2.0 * self.minimum_area_change_ratio),
            )
        else:
            mean_area = sum(item.area for item in observations) / len(observations)
            area_span = (
                max(item.area for item in observations) - min(item.area for item in observations)
            ) / mean_area
            center_span = max(
                math.hypot(
                    (item.center_x - first.center_x) / mean_width,
                    (item.center_y - first.center_y) / mean_height,
                )
                for item in observations
            )
            if (
                area_span <= self.maximum_stopped_area_change_ratio
                and center_span <= self.maximum_stopped_center_change_ratio
            ):
                state = BusMotionState.STOPPED
                area_score = 1.0 - area_span / max(
                    self.maximum_stopped_area_change_ratio,
                    1e-9,
                )
                center_score = 1.0 - center_span / max(
                    self.maximum_stopped_center_change_ratio,
                    1e-9,
                )
                strength = 0.5 + 0.5 * max(0.0, min(area_score, center_score))

        confidence = (
            max(0.0, min(1.0, mean_detection_confidence * (0.5 + 0.5 * strength)))
            if state is not BusMotionState.UNKNOWN
            else 0.0
        )
        return _MotionEvidence(
            state=state,
            confidence=confidence,
            area_change_ratio=area_change,
            center_x_change_ratio=center_x_change,
            center_y_change_ratio=center_y_change,
            center_displacement_ratio=center_displacement,
            increasing_consistency=increasing_consistency,
            decreasing_consistency=decreasing_consistency,
        )

    def _stabilize_motion(
        self,
        history: _BusHistory,
        evidence: _MotionEvidence,
    ) -> tuple[BusMotionState, float, BusMotionState | None, bool]:
        observed = evidence.state
        if observed is BusMotionState.UNKNOWN:
            history.motion_candidate = None
            history.motion_candidate_frames = 0
            history.motion_candidate_confidences.clear()
            history.motion_confirmed_frames = 0
            return BusMotionState.UNKNOWN, 0.0, None, False

        if observed is history.confirmed_motion:
            history.motion_candidate = None
            history.motion_candidate_frames = 0
            history.motion_candidate_confidences.clear()
            history.motion_confirmed_frames += 1
            return observed, evidence.confidence, None, False

        if observed is history.motion_candidate:
            history.motion_candidate_frames += 1
            history.motion_candidate_confidences.append(evidence.confidence)
        else:
            history.motion_candidate = observed
            history.motion_candidate_frames = 1
            history.motion_candidate_confidences = [evidence.confidence]

        if history.motion_candidate_frames < self.minimum_motion_confirmed_frames:
            history.motion_confirmed_frames = 0
            return BusMotionState.UNKNOWN, 0.0, None, False

        previous_state = history.confirmed_motion
        confidence = sum(history.motion_candidate_confidences) / len(
            history.motion_candidate_confidences
        )
        history.confirmed_motion = observed
        history.motion_confirmed_frames = history.motion_candidate_frames
        history.motion_candidate = None
        history.motion_candidate_frames = 0
        history.motion_candidate_confidences.clear()
        return observed, confidence, previous_state, previous_state is not None

    def _route_display_crop(
        self,
        crop: ImageArray | None,
    ) -> tuple[ImageArray | None, tuple[int, int, int, int] | None]:
        if crop is None or crop.ndim < 2 or crop.size == 0:
            return None, None
        height, width = crop.shape[:2]
        if height < 2 or width < 2:
            return None, None
        x_start, y_start, x_end, y_end = self.route_roi
        left = max(0, min(width, math.floor(width * x_start)))
        top = max(0, min(height, math.floor(height * y_start)))
        right = max(0, min(width, math.ceil(width * x_end - 1e-9)))
        bottom = max(0, min(height, math.ceil(height * y_end - 1e-9)))
        if right <= left or bottom <= top:
            return None, None
        roi = crop[top:bottom, left:right]
        if roi.size == 0:
            return None, None
        return roi, (left, top, right, bottom)

    def _route_observation(
        self,
        result: OcrResult,
    ) -> tuple[str | None, float]:
        candidates: dict[str, float] = {}
        for line in result.lines:
            confidence = float(line.confidence)
            if not math.isfinite(confidence) or confidence < self.minimum_ocr_confidence:
                continue
            text = line.text.strip().upper()
            if any(marker in text for marker in ("?", "*", "�")):
                continue
            matches = {match.group(0).upper() for match in _ROUTE_TOKEN.finditer(text)}
            for route in matches:
                candidates[route] = max(confidence, candidates.get(route, 0.0))
        if not candidates:
            return None, 0.0
        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) > 1:
            confidence_margin = ranked[0][1] - ranked[1][1]
            if confidence_margin < self.minimum_route_confidence_margin:
                return None, 0.0
        return ranked[0]

    def _observe_route(
        self,
        history: _BusHistory,
        route: str | None,
        confidence: float,
    ) -> bool:
        if route is None:
            history.route_candidate = None
            history.route_candidate_frames = 0
            history.route_candidate_confidences.clear()
            history.route_observed_frames = 0
            return False

        if route == history.route_candidate:
            history.route_candidate_frames += 1
            history.route_candidate_confidences.append(confidence)
        else:
            history.route_candidate = route
            history.route_candidate_frames = 1
            history.route_candidate_confidences = [confidence]

        history.route_observed_frames = history.route_candidate_frames
        if history.route_candidate_frames < self.minimum_ocr_confirmed_frames:
            return False

        changed = history.confirmed_route is not None and history.confirmed_route != route
        history.confirmed_route = route
        history.confirmed_route_confidence = sum(history.route_candidate_confidences) / len(
            history.route_candidate_confidences
        )
        return changed

    def analyze(
        self,
        detection: Detection,
        *,
        stable_id: str,
        crop: ImageArray | None = None,
        precomputed_signal_result: SignalStateResult | None = None,
    ) -> AnalysisResult:
        del precomputed_signal_result  # Part of the common analyzer interface.
        if not stable_id.strip():
            raise ValueError("stable_id must not be empty")

        history = self._history_by_stable_id.setdefault(stable_id, _BusHistory())
        observations_are_consecutive = self._start_frame(history, detection.frame_index)
        uncertainty_reasons: list[str] = []

        geometry = self._geometry(detection)
        detection_is_reliable = (
            math.isfinite(float(detection.confidence))
            and detection.confidence >= self.minimum_detection_confidence
        )
        if geometry is None:
            history.geometry.clear()
            self._break_candidates(history)
            history.confirmed_motion = None
            history.unreliable_motion_frames = 0
            uncertainty_reasons.append("invalid_bus_bbox")
        elif not detection_is_reliable:
            history.unreliable_motion_frames += 1
            if history.unreliable_motion_frames > self.maximum_motion_frame_gap:
                history.geometry.clear()
                self._break_candidates(history)
                history.confirmed_motion = None
            uncertainty_reasons.append("low_detection_confidence")
        else:
            history.unreliable_motion_frames = 0
            history.geometry.append(geometry)
            while len(history.geometry) > self.motion_window_frames:
                history.geometry.popleft()

        if geometry is None or not detection_is_reliable:
            motion_evidence = _MotionEvidence(BusMotionState.UNKNOWN, 0.0)
            state = BusMotionState.UNKNOWN
            motion_confidence = 0.0
            previous_state = None
            motion_changed = False
        else:
            motion_evidence = self._motion_evidence(history)
            state, motion_confidence, previous_state, motion_changed = self._stabilize_motion(
                history,
                motion_evidence,
            )
        if state is BusMotionState.UNKNOWN and not uncertainty_reasons:
            uncertainty_reasons.append("insufficient_or_ambiguous_motion")

        has_confident_motion = (
            state is not BusMotionState.UNKNOWN
            and motion_confidence >= self.minimum_motion_confidence
        )
        route_motion_is_relevant = not self.route_ocr_requires_relevant_motion or (
            has_confident_motion and state in {BusMotionState.APPROACHING, BusMotionState.STOPPED}
        )
        route_ocr_is_due = (
            history.last_route_ocr_frame is None
            or detection.frame_index - history.last_route_ocr_frame
            >= self.route_ocr_interval_frames
        )
        observed_route: str | None = None
        observed_route_confidence = 0.0
        route_observation_is_new = False
        route_roi_xyxy: tuple[int, int, int, int] | None = None
        ocr_engine_name: str | None = None
        ocr_error: str | None = None
        ocr_was_run = False
        if not detection_is_reliable or geometry is None:
            self._observe_route(history, None, 0.0)
            history.last_route_observed = None
            history.last_route_observed_confidence = 0.0
        elif self.ocr_engine is None:
            self._observe_route(history, None, 0.0)
            history.last_route_observed = None
            history.last_route_observed_confidence = 0.0
            uncertainty_reasons.append("ocr_engine_unavailable")
        elif not route_motion_is_relevant:
            self._observe_route(history, None, 0.0)
            history.last_route_observed = None
            history.last_route_observed_confidence = 0.0
            uncertainty_reasons.append("route_ocr_waiting_for_relevant_motion")
        elif not route_ocr_is_due:
            observed_route = history.last_route_observed
            observed_route_confidence = history.last_route_observed_confidence
            ocr_engine_name = "deferred"
        else:
            history.last_route_ocr_frame = detection.frame_index
            ocr_was_run = True
            route_crop, route_roi_xyxy = self._route_display_crop(crop)
            if route_crop is None:
                self._observe_route(history, None, 0.0)
                history.last_route_observed = None
                history.last_route_observed_confidence = 0.0
                uncertainty_reasons.append("bus_crop_unavailable")
            else:
                try:
                    ocr_result = self.ocr_engine.recognize(route_crop)
                except Exception as error:  # OCR plugins must not stop the video loop.
                    self._observe_route(history, None, 0.0)
                    history.last_route_observed = None
                    history.last_route_observed_confidence = 0.0
                    ocr_error = type(error).__name__
                    uncertainty_reasons.append("ocr_engine_error")
                else:
                    ocr_engine_name = ocr_result.engine_name
                    ocr_error = ocr_result.error
                    if not ocr_result.is_available:
                        self._observe_route(history, None, 0.0)
                        history.last_route_observed = None
                        history.last_route_observed_confidence = 0.0
                        uncertainty_reasons.append("ocr_engine_unavailable")
                    else:
                        observed_route, observed_route_confidence = self._route_observation(
                            ocr_result
                        )
                        if observed_route is None:
                            self._observe_route(history, None, 0.0)
                            history.last_route_observed = None
                            history.last_route_observed_confidence = 0.0
                            uncertainty_reasons.append("route_ocr_uncertain")
                        else:
                            history.last_route_observed = observed_route
                            history.last_route_observed_confidence = observed_route_confidence
                            route_observation_is_new = True

        route_changed = (
            self._observe_route(
                history,
                observed_route,
                min(float(detection.confidence), observed_route_confidence),
            )
            if route_observation_is_new and observed_route is not None
            else False
        )

        route_is_current = observed_route is not None and observed_route == history.confirmed_route
        route_number = history.confirmed_route if route_is_current else None
        route_confidence = history.confirmed_route_confidence if route_number else 0.0
        if state is not BusMotionState.UNKNOWN and not has_confident_motion:
            uncertainty_reasons.append("motion_confidence_below_threshold")
        has_confirmed_route = route_number is not None
        is_uncertain = not has_confident_motion and not has_confirmed_route
        confidence = max(
            motion_confidence if has_confident_motion else 0.0,
            route_confidence if has_confirmed_route else 0.0,
        )

        attributes: dict[str, object] = {
            "class_name": detection.class_name,
            "detection_confidence": detection.confidence,
            "route_number": route_number,
            "route_confidence": route_confidence,
            "route_is_current": route_is_current,
            "last_confirmed_route": history.confirmed_route,
            "last_confirmed_route_confidence": history.confirmed_route_confidence,
            "ocr_confirmed_frames": history.route_observed_frames,
            "ocr_observed_route": observed_route,
            "ocr_observed_confidence": observed_route_confidence,
            "ocr_candidate_route": history.route_candidate,
            "ocr_candidate_frames": history.route_candidate_frames,
            "route_changed": route_changed,
            "route_roi_xyxy": route_roi_xyxy,
            "ocr_engine_name": ocr_engine_name,
            "ocr_error": ocr_error,
            "ocr_was_run": ocr_was_run,
            "ocr_interval_frames": self.route_ocr_interval_frames,
            "route_motion_is_relevant": route_motion_is_relevant,
            "motion_observed_state": motion_evidence.state.value,
            "motion_stabilized_state": state.value,
            "motion_confidence": motion_confidence,
            "motion_confirmed_frames": history.motion_confirmed_frames,
            "motion_candidate_state": (
                history.motion_candidate.value if history.motion_candidate is not None else None
            ),
            "motion_candidate_frames": history.motion_candidate_frames,
            "bbox_area": geometry.area if geometry is not None else None,
            "bbox_center": (
                (geometry.center_x, geometry.center_y) if geometry is not None else None
            ),
            "area_change_ratio": motion_evidence.area_change_ratio,
            "center_x_change_ratio": motion_evidence.center_x_change_ratio,
            "center_y_change_ratio": motion_evidence.center_y_change_ratio,
            "center_displacement_ratio": motion_evidence.center_displacement_ratio,
            "increasing_consistency": motion_evidence.increasing_consistency,
            "decreasing_consistency": motion_evidence.decreasing_consistency,
            "motion_window_observations": len(history.geometry),
            "unreliable_motion_frames": history.unreliable_motion_frames,
            "observations_are_consecutive": observations_are_consecutive,
            "previous_state": previous_state.value if previous_state is not None else None,
            "changed": motion_changed,
            "motion_is_uncertain": not has_confident_motion,
            "uncertainty_reasons": tuple(dict.fromkeys(uncertainty_reasons)),
        }
        if is_uncertain and uncertainty_reasons:
            attributes["reason"] = uncertainty_reasons[0]

        return AnalysisResult(
            object_type="bus",
            stable_id=stable_id,
            state=(state.value if has_confident_motion else BusMotionState.UNKNOWN.value),
            confidence=confidence,
            attributes=attributes,
            is_uncertain=is_uncertain,
        )

    def reset(self, stable_id: str | None = None) -> None:
        """Forget one bus history, or all bus histories when no ID is supplied."""
        if stable_id is None:
            self._history_by_stable_id.clear()
        else:
            self._history_by_stable_id.pop(stable_id, None)
