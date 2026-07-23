from __future__ import annotations

import math
from dataclasses import dataclass

from ..object_types import AnalyzerKind, normalize_object_type, object_class_spec
from ..signals import (
    HsvSignalStateClassifier,
    ImageArray,
    SignalStateClassifier,
    SignalStateResult,
)
from ..types import AnalysisResult, Detection, SignalState


def _bounded_confidence(value: object) -> float:
    """Normalize external evidence to a conservative unit-interval score."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return 0.0
    return confidence


@dataclass(slots=True)
class _SignalHistory:
    confirmed_state: SignalState | None = None
    candidate_state: SignalState | None = None
    candidate_frames: int = 0
    confirmed_frames: int = 0
    last_frame_index: int | None = None


class TrafficLightAnalyzer:
    """Stabilize conservative HSV observations independently for each stable ID."""

    def __init__(
        self,
        classifier: SignalStateClassifier | None = None,
        *,
        minimum_confirmed_frames: int = 3,
        minimum_detection_confidence: float = 0.2,
        enabled: bool = True,
    ) -> None:
        if (
            not isinstance(minimum_confirmed_frames, int)
            or isinstance(minimum_confirmed_frames, bool)
            or minimum_confirmed_frames < 1
        ):
            raise ValueError("minimum_confirmed_frames must be a positive integer")
        if (
            not math.isfinite(minimum_detection_confidence)
            or not 0.0 <= minimum_detection_confidence <= 1.0
        ):
            raise ValueError("minimum_detection_confidence must be finite and between 0 and 1")
        self.minimum_confirmed_frames = minimum_confirmed_frames
        self.minimum_detection_confidence = minimum_detection_confidence
        self.enabled = enabled
        self.classifier = (
            (classifier if classifier is not None else HsvSignalStateClassifier())
            if enabled
            else None
        )
        self._history_by_stable_id: dict[str, _SignalHistory] = {}

    @staticmethod
    def _as_signal_state(value: object) -> SignalState:
        try:
            return SignalState(value)
        except (TypeError, ValueError):
            return SignalState.UNKNOWN

    @staticmethod
    def _evidence(
        result: SignalStateResult | None,
    ) -> tuple[SignalState, float, float, float, float]:
        if result is None:
            return SignalState.UNKNOWN, 0.0, 0.0, 0.0, 0.0
        return (
            TrafficLightAnalyzer._as_signal_state(result.state),
            _bounded_confidence(result.confidence),
            float(result.red_ratio),
            float(result.green_ratio),
            float(getattr(result, "yellow_ratio", 0.0)),
        )

    def _observe(
        self,
        history: _SignalHistory,
        observed_state: SignalState,
    ) -> tuple[SignalState, SignalState | None, bool, bool]:
        if observed_state is SignalState.UNKNOWN:
            history.candidate_state = None
            history.candidate_frames = 0
            history.confirmed_frames = 0
            return SignalState.UNKNOWN, None, False, True

        if history.confirmed_state is None:
            if history.candidate_state is observed_state:
                history.candidate_frames += 1
            else:
                history.candidate_state = observed_state
                history.candidate_frames = 1
            if history.candidate_frames < self.minimum_confirmed_frames:
                return SignalState.UNKNOWN, None, False, True

            history.confirmed_state = observed_state
            history.confirmed_frames = history.candidate_frames
            history.candidate_state = None
            history.candidate_frames = 0
            return observed_state, None, False, False

        if observed_state is history.confirmed_state:
            history.candidate_state = None
            history.candidate_frames = 0
            history.confirmed_frames += 1
            return history.confirmed_state, None, False, False

        history.confirmed_frames = 0
        if history.candidate_state is observed_state:
            history.candidate_frames += 1
        else:
            history.candidate_state = observed_state
            history.candidate_frames = 1
        if history.candidate_frames < self.minimum_confirmed_frames:
            return history.confirmed_state, None, False, True

        previous_state = history.confirmed_state
        history.confirmed_state = observed_state
        history.confirmed_frames = history.candidate_frames
        history.candidate_state = None
        history.candidate_frames = 0
        return observed_state, previous_state, True, False

    @staticmethod
    def _start_frame(history: _SignalHistory, frame_index: int) -> bool:
        """Break candidate streaks unless this observation is the next video frame."""
        is_consecutive = (
            history.last_frame_index is None or frame_index == history.last_frame_index + 1
        )
        if not is_consecutive:
            history.candidate_state = None
            history.candidate_frames = 0
            history.confirmed_frames = 0
        history.last_frame_index = frame_index
        return is_consecutive

    def analyze(
        self,
        detection: Detection,
        *,
        stable_id: str,
        crop: ImageArray | None = None,
        precomputed_signal_result: SignalStateResult | None = None,
    ) -> AnalysisResult:
        result = precomputed_signal_result
        disabled_reason: str | None = None
        detection_confidence = _bounded_confidence(detection.confidence)
        detection_is_reliable = detection_confidence >= self.minimum_detection_confidence
        if not self.enabled:
            result = None
            disabled_reason = "signal_state_analysis_disabled"
        elif not detection_is_reliable:
            result = None
            disabled_reason = "low_detection_confidence"
        elif result is None and crop is None:
            disabled_reason = "signal_crop_unavailable"
        elif result is None:
            if self.classifier is None:  # Defensive: enabled analyzers always have one.
                disabled_reason = "signal_classifier_unavailable"
            else:
                result = self.classifier.classify(crop)

        observed, evidence_confidence, red_ratio, green_ratio, yellow_ratio = self._evidence(result)
        history = self._history_by_stable_id.setdefault(stable_id, _SignalHistory())
        observations_are_consecutive = self._start_frame(history, detection.frame_index)
        state, previous_state, changed, is_uncertain = self._observe(
            history,
            observed,
        )
        detected_object_type = normalize_object_type(detection.class_name)
        class_spec = object_class_spec(detected_object_type)
        object_type = (
            detected_object_type
            if class_spec.analyzer is AnalyzerKind.TRAFFIC_LIGHT
            else "traffic_light"
        )
        signal_type_is_confirmed = class_spec.signal_type is not None and detection_is_reliable
        attributes: dict[str, object] = {
            "detection_confidence": detection_confidence,
            "minimum_detection_confidence": self.minimum_detection_confidence,
            "observed_state": observed.value,
            # Keep detector and HSV evidence separate. The HSV score measures
            # relative color evidence and is not treated as a calibrated
            # probability or allowed to override weak detector evidence.
            "signal_state_confidence": evidence_confidence,
            "signal_evidence_confidence": evidence_confidence,
            "red_ratio": red_ratio,
            "green_ratio": green_ratio,
            "yellow_ratio": yellow_ratio,
            "confirmed_frames": history.confirmed_frames,
            "candidate_state": (
                history.candidate_state.value if history.candidate_state is not None else None
            ),
            "candidate_frames": history.candidate_frames,
            "observations_are_consecutive": observations_are_consecutive,
            "previous_state": previous_state.value if previous_state is not None else None,
            "changed": changed,
            "class_name": detection.class_name,
            "signal_type": class_spec.signal_type if signal_type_is_confirmed else "UNKNOWN",
            "signal_type_is_uncertain": not signal_type_is_confirmed,
        }
        if disabled_reason is not None:
            attributes["reason"] = disabled_reason

        confidence = min(detection_confidence, evidence_confidence) if not is_uncertain else 0.0
        attributes["combined_confidence"] = confidence
        return AnalysisResult(
            object_type=object_type,
            stable_id=stable_id,
            state=state.value,
            confidence=confidence,
            attributes=attributes,
            is_uncertain=is_uncertain,
        )

    def reset(self, stable_id: str | None = None) -> None:
        """Forget one signal's history, or all histories when no ID is supplied."""
        if stable_id is None:
            self._history_by_stable_id.clear()
        else:
            self._history_by_stable_id.pop(stable_id, None)
