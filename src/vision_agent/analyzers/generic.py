from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..signals import ImageArray, SignalStateResult
from ..types import AnalysisResult, Detection
from ..vlm import VisionLanguageModel
from .base import normalize_object_type


@dataclass(slots=True)
class _DescriptionHistory:
    candidate: str | None = None
    candidate_results: int = 0
    confirmed: str | None = None
    confirmed_results: int = 0
    confidence: float = 0.0
    last_inference_frame: int | None = None
    last_observation_frame: int | None = None
    seen_streak: int = 0


class GenericVisionAnalyzer:
    """Temporally stabilize a crop-scoped VLM fallback for unknown object types."""

    DEFAULT_PROMPT = (
        "이 객체 crop에서 직접 보이는 사실만 한국어 한 문장으로 설명하세요. "
        "글자, 신호 상태, 장소명은 추측하지 말고 불명확하면 알 수 없다고 답하세요."
    )

    def __init__(
        self,
        model: VisionLanguageModel | None = None,
        *,
        minimum_confirmed_results: int = 2,
        minimum_detection_confidence: float = 0.5,
        minimum_confidence: float = 0.5,
        inference_interval_frames: int = 15,
        similarity_threshold: float = 1.0,
        prompt: str = DEFAULT_PROMPT,
        allowed_object_types: Collection[str] | None = None,
        minimum_seen_frames_before_inference: int = 1,
    ) -> None:
        if minimum_confirmed_results < 1:
            raise ValueError("minimum_confirmed_results must be at least 1")
        for name, value in (
            ("minimum_detection_confidence", minimum_detection_confidence),
            ("minimum_confidence", minimum_confidence),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if inference_interval_frames < 1:
            raise ValueError("inference_interval_frames must be at least 1")
        if not math.isfinite(similarity_threshold) or not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if not prompt.strip():
            raise ValueError("prompt must not be blank")
        if (
            not isinstance(minimum_seen_frames_before_inference, int)
            or isinstance(minimum_seen_frames_before_inference, bool)
            or minimum_seen_frames_before_inference < 1
        ):
            raise ValueError("minimum_seen_frames_before_inference must be a positive integer")
        self.model = model
        self.minimum_confirmed_results = minimum_confirmed_results
        self.minimum_detection_confidence = minimum_detection_confidence
        self.minimum_confidence = minimum_confidence
        self.inference_interval_frames = inference_interval_frames
        self.similarity_threshold = similarity_threshold
        self.prompt = prompt.strip()
        self.allowed_object_types = (
            None
            if allowed_object_types is None
            else frozenset(normalize_object_type(value) for value in allowed_object_types)
        )
        self.minimum_seen_frames_before_inference = minimum_seen_frames_before_inference
        self._history_by_stable_id: dict[str, _DescriptionHistory] = {}

    @staticmethod
    def _normalize_description(value: str) -> str:
        return " ".join(value.strip().split())[:300]

    def _is_similar(self, first: str, second: str) -> bool:
        return SequenceMatcher(None, first.casefold(), second.casefold()).ratio() >= (
            self.similarity_threshold
        )

    def _result(
        self,
        detection: Detection,
        stable_id: str,
        history: _DescriptionHistory,
        *,
        observed_description: str | None,
        description_changed: bool,
        inference_performed: bool,
        reason: str | None = None,
        is_uncertain: bool | None = None,
    ) -> AnalysisResult:
        confirmed = history.confirmed
        uncertain = confirmed is None if is_uncertain is None else is_uncertain
        attributes: dict[str, object] = {
            "class_name": detection.class_name,
            "detection_confidence": detection.confidence,
            "description": confirmed,
            "observed_description": observed_description,
            "description_changed": description_changed,
            "confirmed_results": history.confirmed_results,
            "candidate_results": history.candidate_results,
            "seen_streak": history.seen_streak,
            "vlm_confidence": history.confidence,
            "inference_performed": inference_performed,
            "backend": type(self.model).__name__ if self.model is not None else None,
        }
        if reason is not None:
            attributes["reason"] = reason
        return AnalysisResult(
            object_type=normalize_object_type(detection.class_name),
            stable_id=stable_id,
            state="DESCRIBED" if confirmed is not None else "UNKNOWN",
            confidence=(
                min(history.confidence, float(detection.confidence))
                if confirmed is not None and not uncertain
                else 0.0
            ),
            attributes=attributes,
            is_uncertain=uncertain,
        )

    def analyze(
        self,
        detection: Detection,
        *,
        stable_id: str,
        crop: ImageArray | None = None,
        precomputed_signal_result: SignalStateResult | None = None,
    ) -> AnalysisResult:
        history = self._history_by_stable_id.setdefault(stable_id, _DescriptionHistory())
        if (
            history.last_observation_frame is not None
            and detection.frame_index != history.last_observation_frame + 1
        ):
            history.candidate = None
            history.candidate_results = 0
            history.last_inference_frame = None
            history.seen_streak = 0
        try:
            detection_confidence = float(detection.confidence)
        except (TypeError, ValueError):
            detection_confidence = 0.0
        if (
            not math.isfinite(detection_confidence)
            or detection_confidence < self.minimum_detection_confidence
        ):
            history.candidate = None
            history.candidate_results = 0
            history.last_inference_frame = None
            history.last_observation_frame = detection.frame_index
            history.seen_streak = 0
            return self._result(
                detection,
                stable_id,
                history,
                observed_description=None,
                description_changed=False,
                inference_performed=False,
                reason="low_detection_confidence",
                is_uncertain=True,
            )
        history.seen_streak += 1
        history.last_observation_frame = detection.frame_index
        object_type = normalize_object_type(detection.class_name)
        if self.model is None:
            return self._result(
                detection,
                stable_id,
                history,
                observed_description=None,
                description_changed=False,
                inference_performed=False,
                reason="generic_vision_disabled",
                is_uncertain=True,
            )
        if (
            self.allowed_object_types is not None
            and object_type not in self.allowed_object_types
        ):
            return self._result(
                detection,
                stable_id,
                history,
                observed_description=None,
                description_changed=False,
                inference_performed=False,
                reason="generic_vision_class_not_allowed",
                is_uncertain=True,
            )
        if history.seen_streak < self.minimum_seen_frames_before_inference:
            return self._result(
                detection,
                stable_id,
                history,
                observed_description=None,
                description_changed=False,
                inference_performed=False,
                reason="generic_object_not_stable",
                is_uncertain=True,
            )
        if crop is None:
            return self._result(
                detection,
                stable_id,
                history,
                observed_description=None,
                description_changed=False,
                inference_performed=False,
                reason="object_crop_unavailable",
                is_uncertain=True,
            )

        if (
            history.last_inference_frame is not None
            and detection.frame_index - history.last_inference_frame
            < self.inference_interval_frames
        ):
            return self._result(
                detection,
                stable_id,
                history,
                observed_description=None,
                description_changed=False,
                inference_performed=False,
            )

        history.last_inference_frame = detection.frame_index
        try:
            vlm_result = self.model.describe(crop, self.prompt)
        except Exception as exc:
            history.candidate = None
            history.candidate_results = 0
            return self._result(
                detection,
                stable_id,
                history,
                observed_description=None,
                description_changed=False,
                inference_performed=True,
                reason=f"vlm_backend_error:{type(exc).__name__}",
                is_uncertain=True,
            )
        observed = (
            self._normalize_description(vlm_result.description)
            if vlm_result.description is not None
            else None
        )
        confidence = vlm_result.confidence or 0.0
        if observed is None or confidence < self.minimum_confidence:
            history.candidate = None
            history.candidate_results = 0
            return self._result(
                detection,
                stable_id,
                history,
                observed_description=observed,
                description_changed=False,
                inference_performed=True,
                reason=vlm_result.error or "vlm_result_below_threshold",
                is_uncertain=True,
            )

        history.confidence = confidence
        if history.confirmed is not None and self._is_similar(observed, history.confirmed):
            history.confirmed_results += 1
            history.candidate = None
            history.candidate_results = 0
            return self._result(
                detection,
                stable_id,
                history,
                observed_description=observed,
                description_changed=False,
                inference_performed=True,
                is_uncertain=False,
            )

        if history.candidate is not None and self._is_similar(observed, history.candidate):
            history.candidate_results += 1
        else:
            history.candidate = observed
            history.candidate_results = 1

        if history.candidate_results < self.minimum_confirmed_results:
            return self._result(
                detection,
                stable_id,
                history,
                observed_description=observed,
                description_changed=False,
                inference_performed=True,
                is_uncertain=True,
            )

        previous = history.confirmed
        history.confirmed = observed
        history.confirmed_results = history.candidate_results
        history.candidate = None
        history.candidate_results = 0
        return self._result(
            detection,
            stable_id,
            history,
            observed_description=observed,
            description_changed=previous is not None and previous != observed,
            inference_performed=True,
            is_uncertain=False,
        )

    def reset(self, stable_id: str | None = None) -> None:
        if stable_id is None:
            self._history_by_stable_id.clear()
        else:
            self._history_by_stable_id.pop(stable_id, None)
