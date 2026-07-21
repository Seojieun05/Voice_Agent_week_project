from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..ocr import OcrEngine, OcrLine, OcrResult, RapidOcrEngine
from ..signals import ImageArray, SignalStateResult
from ..types import AnalysisResult, Detection
from .base import normalize_object_type


@dataclass(slots=True)
class _TextHistory:
    confirmed_text: str | None = None
    confirmed_confidence: float = 0.0
    candidate_text: str | None = None
    candidate_fingerprint: str | None = None
    candidate_frames: int = 0
    candidate_confidence_sum: float = 0.0
    candidate_best_confidence: float = 0.0
    last_frame_index: int | None = None


def _text_fingerprint(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _joined_text(lines: tuple[OcrLine, ...]) -> str:
    return " ".join(line.text.strip() for line in lines if line.text.strip())


def _safe_text_match(first: str, second: str, threshold: float) -> bool:
    if first == second:
        return True
    if any(character.isdigit() for character in first + second):
        return False
    return SequenceMatcher(None, first, second).ratio() >= threshold


class TextObjectAnalyzer:
    """Confirm local OCR text only after consecutive matching observations."""

    def __init__(
        self,
        ocr_engine: OcrEngine | None = None,
        *,
        minimum_confirmed_frames: int = 3,
        minimum_detection_confidence: float = 0.5,
        minimum_ocr_confidence: float = 0.6,
        minimum_text_similarity: float = 1.0,
        minimum_crop_width: int = 32,
        minimum_crop_height: int = 16,
        minimum_crop_pixels: int = 512,
        minimum_text_width: int = 8,
        minimum_text_height: int = 8,
        minimum_text_pixels: int = 64,
    ) -> None:
        if (
            not isinstance(minimum_confirmed_frames, int)
            or isinstance(minimum_confirmed_frames, bool)
            or minimum_confirmed_frames < 1
        ):
            raise ValueError("minimum_confirmed_frames must be a positive integer")
        for name, value in (
            ("minimum_detection_confidence", minimum_detection_confidence),
            ("minimum_ocr_confidence", minimum_ocr_confidence),
            ("minimum_text_similarity", minimum_text_similarity),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        for name, value in (
            ("minimum_crop_width", minimum_crop_width),
            ("minimum_crop_height", minimum_crop_height),
            ("minimum_crop_pixels", minimum_crop_pixels),
            ("minimum_text_width", minimum_text_width),
            ("minimum_text_height", minimum_text_height),
            ("minimum_text_pixels", minimum_text_pixels),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        self.ocr_engine = ocr_engine if ocr_engine is not None else RapidOcrEngine()
        self.minimum_confirmed_frames = minimum_confirmed_frames
        self.minimum_detection_confidence = minimum_detection_confidence
        self.minimum_ocr_confidence = minimum_ocr_confidence
        self.minimum_text_similarity = minimum_text_similarity
        self.minimum_crop_width = minimum_crop_width
        self.minimum_crop_height = minimum_crop_height
        self.minimum_crop_pixels = minimum_crop_pixels
        self.minimum_text_width = minimum_text_width
        self.minimum_text_height = minimum_text_height
        self.minimum_text_pixels = minimum_text_pixels
        self._history_by_stable_id: dict[str, _TextHistory] = {}

    def _text_region_is_large_enough(
        self,
        line: OcrLine,
        *,
        crop_width: int,
        crop_height: int,
    ) -> bool:
        if line.bbox is None:
            return False
        left, top, right, bottom = line.bbox
        visible_width = max(0, min(crop_width, right) - max(0, left))
        visible_height = max(0, min(crop_height, bottom) - max(0, top))
        return (
            visible_width >= self.minimum_text_width
            and visible_height >= self.minimum_text_height
            and visible_width * visible_height >= self.minimum_text_pixels
        )

    @staticmethod
    def _clear_candidate(history: _TextHistory) -> None:
        history.candidate_text = None
        history.candidate_fingerprint = None
        history.candidate_frames = 0
        history.candidate_confidence_sum = 0.0
        history.candidate_best_confidence = 0.0

    def _start_frame(self, history: _TextHistory, frame_index: int) -> bool:
        is_consecutive = (
            history.last_frame_index is None or frame_index == history.last_frame_index + 1
        )
        if not is_consecutive:
            self._clear_candidate(history)
        history.last_frame_index = frame_index
        return is_consecutive

    def _uncertain_result(
        self,
        detection: Detection,
        stable_id: str,
        history: _TextHistory,
        *,
        reason: str,
        observations_are_consecutive: bool,
        observed_text: str | None = None,
        ocr_confidence: float = 0.0,
        ocr_result: OcrResult | None = None,
    ) -> AnalysisResult:
        return AnalysisResult(
            object_type=normalize_object_type(detection.class_name),
            stable_id=stable_id,
            state=None,
            confidence=0.0,
            attributes={
                "class_name": detection.class_name,
                "detection_confidence": detection.confidence,
                "text": history.confirmed_text,
                "observed_text": observed_text,
                "ocr_confidence": ocr_confidence,
                "confirmed_text_confidence": history.confirmed_confidence,
                "ocr_confirmed_frames": 0,
                "ocr_candidate_frames": history.candidate_frames,
                "observations_are_consecutive": observations_are_consecutive,
                "text_changed": False,
                "previous_text": None,
                "ocr_engine": ocr_result.engine_name if ocr_result is not None else None,
                "ocr_engine_available": (
                    ocr_result.is_available if ocr_result is not None else None
                ),
                "ocr_error": ocr_result.error if ocr_result is not None else None,
                "reason": reason,
            },
            is_uncertain=True,
        )

    def _observe_text(
        self,
        history: _TextHistory,
        text: str,
        confidence: float,
    ) -> tuple[bool, str | None]:
        fingerprint = _text_fingerprint(text)
        candidate_matches = history.candidate_fingerprint is not None and _safe_text_match(
            history.candidate_fingerprint,
            fingerprint,
            self.minimum_text_similarity,
        )
        if candidate_matches:
            history.candidate_frames += 1
            history.candidate_confidence_sum += confidence
            if confidence > history.candidate_best_confidence:
                history.candidate_text = text
                history.candidate_best_confidence = confidence
        else:
            history.candidate_text = text
            history.candidate_fingerprint = fingerprint
            history.candidate_frames = 1
            history.candidate_confidence_sum = confidence
            history.candidate_best_confidence = confidence

        if history.candidate_frames < self.minimum_confirmed_frames:
            return False, None

        confirmed_text = history.candidate_text or text
        confirmed_confidence = history.candidate_confidence_sum / history.candidate_frames
        previous_text = history.confirmed_text
        previous_fingerprint = (
            _text_fingerprint(previous_text) if previous_text is not None else None
        )
        confirmed_fingerprint = _text_fingerprint(confirmed_text)
        text_changed = False
        if previous_fingerprint is not None:
            text_changed = not _safe_text_match(
                previous_fingerprint,
                confirmed_fingerprint,
                self.minimum_text_similarity,
            )
            if not text_changed:
                confirmed_text = previous_text
        history.confirmed_text = confirmed_text
        history.confirmed_confidence = confirmed_confidence
        return True, previous_text if text_changed else None

    def analyze(
        self,
        detection: Detection,
        *,
        stable_id: str,
        crop: ImageArray | None = None,
        precomputed_signal_result: SignalStateResult | None = None,
    ) -> AnalysisResult:
        history = self._history_by_stable_id.setdefault(stable_id, _TextHistory())
        observations_are_consecutive = self._start_frame(history, detection.frame_index)

        try:
            detection_confidence = float(detection.confidence)
        except (TypeError, ValueError):
            detection_confidence = 0.0
        if (
            not math.isfinite(detection_confidence)
            or detection_confidence < self.minimum_detection_confidence
        ):
            self._clear_candidate(history)
            return self._uncertain_result(
                detection,
                stable_id,
                history,
                reason="low_detection_confidence",
                observations_are_consecutive=observations_are_consecutive,
            )

        if crop is None or crop.ndim not in {2, 3} or crop.size == 0:
            self._clear_candidate(history)
            return self._uncertain_result(
                detection,
                stable_id,
                history,
                reason="text_crop_unavailable",
                observations_are_consecutive=observations_are_consecutive,
            )

        crop_height, crop_width = crop.shape[:2]
        if (
            crop_width < self.minimum_crop_width
            or crop_height < self.minimum_crop_height
            or crop_width * crop_height < self.minimum_crop_pixels
        ):
            self._clear_candidate(history)
            return self._uncertain_result(
                detection,
                stable_id,
                history,
                reason="text_crop_too_small",
                observations_are_consecutive=observations_are_consecutive,
            )

        try:
            ocr_result = self.ocr_engine.recognize(crop)
        except Exception as exc:
            self._clear_candidate(history)
            fallback_result = OcrResult(
                engine_name=type(self.ocr_engine).__name__,
                error=f"{type(exc).__name__}: {exc}",
            )
            return self._uncertain_result(
                detection,
                stable_id,
                history,
                reason="ocr_engine_error",
                observations_are_consecutive=observations_are_consecutive,
                ocr_result=fallback_result,
            )

        if not ocr_result.is_available:
            self._clear_candidate(history)
            return self._uncertain_result(
                detection,
                stable_id,
                history,
                reason="ocr_engine_unavailable",
                observations_are_consecutive=observations_are_consecutive,
                ocr_result=ocr_result,
            )
        if ocr_result.error is not None:
            self._clear_candidate(history)
            return self._uncertain_result(
                detection,
                stable_id,
                history,
                reason="ocr_engine_error",
                observations_are_consecutive=observations_are_consecutive,
                ocr_result=ocr_result,
            )

        confident_lines = tuple(
            line
            for line in ocr_result.lines
            if line.text.strip() and line.confidence >= self.minimum_ocr_confidence
        )
        accepted_lines = tuple(
            line
            for line in confident_lines
            if self._text_region_is_large_enough(
                line,
                crop_width=crop_width,
                crop_height=crop_height,
            )
        )
        accepted_result = OcrResult(
            lines=accepted_lines,
            engine_name=ocr_result.engine_name,
            is_available=ocr_result.is_available,
        )
        observed_text = _joined_text(accepted_lines)
        ocr_confidence = accepted_result.confidence
        if not observed_text:
            self._clear_candidate(history)
            if confident_lines:
                reason = "ocr_text_region_too_small_or_unavailable"
            elif ocr_result.text:
                reason = "ocr_confidence_below_threshold"
            else:
                reason = "ocr_text_not_detected"
            return self._uncertain_result(
                detection,
                stable_id,
                history,
                reason=reason,
                observations_are_consecutive=observations_are_consecutive,
                observed_text=ocr_result.text or None,
                ocr_confidence=ocr_result.confidence,
                ocr_result=ocr_result,
            )

        confirmed, previous_text = self._observe_text(
            history,
            observed_text,
            min(detection_confidence, ocr_confidence),
        )
        if not confirmed:
            return self._uncertain_result(
                detection,
                stable_id,
                history,
                reason="ocr_text_not_stable",
                observations_are_consecutive=observations_are_consecutive,
                observed_text=observed_text,
                ocr_confidence=ocr_confidence,
                ocr_result=ocr_result,
            )

        return AnalysisResult(
            object_type=normalize_object_type(detection.class_name),
            stable_id=stable_id,
            state=None,
            confidence=history.confirmed_confidence,
            attributes={
                "class_name": detection.class_name,
                "detection_confidence": detection_confidence,
                "text": history.confirmed_text,
                "observed_text": observed_text,
                "ocr_confidence": ocr_confidence,
                "confirmed_text_confidence": history.confirmed_confidence,
                "ocr_confirmed_frames": history.candidate_frames,
                "ocr_candidate_frames": history.candidate_frames,
                "observations_are_consecutive": observations_are_consecutive,
                "text_changed": previous_text is not None,
                "previous_text": previous_text,
                "ocr_engine": ocr_result.engine_name,
                "ocr_engine_available": True,
                "ocr_error": None,
                "reason": "ocr_text_confirmed",
            },
            is_uncertain=False,
        )

    def reset(self, stable_id: str | None = None) -> None:
        """Forget one text object's history, or all histories when omitted."""
        if stable_id is None:
            self._history_by_stable_id.clear()
        else:
            self._history_by_stable_id.pop(stable_id, None)
