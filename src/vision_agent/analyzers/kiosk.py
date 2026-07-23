from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass

from ..object_types import is_kiosk_object_type, normalize_object_type
from ..ocr import OcrEngine, OcrResult
from ..signals import ImageArray, SignalStateResult
from ..types import AnalysisResult, Detection


ORDER_TYPE_SELECTION = "ORDER_TYPE_SELECTION"
PAYMENT = "PAYMENT"
CONFIRMATION = "CONFIRMATION"
UNKNOWN = "UNKNOWN"

_DINE_IN_PHRASES = (
    "매장 식사",
    "매장에서",
    "먹고 가기",
    "eat in",
    "dine in",
)
_TAKEOUT_PHRASES = (
    "포장",
    "가져가기",
    "take out",
    "takeout",
    "to go",
)
_PAYMENT_STRONG_PHRASES = (
    "결제 방법",
    "결제 수단",
    "결제하기",
    "카드를 넣어",
    "카드를 삽입",
    "payment method",
    "select payment",
    "insert card",
    "pay now",
)
_PAYMENT_ACTION_PHRASES = ("결제", "payment", "pay")
_PAYMENT_METHOD_PHRASES = ("카드", "현금", "간편 결제", "card", "cash")
_CONFIRMATION_PHRASES = (
    "주문 완료",
    "주문이 완료",
    "결제 완료",
    "결제가 완료",
    "주문하시겠습니까",
    "결제하시겠습니까",
    "order complete",
    "order confirmed",
    "payment complete",
    "confirm order",
)
_OPTION_CONTENT_PHRASES = (
    *_DINE_IN_PHRASES,
    *_TAKEOUT_PHRASES,
    *_PAYMENT_METHOD_PHRASES,
)
_EXACT_OPTION_LABELS = (
    "확인",
    "취소",
    "이전",
    "다음",
    "주문하기",
    "결제하기",
    "confirm",
    "cancel",
    "back",
    "next",
    "order",
    "pay now",
)


@dataclass(slots=True)
class _KioskHistory:
    confirmed_stage: str | None = None
    candidate_stage: str | None = None
    candidate_stage_frames: int = 0
    confirmed_stage_frames: int = 0
    confirmed_fingerprint: str | None = None
    candidate_fingerprint: str | None = None
    candidate_fingerprint_frames: int = 0
    last_frame_index: int | None = None
    last_ocr_frame: int | None = None
    last_ocr_lines: tuple[str, ...] = ()
    last_ocr_confidence: float = 0.0
    last_button_candidates: tuple[dict[str, object], ...] = ()


def _normalize_display_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = normalized.replace("\u200b", " ").replace("\ufeff", " ")
    return " ".join(normalized.split()).strip()


def _normalize_search_text(value: str) -> str:
    normalized = _normalize_display_text(value).casefold()
    return " ".join(re.sub(r"[^0-9a-z가-힣]+", " ", normalized).split())


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_normalize_search_text(phrase) in text for phrase in phrases)


def _infer_stage(lines: tuple[str, ...]) -> str:
    searchable = " ".join(_normalize_search_text(line) for line in lines)
    if not searchable:
        return UNKNOWN

    if _contains_phrase(searchable, _CONFIRMATION_PHRASES):
        return CONFIRMATION

    has_dine_in = _contains_phrase(searchable, _DINE_IN_PHRASES)
    has_takeout = _contains_phrase(searchable, _TAKEOUT_PHRASES)
    if has_dine_in and has_takeout:
        return ORDER_TYPE_SELECTION

    has_payment_action = _contains_phrase(searchable, _PAYMENT_ACTION_PHRASES)
    has_payment_method = _contains_phrase(searchable, _PAYMENT_METHOD_PHRASES)
    if _contains_phrase(searchable, _PAYMENT_STRONG_PHRASES) or (
        has_payment_action and has_payment_method
    ):
        return PAYMENT

    return UNKNOWN


def _visible_options(lines: tuple[str, ...]) -> list[str]:
    options: list[str] = []
    seen: set[str] = set()
    exact_option_labels = {_normalize_search_text(option) for option in _EXACT_OPTION_LABELS}
    for line in lines:
        searchable = _normalize_search_text(line)
        if not (
            _contains_phrase(searchable, _OPTION_CONTENT_PHRASES)
            or searchable in exact_option_labels
        ):
            continue
        if searchable in seen:
            continue
        seen.add(searchable)
        options.append(line)
    return options


def _screen_fingerprint(lines: tuple[str, ...]) -> str | None:
    canonical_lines = sorted({_normalize_search_text(line) for line in lines if line})
    canonical_lines = [line for line in canonical_lines if line]
    if not canonical_lines:
        return None
    canonical_screen = "\x1f".join(canonical_lines).encode("utf-8")
    return hashlib.sha256(canonical_screen).hexdigest()


class KioskAnalyzer:
    """Conservatively stabilize OCR-derived kiosk stages and screen changes."""

    def __init__(
        self,
        ocr_engine: OcrEngine | None = None,
        *,
        minimum_detection_confidence: float = 0.5,
        minimum_ocr_confidence: float = 0.6,
        minimum_confirmed_frames: int = 3,
        minimum_screen_change_frames: int = 3,
        ocr_interval_frames: int = 1,
    ) -> None:
        for name, value in (
            ("minimum_detection_confidence", minimum_detection_confidence),
            ("minimum_ocr_confidence", minimum_ocr_confidence),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        for name, value in (
            ("minimum_confirmed_frames", minimum_confirmed_frames),
            ("minimum_screen_change_frames", minimum_screen_change_frames),
            ("ocr_interval_frames", ocr_interval_frames),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        self.ocr_engine = ocr_engine
        self.minimum_detection_confidence = minimum_detection_confidence
        self.minimum_ocr_confidence = minimum_ocr_confidence
        self.minimum_confirmed_frames = minimum_confirmed_frames
        self.minimum_screen_change_frames = minimum_screen_change_frames
        self.ocr_interval_frames = ocr_interval_frames
        self._history_by_stable_id: dict[str, _KioskHistory] = {}

    @staticmethod
    def _reset_candidates(history: _KioskHistory) -> None:
        history.candidate_stage = None
        history.candidate_stage_frames = 0
        history.candidate_fingerprint = None
        history.candidate_fingerprint_frames = 0

    def _history_for(self, stable_id: str, frame_index: int) -> _KioskHistory:
        history = self._history_by_stable_id.setdefault(stable_id, _KioskHistory())
        if history.last_frame_index is not None and frame_index != history.last_frame_index + 1:
            self._reset_candidates(history)
            history.confirmed_stage_frames = 0
            history.last_ocr_frame = None
        history.last_frame_index = frame_index
        return history

    def _ocr_is_due(self, history: _KioskHistory, frame_index: int) -> bool:
        return (
            history.last_ocr_frame is None
            or frame_index - history.last_ocr_frame >= self.ocr_interval_frames
        )

    def _deferred_result(
        self,
        detection: Detection,
        stable_id: str,
        history: _KioskHistory,
        detection_confidence: float,
    ) -> AnalysisResult:
        lines = history.last_ocr_lines
        observed_stage = _infer_stage(lines)
        options = _visible_options(lines)
        screen_is_confirmed = history.confirmed_fingerprint is not None
        stage_is_confirmed = history.confirmed_stage is not None
        combined_confidence = min(detection_confidence, history.last_ocr_confidence)
        detected_object_type = normalize_object_type(detection.class_name)
        return AnalysisResult(
            object_type=(
                detected_object_type if is_kiosk_object_type(detected_object_type) else "kiosk"
            ),
            stable_id=stable_id,
            state=history.confirmed_stage or UNKNOWN,
            confidence=combined_confidence if stage_is_confirmed else 0.0,
            attributes={
                "class_name": detection.class_name,
                "detection_confidence": detection_confidence,
                "visible_text": list(lines),
                "visible_options": options,
                "recognized_buttons": options,
                "button_candidates": [
                    dict(candidate) for candidate in history.last_button_candidates
                ],
                "button_detection_method": "keyword_allowlist_and_ocr_bbox_candidates",
                "ocr_confidence": history.last_ocr_confidence,
                "ocr_was_run": False,
                "ocr_interval_frames": self.ocr_interval_frames,
                "observed_stage": observed_stage,
                "previous_state": None,
                "stage_changed": False,
                "confirmed_frames": history.confirmed_stage_frames,
                "screen_changed": False,
                "screen_is_confirmed": screen_is_confirmed,
                "screen_initial_confirmation": False,
                "screen_confidence": combined_confidence if screen_is_confirmed else 0.0,
                "screen_candidate_frames": history.candidate_fingerprint_frames,
                "observed_screen_fingerprint": _screen_fingerprint(lines),
                "screen_fingerprint": history.confirmed_fingerprint,
                "reason": "ocr_deferred",
            },
            is_uncertain=not stage_is_confirmed,
        )

    def _read_ocr(
        self,
        crop: ImageArray | None,
    ) -> tuple[tuple[str, ...], float, str | None, list[dict[str, object]]]:
        if self.ocr_engine is None:
            return (), 0.0, "ocr_engine_unavailable", []
        if crop is None or crop.size == 0:
            return (), 0.0, "screen_crop_unavailable", []

        try:
            result: OcrResult = self.ocr_engine.recognize(crop)
        except Exception as exc:
            return (), 0.0, f"ocr_failed:{type(exc).__name__}", []

        raw_error = getattr(result, "error", None)
        error = _normalize_display_text(raw_error) if raw_error is not None else ""
        if not getattr(result, "is_available", True):
            return (), 0.0, error or "ocr_unavailable", []
        if error and not getattr(result, "lines", ()):
            return (), 0.0, error, []

        accepted_lines: list[str] = []
        confidences: list[float] = []
        button_candidates: list[dict[str, object]] = []
        seen: set[str] = set()
        for line in getattr(result, "lines", ()):
            try:
                confidence = float(line.confidence)
            except (AttributeError, TypeError, ValueError):
                continue
            if (
                not math.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
                or confidence < self.minimum_ocr_confidence
            ):
                continue
            text = _normalize_display_text(getattr(line, "text", ""))
            deduplication_key = _normalize_search_text(text)
            if not deduplication_key or deduplication_key in seen:
                continue
            seen.add(deduplication_key)
            accepted_lines.append(text)
            confidences.append(confidence)
            raw_bbox = getattr(line, "bbox", None)
            if (
                isinstance(raw_bbox, (tuple, list))
                and len(raw_bbox) == 4
                and all(
                    isinstance(value, int) and not isinstance(value, bool) for value in raw_bbox
                )
                and 1 <= len(deduplication_key) <= 30
            ):
                left, top, right, bottom = raw_bbox
                if right > left and bottom > top:
                    button_candidates.append(
                        {
                            "text": text,
                            "confidence": confidence,
                            "bbox": [left, top, right, bottom],
                        }
                    )

        if not accepted_lines:
            return (), 0.0, "no_confident_text", []
        return (
            tuple(accepted_lines),
            sum(confidences) / len(confidences),
            None,
            button_candidates,
        )

    def _observe_stage(
        self,
        history: _KioskHistory,
        observed_stage: str,
    ) -> tuple[str, int, str | None, bool, bool]:
        if observed_stage == UNKNOWN:
            history.candidate_stage = None
            history.candidate_stage_frames = 0
            history.confirmed_stage_frames = 0
            return UNKNOWN, 0, None, False, True

        if observed_stage == history.confirmed_stage:
            history.candidate_stage = None
            history.candidate_stage_frames = 0
            history.confirmed_stage_frames += 1
            return observed_stage, history.confirmed_stage_frames, None, False, False

        if observed_stage == history.candidate_stage:
            history.candidate_stage_frames += 1
        else:
            history.candidate_stage = observed_stage
            history.candidate_stage_frames = 1

        candidate_frames = history.candidate_stage_frames
        if candidate_frames < self.minimum_confirmed_frames:
            return UNKNOWN, 0, None, False, True

        previous_stage = history.confirmed_stage
        history.confirmed_stage = observed_stage
        history.confirmed_stage_frames = candidate_frames
        history.candidate_stage = None
        history.candidate_stage_frames = 0
        return (
            observed_stage,
            history.confirmed_stage_frames,
            previous_stage,
            previous_stage is not None,
            False,
        )

    def _observe_fingerprint(
        self,
        history: _KioskHistory,
        fingerprint: str | None,
    ) -> tuple[bool, bool, int, bool]:
        if fingerprint is None:
            history.candidate_fingerprint = None
            history.candidate_fingerprint_frames = 0
            return False, False, 0, False
        if fingerprint == history.confirmed_fingerprint:
            history.candidate_fingerprint = None
            history.candidate_fingerprint_frames = 0
            return False, True, 0, False

        if fingerprint == history.candidate_fingerprint:
            history.candidate_fingerprint_frames += 1
        else:
            history.candidate_fingerprint = fingerprint
            history.candidate_fingerprint_frames = 1

        candidate_frames = history.candidate_fingerprint_frames
        if candidate_frames < self.minimum_screen_change_frames:
            return False, False, candidate_frames, False

        had_confirmed_screen = history.confirmed_fingerprint is not None
        history.confirmed_fingerprint = fingerprint
        history.candidate_fingerprint = None
        history.candidate_fingerprint_frames = 0
        return True, True, candidate_frames, not had_confirmed_screen

    def analyze(
        self,
        detection: Detection,
        *,
        stable_id: str,
        crop: ImageArray | None = None,
        precomputed_signal_result: SignalStateResult | None = None,
    ) -> AnalysisResult:
        del precomputed_signal_result
        history = self._history_for(stable_id, detection.frame_index)
        try:
            detection_confidence = float(detection.confidence)
        except (TypeError, ValueError):
            detection_confidence = 0.0
        detection_is_reliable = (
            math.isfinite(detection_confidence)
            and detection_confidence >= self.minimum_detection_confidence
        )
        if not detection_is_reliable:
            history.last_ocr_frame = None
            lines, ocr_confidence, reason, button_candidates = (
                (),
                0.0,
                "low_detection_confidence",
                [],
            )
            ocr_was_run = False
        elif not self._ocr_is_due(history, detection.frame_index):
            return self._deferred_result(
                detection,
                stable_id,
                history,
                detection_confidence,
            )
        else:
            history.last_ocr_frame = detection.frame_index
            ocr_was_run = self.ocr_engine is not None and crop is not None and crop.size > 0
            lines, ocr_confidence, reason, button_candidates = self._read_ocr(crop)
            history.last_ocr_lines = lines
            history.last_ocr_confidence = ocr_confidence
            history.last_button_candidates = tuple(
                dict(candidate) for candidate in button_candidates
            )
        observed_stage = _infer_stage(lines)
        fingerprint = _screen_fingerprint(lines)
        if reason is not None:
            self._reset_candidates(history)
            history.confirmed_stage_frames = 0

        stage, confirmed_frames, previous_stage, stage_changed, is_uncertain = self._observe_stage(
            history, observed_stage
        )
        (
            screen_changed,
            screen_is_confirmed,
            screen_candidate_frames,
            screen_initial_confirmation,
        ) = self._observe_fingerprint(history, fingerprint)
        options = _visible_options(lines)
        combined_confidence = (
            min(detection_confidence, ocr_confidence) if detection_is_reliable else 0.0
        )
        attributes: dict[str, object] = {
            "class_name": detection.class_name,
            "detection_confidence": detection_confidence,
            "visible_text": list(lines),
            "visible_options": options,
            "recognized_buttons": options,
            "button_candidates": button_candidates,
            "button_detection_method": "keyword_allowlist_and_ocr_bbox_candidates",
            "ocr_confidence": ocr_confidence,
            "ocr_was_run": ocr_was_run,
            "ocr_interval_frames": self.ocr_interval_frames,
            "observed_stage": observed_stage,
            "previous_state": previous_stage,
            "stage_changed": stage_changed,
            "confirmed_frames": confirmed_frames,
            "screen_changed": screen_changed,
            "screen_is_confirmed": screen_is_confirmed,
            "screen_initial_confirmation": screen_initial_confirmation,
            "screen_confidence": combined_confidence if screen_is_confirmed else 0.0,
            "screen_candidate_frames": screen_candidate_frames,
            "observed_screen_fingerprint": fingerprint,
            "screen_fingerprint": history.confirmed_fingerprint if screen_is_confirmed else None,
        }
        if reason is not None:
            attributes["reason"] = reason
        elif observed_stage == UNKNOWN:
            attributes["reason"] = "screen_stage_unknown"
        elif is_uncertain:
            attributes["reason"] = "stage_confirmation_pending"

        detected_object_type = normalize_object_type(detection.class_name)
        return AnalysisResult(
            object_type=(
                detected_object_type if is_kiosk_object_type(detected_object_type) else "kiosk"
            ),
            stable_id=stable_id,
            state=stage,
            confidence=combined_confidence if not is_uncertain else 0.0,
            attributes=attributes,
            is_uncertain=is_uncertain,
        )

    def reset(self, stable_id: str | None = None) -> None:
        """Forget one kiosk's history, or all histories when no ID is supplied."""
        if stable_id is None:
            self._history_by_stable_id.clear()
        else:
            self._history_by_stable_id.pop(stable_id, None)
