from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .signals import ImageArray


OcrBoundingBox = tuple[int, int, int, int]

_LANGUAGE_ENUM_NAMES = {
    "ch": "CH",
    "ch_doc": "CH_DOC",
    "en": "EN",
    "arabic": "ARABIC",
    "chinese_cht": "CHINESE_CHT",
    "cyrillic": "CYRILLIC",
    "devanagari": "DEVANAGARI",
    "japan": "JAPAN",
    "korean": "KOREAN",
    "ka": "KA",
    "latin": "LATIN",
    "ta": "TA",
    "te": "TE",
    "eslav": "ESLAV",
    "th": "TH",
    "el": "EL",
}
_LANGUAGE_ALIASES = {
    "chinese": "ch",
    "english": "en",
    "japanese": "japan",
    "ko": "korean",
    "kor": "korean",
    "traditional_chinese": "chinese_cht",
}


@dataclass(frozen=True, slots=True)
class OcrLine:
    """One OCR text line with normalized confidence and an optional xyxy box."""

    text: str
    confidence: float
    bbox: OcrBoundingBox | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")
        if self.bbox is not None:
            if len(self.bbox) != 4 or any(
                not isinstance(value, int) or isinstance(value, bool) for value in self.bbox
            ):
                raise ValueError("bbox must contain four integer xyxy coordinates")
            left, top, right, bottom = self.bbox
            if right <= left or bottom <= top:
                raise ValueError("bbox must have positive width and height")


@dataclass(frozen=True, slots=True)
class OcrResult:
    """Backend-neutral OCR output with availability and diagnostic metadata."""

    lines: tuple[OcrLine, ...] = ()
    engine_name: str = "injected"
    is_available: bool = True
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lines", tuple(self.lines))
        if not isinstance(self.engine_name, str) or not self.engine_name.strip():
            raise ValueError("engine_name must be a non-empty string")

    @property
    def text(self) -> str:
        """Return non-empty lines in backend reading order."""
        return "\n".join(line.text.strip() for line in self.lines if line.text.strip())

    @property
    def confidence(self) -> float:
        """Return a character-weighted confidence for non-empty lines."""
        weighted_lines = [
            (line, len("".join(line.text.split()))) for line in self.lines if line.text.strip()
        ]
        total_weight = sum(weight for _, weight in weighted_lines)
        if total_weight == 0:
            return 0.0
        return sum(line.confidence * weight for line, weight in weighted_lines) / total_weight


@runtime_checkable
class OcrEngine(Protocol):
    """Replaceable local OCR backend used by object-specific analyzers."""

    def recognize(self, image: ImageArray) -> OcrResult:
        """Recognize text without performing narration or network I/O."""
        ...


def _axis_aligned_bbox(raw_box: object) -> OcrBoundingBox | None:
    """Convert a RapidOCR polygon to an integer xyxy box."""
    try:
        coordinates = np.asarray(raw_box, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if coordinates.ndim != 2 or coordinates.shape[0] < 2 or coordinates.shape[1] < 2:
        return None
    xy = coordinates[:, :2]
    if not np.all(np.isfinite(xy)):
        return None
    left = int(math.floor(float(np.min(xy[:, 0]))))
    top = int(math.floor(float(np.min(xy[:, 1]))))
    right = int(math.ceil(float(np.max(xy[:, 0]))))
    bottom = int(math.ceil(float(np.max(xy[:, 1]))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


class RapidOcrEngine:
    """Optional RapidOCR adapter with no import-time heavyweight dependency."""

    def __init__(
        self,
        *,
        language: str = "korean",
        allow_download: bool = False,
        params: Mapping[str, object] | None = None,
        **parameter_overrides: object,
    ) -> None:
        self.language = language.strip().lower()
        self.allow_download = allow_download
        self.params = dict(params or {})
        self.params.update(parameter_overrides)
        if "rec_model_path" in self.params:
            self.params.setdefault("Rec.model_path", self.params.pop("rec_model_path"))
        self._engine: object | None = None
        self._load_error: str | None = None

    def _resolved_params(self, rapidocr: object) -> dict[str, object]:
        params = dict(self.params)
        language = _LANGUAGE_ALIASES.get(self.language, self.language)
        if language == "default":
            return params
        if not self.allow_download and "Rec.model_path" not in params:
            raise RuntimeError(
                "An explicit RapidOCR language requires a local Rec.model_path when "
                "allow_download is False"
            )
        enum_name = _LANGUAGE_ENUM_NAMES.get(language)
        if enum_name is None:
            supported = ", ".join(("default", *_LANGUAGE_ENUM_NAMES))
            raise RuntimeError(f"unsupported OCR language {self.language!r}; choose: {supported}")
        try:
            params.setdefault("Rec.lang_type", getattr(getattr(rapidocr, "LangRec"), enum_name))
            params.setdefault("Rec.ocr_version", getattr(rapidocr, "OCRVersion").PPOCRV5)
            params.setdefault("Rec.model_type", getattr(rapidocr, "ModelType").MOBILE)
            params.setdefault(
                "Rec.engine_type",
                getattr(rapidocr, "EngineType").ONNXRUNTIME,
            )
        except AttributeError as exc:
            raise RuntimeError("installed RapidOCR does not expose requested language options") from exc
        return params

    def _load_engine(self) -> object | None:
        if self._engine is not None or self._load_error is not None:
            return self._engine
        try:
            rapidocr = importlib.import_module("rapidocr")
            rapid_ocr_class = getattr(rapidocr, "RapidOCR")
            self._engine = rapid_ocr_class(params=self._resolved_params(rapidocr))
        except (ImportError, AttributeError, RuntimeError) as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
        return self._engine

    def recognize(self, image: ImageArray) -> OcrResult:
        if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or image.size == 0:
            return OcrResult(engine_name="rapidocr", error="invalid_image")

        engine = self._load_engine()
        if engine is None:
            return OcrResult(
                engine_name="rapidocr",
                is_available=False,
                error=self._load_error or "rapidocr_unavailable",
            )

        try:
            output = engine(image)  # type: ignore[operator]
        except Exception as exc:
            return OcrResult(
                engine_name="rapidocr",
                error=f"{type(exc).__name__}: {exc}",
            )

        boxes = getattr(output, "boxes", None)
        texts = getattr(output, "txts", None)
        scores = getattr(output, "scores", None)
        if texts is None:
            return OcrResult(engine_name="rapidocr")

        text_values = list(texts)
        box_values = list(boxes) if boxes is not None else []
        score_values = list(scores) if scores is not None else []
        lines: list[OcrLine] = []
        for index, raw_text in enumerate(text_values):
            text = str(raw_text).strip()
            if not text:
                continue
            try:
                confidence = float(score_values[index]) if index < len(score_values) else 0.0
            except (TypeError, ValueError):
                continue
            if not math.isfinite(confidence):
                continue
            confidence = min(1.0, max(0.0, confidence))
            bbox = _axis_aligned_bbox(box_values[index]) if index < len(box_values) else None
            lines.append(OcrLine(text=text, confidence=confidence, bbox=bbox))
        return OcrResult(lines=tuple(lines), engine_name="rapidocr")


class UnavailableOcrEngine:
    """Explicit fallback useful when an application disables optional OCR."""

    def __init__(self, reason: str = "ocr_disabled") -> None:
        self.reason = reason

    def recognize(self, image: ImageArray) -> OcrResult:
        return OcrResult(
            engine_name="unavailable",
            is_available=False,
            error=self.reason,
        )
