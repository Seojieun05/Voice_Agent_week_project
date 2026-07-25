from __future__ import annotations

import importlib
import inspect
import math
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

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

_DEFAULT_OFFLINE_MODEL_FILENAMES = {
    "Det": "PP-OCRv6_det_small.onnx",
    "Cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "Rec": "PP-OCRv6_rec_small.onnx",
}

_RAPIDOCR_RUNTIME_LOCK = threading.RLock()


@contextmanager
def _rapidocr_download_policy(*, allow_download: bool) -> Iterator[None]:
    """Serialize RapidOCR work and hard-block its 3.9.x downloader offline."""
    with _RAPIDOCR_RUNTIME_LOCK:
        if allow_download:
            yield
            return

        try:
            download_module = importlib.import_module("rapidocr.utils.download_file")
            download_class = getattr(download_module, "DownloadFile")
            original_run = inspect.getattr_static(download_class, "run")
        except (ImportError, AttributeError, TypeError) as exc:
            raise RuntimeError(
                "cannot enforce RapidOCR offline mode: downloader hook unavailable"
            ) from exc

        def blocked_download(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise RuntimeError("RapidOCR download blocked because allow_download is False")

        try:
            setattr(download_class, "run", classmethod(blocked_download))
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "cannot enforce RapidOCR offline mode: downloader hook is immutable"
            ) from exc
        try:
            yield
        finally:
            setattr(download_class, "run", original_run)


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

    @staticmethod
    def _local_model_file(value: object, *, parameter_name: str) -> Path:
        try:
            path = Path(value).expanduser()  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{parameter_name} must be a local model file path") from exc
        if not path.is_file():
            raise RuntimeError(f"{parameter_name} is not a local model file: {path}")
        return path

    @staticmethod
    def _installed_config(rapidocr: object, params: Mapping[str, object]) -> object | None:
        """Load RapidOCR configuration without running its constructor."""
        try:
            rapid_ocr_class = getattr(rapidocr, "RapidOCR")
            load_config = getattr(rapid_ocr_class, "_load_config")
            uninitialized = object.__new__(rapid_ocr_class)
            return load_config(uninitialized, None, dict(params))
        except Exception:
            return None

    @staticmethod
    def _model_roots(
        rapidocr: object,
        params: Mapping[str, object],
        installed_config: object | None,
    ) -> tuple[Path, ...]:
        roots: list[Path] = []
        configured_roots = [params.get("Global.model_root_dir")]
        if installed_config is not None:
            global_config = getattr(installed_config, "Global", None)
            configured_roots.append(getattr(global_config, "model_root_dir", None))
        for configured_root in configured_roots:
            if configured_root is None:
                continue
            try:
                roots.append(Path(configured_root).expanduser())  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Global.model_root_dir must be a local directory path") from exc

        module_file = getattr(rapidocr, "__file__", None)
        if module_file is not None:
            try:
                roots.append(Path(module_file).resolve().parent / "models")
            except (TypeError, ValueError, OSError):
                pass

        module_paths = getattr(rapidocr, "__path__", ())
        for module_path in module_paths:
            try:
                roots.append(Path(module_path).resolve() / "models")
            except (TypeError, ValueError, OSError):
                continue

        unique_roots: list[Path] = []
        for root in roots:
            if root not in unique_roots:
                unique_roots.append(root)
        return tuple(unique_roots)

    def _offline_model_filenames(
        self,
        installed_config: object | None,
    ) -> dict[str, str]:
        filenames = dict(_DEFAULT_OFFLINE_MODEL_FILENAMES)
        language = _LANGUAGE_ALIASES.get(self.language, self.language)
        if language != "default":
            filenames["Rec"] = f"{language}_PP-OCRv5_rec_mobile.onnx"
        if installed_config is None:
            return filenames

        try:
            base_module = importlib.import_module("rapidocr.inference_engine.base")
            file_info_class = getattr(base_module, "FileInfo")
            infer_session_class = getattr(base_module, "InferSession")
        except (ImportError, AttributeError):
            return filenames

        for component in filenames:
            try:
                component_config = getattr(installed_config, component)
                file_info = file_info_class(
                    engine_type=component_config.engine_type,
                    ocr_version=component_config.ocr_version,
                    task_type=component_config.task_type,
                    lang_type=component_config.lang_type,
                    model_type=component_config.model_type,
                )
                model_info = infer_session_class.get_model_url(file_info)
                model_url = str(model_info["model_dir"])
                filename = Path(urlparse(model_url).path).name
            except Exception:
                continue
            if filename:
                filenames[component] = filename
        return filenames

    def _resolve_offline_model_paths(
        self,
        rapidocr: object,
        params: Mapping[str, object],
    ) -> dict[str, object]:
        """Pin every RapidOCR component to a verified local file.

        RapidOCR initializes detection, classification, and recognition sessions in
        its constructor even when one of those stages is disabled at call time. A
        missing ``*.model_path`` therefore enters the RapidOCR downloader. Resolve
        all three paths before construction so ``allow_download=False`` is a strict
        no-network contract.
        """
        resolved = dict(params)
        components = tuple(_DEFAULT_OFFLINE_MODEL_FILENAMES)

        # Validate caller-provided paths first. Do not silently fall back to a
        # bundled model when a misspelled explicit path was supplied.
        for component in components:
            parameter_name = f"{component}.model_path"
            if parameter_name in resolved:
                path = self._local_model_file(
                    resolved[parameter_name],
                    parameter_name=parameter_name,
                )
                resolved[parameter_name] = str(path)

        installed_config = self._installed_config(rapidocr, resolved)
        roots = self._model_roots(rapidocr, resolved, installed_config)
        filenames = self._offline_model_filenames(installed_config)
        for component in components:
            parameter_name = f"{component}.model_path"
            if parameter_name in resolved:
                continue
            filename = filenames[component]
            local_path = None
            for root in roots:
                candidate = root / filename
                if candidate.is_file():
                    local_path = candidate
                    break
            if local_path is None:
                searched = ", ".join(str(root / filename) for root in roots) or filename
                raise RuntimeError(
                    f"{parameter_name} has no local model file; searched: {searched}"
                )
            resolved[parameter_name] = str(local_path)
        return resolved

    def _resolved_params(self, rapidocr: object) -> dict[str, object]:
        params = dict(self.params)
        language = _LANGUAGE_ALIASES.get(self.language, self.language)
        if language != "default":
            enum_name = _LANGUAGE_ENUM_NAMES.get(language)
            if enum_name is None:
                supported = ", ".join(("default", *_LANGUAGE_ENUM_NAMES))
                raise RuntimeError(
                    f"unsupported OCR language {self.language!r}; choose: {supported}"
                )
            try:
                params.setdefault(
                    "Rec.lang_type",
                    getattr(getattr(rapidocr, "LangRec"), enum_name),
                )
                params.setdefault(
                    "Rec.ocr_version",
                    getattr(rapidocr, "OCRVersion").PPOCRV5,
                )
                params.setdefault(
                    "Rec.model_type",
                    getattr(rapidocr, "ModelType").MOBILE,
                )
                params.setdefault(
                    "Rec.engine_type",
                    getattr(rapidocr, "EngineType").ONNXRUNTIME,
                )
            except AttributeError as exc:
                raise RuntimeError(
                    "installed RapidOCR does not expose requested language options"
                ) from exc
        if not self.allow_download:
            params = self._resolve_offline_model_paths(rapidocr, params)
        return params

    def _load_engine(self) -> object | None:
        if self._engine is not None or self._load_error is not None:
            return self._engine
        try:
            rapidocr = importlib.import_module("rapidocr")
            params = self._resolved_params(rapidocr)
            rapid_ocr_class = getattr(rapidocr, "RapidOCR")
            with _rapidocr_download_policy(allow_download=self.allow_download):
                self._engine = rapid_ocr_class(params=params)
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
            with _rapidocr_download_policy(allow_download=self.allow_download):
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
