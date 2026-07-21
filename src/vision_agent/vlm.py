from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .signals import ImageArray


@dataclass(frozen=True, slots=True)
class VisionLanguageResult:
    """One backend response. Generative confidence must be explicitly calibrated."""

    description: str | None
    confidence: float | None = None
    raw_output: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be None or a finite value between 0 and 1")


class VisionLanguageModel(Protocol):
    """Backend contract for a crop-scoped, non-safety Generic Vision fallback."""

    def describe(self, image: ImageArray, prompt: str) -> VisionLanguageResult:
        """Describe only directly visible evidence in one object crop."""
        ...


def _extract_text(value: object) -> str | None:
    if isinstance(value, str):
        normalized = " ".join(value.split())
        return normalized or None
    if isinstance(value, Mapping):
        for key in ("generated_text", "text", "content"):
            if key in value:
                text = _extract_text(value[key])
                if text:
                    return text
        return None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in reversed(value):
            text = _extract_text(item)
            if text:
                return text
    return None


class TransformersVisionLanguageModel:
    """Lazy local adapter for Transformers' ``image-text-to-text`` pipeline.

    A Hugging Face model identifier may download weights only when
    ``allow_download=True``. Otherwise ``model_name_or_path`` must be a local path.
    The pipeline API does not expose calibrated sequence confidence, so callers must
    explicitly choose ``assumed_confidence`` and still apply temporal confirmation.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | int | None = None,
        max_new_tokens: int = 64,
        allow_download: bool = False,
        assumed_confidence: float = 0.6,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not model_name_or_path.strip():
            raise ValueError("model_name_or_path must not be blank")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if not math.isfinite(assumed_confidence) or not 0.0 <= assumed_confidence <= 1.0:
            raise ValueError("assumed_confidence must be between 0 and 1")
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.allow_download = allow_download
        self.assumed_confidence = assumed_confidence
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any | None = None

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if self._pipeline_factory is None:
            if not self.allow_download and not Path(self.model_name_or_path).exists():
                raise RuntimeError(
                    "VLM 다운로드가 비활성화되어 있습니다. 로컬 모델 경로를 지정하거나 "
                    "--allow-vlm-download를 사용하세요."
                )
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Transformers VLM이 설치되지 않았습니다. `pip install -e '.[vlm]'`을 "
                    "실행하세요."
                ) from exc
            self._pipeline_factory = pipeline

        kwargs: dict[str, object] = {
            "task": "image-text-to-text",
            "model": self.model_name_or_path,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        self._pipeline = self._pipeline_factory(**kwargs)
        return self._pipeline

    def describe(self, image: ImageArray, prompt: str) -> VisionLanguageResult:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            return VisionLanguageResult(None, error="invalid_image")
        try:
            from PIL import Image

            rgb_image = np.ascontiguousarray(image[:, :, ::-1])
            pil_image = Image.fromarray(rgb_image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            output = self._load_pipeline()(
                text=messages,
                images=[pil_image],
                max_new_tokens=self.max_new_tokens,
                return_full_text=False,
            )
        except Exception as exc:  # optional backend/model errors must not stop the video loop
            return VisionLanguageResult(None, error=f"{type(exc).__name__}: {exc}")

        description = _extract_text(output)
        if description is None:
            return VisionLanguageResult(None, raw_output=str(output), error="empty_vlm_output")
        return VisionLanguageResult(
            description=description,
            confidence=self.assumed_confidence,
            raw_output=description,
        )
