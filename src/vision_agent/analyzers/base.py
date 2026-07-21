from __future__ import annotations

import re
from typing import Protocol

from ..signals import ImageArray, SignalStateResult
from ..types import AnalysisResult, Detection


def normalize_object_type(class_name: str) -> str:
    """Normalize detector labels for routing without changing Detection itself."""
    normalized = re.sub(r"[\s-]+", "_", class_name.strip().lower())
    return normalized or "unknown"


def resolve_stable_id(detection: Detection, stable_id: str | None) -> str:
    """Use an assigned stable ID or a raw-track compatibility fallback."""
    if stable_id is not None and stable_id.strip():
        return stable_id.strip()
    if detection.track_id is not None:
        return f"track-{detection.track_id}"
    raise ValueError("stable_id is required when detection.track_id is unavailable")


class ObjectAnalyzer(Protocol):
    """Common contract for analyzers that operate on one tracked object."""

    def analyze(
        self,
        detection: Detection,
        *,
        stable_id: str,
        crop: ImageArray | None = None,
        precomputed_signal_result: SignalStateResult | None = None,
    ) -> AnalysisResult:
        """Return structured evidence without generating narration or calling TTS."""
        ...
