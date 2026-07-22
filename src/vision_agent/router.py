from __future__ import annotations

from .analyzers import (
    BusAnalyzer,
    GenericVisionAnalyzer,
    KioskAnalyzer,
    ObjectAnalyzer,
    TextObjectAnalyzer,
    TrafficLightAnalyzer,
)
from .analyzers.base import resolve_stable_id
from .object_types import AnalyzerKind, object_class_spec
from .signals import ImageArray, SignalStateResult
from .types import AnalysisResult, Detection


class ObjectRouter:
    """Route each tracked detection to one object-specific analyzer."""

    def __init__(
        self,
        *,
        traffic_light_analyzer: ObjectAnalyzer | None = None,
        bus_analyzer: ObjectAnalyzer | None = None,
        kiosk_analyzer: ObjectAnalyzer | None = None,
        text_object_analyzer: ObjectAnalyzer | None = None,
        generic_vision_analyzer: ObjectAnalyzer | None = None,
    ) -> None:
        self.traffic_light_analyzer = (
            traffic_light_analyzer if traffic_light_analyzer is not None else TrafficLightAnalyzer()
        )
        self.bus_analyzer = bus_analyzer if bus_analyzer is not None else BusAnalyzer()
        self.kiosk_analyzer = kiosk_analyzer if kiosk_analyzer is not None else KioskAnalyzer()
        self.text_object_analyzer = (
            text_object_analyzer if text_object_analyzer is not None else TextObjectAnalyzer()
        )
        self.generic_vision_analyzer = (
            generic_vision_analyzer
            if generic_vision_analyzer is not None
            else GenericVisionAnalyzer()
        )

    def analyzer_for(self, class_name: str) -> ObjectAnalyzer:
        analyzer_kind = object_class_spec(class_name).analyzer
        if analyzer_kind is AnalyzerKind.TRAFFIC_LIGHT:
            return self.traffic_light_analyzer
        if analyzer_kind is AnalyzerKind.BUS:
            return self.bus_analyzer
        if analyzer_kind is AnalyzerKind.KIOSK:
            return self.kiosk_analyzer
        if analyzer_kind is AnalyzerKind.TEXT:
            return self.text_object_analyzer
        return self.generic_vision_analyzer

    def route_detection(
        self,
        detection: Detection,
        *,
        stable_id: str | None = None,
        crop: ImageArray | None = None,
        precomputed_signal_result: SignalStateResult | None = None,
    ) -> AnalysisResult:
        analyzer = self.analyzer_for(detection.class_name)
        return analyzer.analyze(
            detection,
            stable_id=resolve_stable_id(detection, stable_id),
            crop=crop,
            precomputed_signal_result=precomputed_signal_result,
        )

    def route(
        self,
        detection: Detection,
        *,
        stable_id: str | None = None,
        crop: ImageArray | None = None,
        precomputed_signal_result: SignalStateResult | None = None,
    ) -> AnalysisResult:
        """Short alias for callers that already express detection routing in context."""
        return self.route_detection(
            detection,
            stable_id=stable_id,
            crop=crop,
            precomputed_signal_result=precomputed_signal_result,
        )

    def reset(self, stable_id: str | None = None) -> None:
        """Reset per-object analyzer state after disappearance, or reset all state."""
        seen_analyzers: set[int] = set()
        for analyzer in (
            self.traffic_light_analyzer,
            self.bus_analyzer,
            self.kiosk_analyzer,
            self.text_object_analyzer,
            self.generic_vision_analyzer,
        ):
            analyzer_identity = id(analyzer)
            if analyzer_identity in seen_analyzers:
                continue
            seen_analyzers.add(analyzer_identity)
            reset = getattr(analyzer, "reset", None)
            if callable(reset):
                reset(stable_id)
