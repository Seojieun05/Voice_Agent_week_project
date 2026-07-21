from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from vision_agent.router import ObjectRouter
from vision_agent.signals import ImageArray, SignalStateResult
from vision_agent.types import AnalysisResult, Detection, SignalState


def detection(class_name: str, *, track_id: int | None = 7) -> Detection:
    return Detection(
        frame_index=0,
        timestamp_s=0.0,
        class_id=0,
        class_name=class_name,
        confidence=0.8,
        xyxy=(0.0, 0.0, 10.0, 10.0),
        track_id=track_id,
    )


@dataclass
class _RecordingAnalyzer:
    name: str
    calls: list[tuple[Detection, str, ImageArray | None, SignalStateResult | None]] = field(
        default_factory=list
    )
    reset_calls: list[str | None] = field(default_factory=list)

    def analyze(
        self,
        item: Detection,
        *,
        stable_id: str,
        crop: ImageArray | None = None,
        precomputed_signal_result: SignalStateResult | None = None,
    ) -> AnalysisResult:
        self.calls.append((item, stable_id, crop, precomputed_signal_result))
        return AnalysisResult(
            object_type=self.name,
            stable_id=stable_id,
            state=None,
            confidence=0.0,
            attributes={},
            is_uncertain=True,
        )

    def reset(self, stable_id: str | None = None) -> None:
        self.reset_calls.append(stable_id)


@pytest.fixture
def injected_router() -> tuple[ObjectRouter, dict[str, _RecordingAnalyzer]]:
    analyzers = {
        name: _RecordingAnalyzer(name) for name in ("traffic", "bus", "kiosk", "text", "generic")
    }
    router = ObjectRouter(
        traffic_light_analyzer=analyzers["traffic"],
        bus_analyzer=analyzers["bus"],
        kiosk_analyzer=analyzers["kiosk"],
        text_object_analyzer=analyzers["text"],
        generic_vision_analyzer=analyzers["generic"],
    )
    return router, analyzers


@pytest.mark.parametrize("class_name", ["traffic light", "traffic_light", "Traffic-Light"])
def test_traffic_light_aliases_route_to_traffic_analyzer(
    injected_router: tuple[ObjectRouter, dict[str, _RecordingAnalyzer]],
    class_name: str,
) -> None:
    router, _ = injected_router

    assert (
        router.route_detection(detection(class_name), stable_id="stable-1").object_type == "traffic"
    )


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("bus", "bus"),
        ("kiosk", "kiosk"),
        ("self-service kiosk", "kiosk"),
        ("touchscreen_kiosk", "kiosk"),
        ("sign", "text"),
        ("stop sign", "text"),
        ("display", "text"),
        ("screen", "text"),
        ("monitor", "text"),
        ("tv", "text"),
        ("person", "generic"),
        ("unknown", "generic"),
    ],
)
def test_object_classes_route_to_their_injected_analyzers(
    injected_router: tuple[ObjectRouter, dict[str, _RecordingAnalyzer]],
    class_name: str,
    expected: str,
) -> None:
    router, analyzers = injected_router

    result = router.route_detection(detection(class_name), stable_id="stable-4")

    assert result.object_type == expected
    assert len(analyzers[expected].calls) == 1


def test_router_forwards_stable_id_crop_and_precomputed_signal_result(
    injected_router: tuple[ObjectRouter, dict[str, _RecordingAnalyzer]],
) -> None:
    router, analyzers = injected_router
    crop = np.zeros((4, 5, 3), dtype=np.uint8)
    signal_result = SignalStateResult(SignalState.RED, 0.9, 0.2, 0.0, 0.0)
    item = detection("traffic light")

    router.route_detection(
        item,
        stable_id="stable-9",
        crop=crop,
        precomputed_signal_result=signal_result,
    )

    called_item, called_id, called_crop, called_signal_result = analyzers["traffic"].calls[0]
    assert called_item is item
    assert called_id == "stable-9"
    assert called_crop is crop
    assert called_signal_result is signal_result


def test_router_uses_raw_track_only_as_compatibility_fallback(
    injected_router: tuple[ObjectRouter, dict[str, _RecordingAnalyzer]],
) -> None:
    router, _ = injected_router

    result = router.route(detection("bus", track_id=17))

    assert result.stable_id == "track-17"


def test_router_requires_stable_id_when_raw_track_id_is_unavailable(
    injected_router: tuple[ObjectRouter, dict[str, _RecordingAnalyzer]],
) -> None:
    router, _ = injected_router

    with pytest.raises(ValueError, match="stable_id is required"):
        router.route_detection(detection("unknown panel", track_id=None))


def test_router_resets_each_injected_analyzer_for_disappeared_stable_id(
    injected_router: tuple[ObjectRouter, dict[str, _RecordingAnalyzer]],
) -> None:
    router, analyzers = injected_router

    router.reset("stable-3")

    assert all(analyzer.reset_calls == ["stable-3"] for analyzer in analyzers.values())


def test_router_only_resets_a_shared_analyzer_instance_once() -> None:
    shared = _RecordingAnalyzer("shared")
    router = ObjectRouter(
        traffic_light_analyzer=shared,
        bus_analyzer=shared,
        kiosk_analyzer=shared,
        text_object_analyzer=shared,
        generic_vision_analyzer=shared,
    )

    router.reset()

    assert shared.reset_calls == [None]
