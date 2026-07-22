from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_agent.analyzers import KioskAnalyzer, TextObjectAnalyzer, TrafficLightAnalyzer
from vision_agent.event_manager import (
    OBJECT_STATE_CHANGED,
    SCREEN_CHANGED,
    TEXT_CONFIRMED,
    SceneEventManager,
)
from vision_agent.narration import NarrationPolicy
from vision_agent.object_types import AnalyzerKind, object_class_spec
from vision_agent.ocr import OcrLine, OcrResult
from vision_agent.pipeline import PipelineConfig, run_video_pipeline
from vision_agent.router import ObjectRouter
from vision_agent.signals import SignalStateResult, SignalTargetSelector
from vision_agent.types import AnalysisEvent, AnalysisResult, Detection, SignalState


def detection(
    class_name: str,
    *,
    frame_index: int = 0,
    class_id: int = 42,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        frame_index=frame_index,
        timestamp_s=frame_index / 30.0,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        xyxy=(0.0, 0.0, 200.0, 100.0),
        track_id=1,
    )


@dataclass
class _RecordingAnalyzer:
    name: str
    calls: int = 0

    def analyze(
        self,
        item: Detection,
        *,
        stable_id: str,
        crop: np.ndarray | None = None,
        precomputed_signal_result: SignalStateResult | None = None,
    ) -> AnalysisResult:
        del item, crop, precomputed_signal_result
        self.calls += 1
        return AnalysisResult(
            object_type=self.name,
            stable_id=stable_id,
            state=None,
            confidence=0.0,
            is_uncertain=True,
        )


def _recording_router() -> ObjectRouter:
    return ObjectRouter(
        traffic_light_analyzer=_RecordingAnalyzer("traffic"),
        bus_analyzer=_RecordingAnalyzer("bus"),
        kiosk_analyzer=_RecordingAnalyzer("kiosk"),
        text_object_analyzer=_RecordingAnalyzer("text"),
        generic_vision_analyzer=_RecordingAnalyzer("generic"),
    )


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("pedestrian signal", "traffic"),
        ("vehicle-traffic-light", "traffic"),
        ("kiosk", "kiosk"),
        ("self-service kiosk", "kiosk"),
        ("touchscreen_kiosk", "kiosk"),
        ("ticket machine", "text"),
        ("bus-route-display", "text"),
        ("sign", "text"),
        ("display", "text"),
        ("screen", "text"),
        ("monitor", "text"),
        ("reverse vending machine", "generic"),
        ("unknown panel", "generic"),
    ],
)
def test_custom_detector_classes_route_to_safe_analyzer(
    class_name: str,
    expected: str,
) -> None:
    result = _recording_router().route_detection(
        detection(class_name),
        stable_id="stable-1",
    )

    assert result.object_type == expected


@pytest.mark.parametrize(
    ("class_name", "kind"),
    [
        ("pedestrian_signal", AnalyzerKind.TRAFFIC_LIGHT),
        ("vehicle_traffic_light", AnalyzerKind.TRAFFIC_LIGHT),
        ("ticket_machine", AnalyzerKind.TEXT),
        ("bus_route_display", AnalyzerKind.TEXT),
        ("reverse_vending_machine", AnalyzerKind.GENERIC),
    ],
)
def test_object_class_registry_is_the_single_runtime_contract(
    class_name: str,
    kind: AnalyzerKind,
) -> None:
    assert object_class_spec(class_name).analyzer is kind


@pytest.mark.parametrize(
    ("class_name", "object_type", "signal_type", "type_is_uncertain"),
    [
        ("traffic light", "traffic_light", "UNKNOWN", True),
        ("pedestrian signal", "pedestrian_signal", "PEDESTRIAN", False),
        ("vehicle-traffic-light", "vehicle_traffic_light", "VEHICLE", False),
    ],
)
def test_traffic_analyzer_preserves_detector_subtype(
    class_name: str,
    object_type: str,
    signal_type: str,
    type_is_uncertain: bool,
) -> None:
    analyzer = TrafficLightAnalyzer(minimum_confirmed_frames=1)
    analysis = analyzer.analyze(
        detection(class_name),
        stable_id="stable-1",
        precomputed_signal_result=SignalStateResult(
            SignalState.GREEN,
            0.9,
            red_ratio=0.0,
            green_ratio=0.2,
        ),
    )

    assert analysis.object_type == object_type
    assert analysis.state == "GREEN"
    assert analysis.attributes["signal_type"] == signal_type
    assert analysis.attributes["signal_type_is_uncertain"] is type_is_uncertain


@pytest.mark.parametrize(
    ("class_name", "object_type"),
    [
        ("kiosk", "kiosk"),
        ("self-service kiosk", "self_service_kiosk"),
        ("touchscreen_kiosk", "touchscreen_kiosk"),
    ],
)
def test_kiosk_analyzer_preserves_supported_detector_class(
    class_name: str,
    object_type: str,
) -> None:
    analysis = KioskAnalyzer().analyze(
        detection(class_name),
        stable_id="stable-2",
    )

    assert analysis.object_type == object_type


def test_signal_selector_uses_names_not_custom_model_class_ids() -> None:
    selector = SignalTargetSelector()
    pedestrian = detection("pedestrian_signal", class_id=37)
    vehicle = detection("vehicle_traffic_light", class_id=38)
    ticket_with_coco_signal_id = detection("ticket_machine", class_id=9)

    assert selector.is_signal_detection(pedestrian) is True
    assert selector.is_signal_detection(vehicle) is True
    assert selector.is_signal_detection(ticket_with_coco_signal_id) is False
    assert selector.select_indices([ticket_with_coco_signal_id]) == []


def test_signal_selector_keeps_overlapping_different_subtypes() -> None:
    selector = SignalTargetSelector()
    pedestrian = detection("pedestrian_signal", class_id=37)
    vehicle = detection("vehicle_traffic_light", class_id=38)

    assert selector.select_indices([pedestrian, vehicle]) == [0, 1]


def test_vehicle_signal_transition_uses_dedicated_event_and_template() -> None:
    manager = SceneEventManager(auto_presence=False)
    policy = NarrationPolicy()

    def result(state: str) -> AnalysisResult:
        return AnalysisResult(
            object_type="vehicle_traffic_light",
            stable_id="stable-3",
            state=state,
            confidence=0.9,
            attributes={
                "detection_confidence": 0.9,
                "minimum_detection_confidence": 0.2,
                "signal_type": "VEHICLE",
                "signal_type_is_uncertain": False,
            },
        )

    assert manager.update([result("GREEN")], 0.0) == []
    events = manager.update([result("RED")], 0.1)

    assert [event.event_type for event in events] == [OBJECT_STATE_CHANGED]
    assert policy.narrate(events) == ["차량 신호가 빨간색으로 바뀌었습니다."]


@pytest.mark.parametrize(
    ("object_type", "expected"),
    [
        ("self_service_kiosk", "무인 키오스크 화면이 바뀌었습니다."),
        ("touchscreen_kiosk", "터치스크린 키오스크 화면이 바뀌었습니다."),
    ],
)
def test_kiosk_subclasses_emit_screen_event(
    object_type: str,
    expected: str,
) -> None:
    manager = SceneEventManager(auto_presence=False)
    analysis = AnalysisResult(
        object_type=object_type,
        stable_id="stable-4",
        state="UNKNOWN",
        confidence=0.0,
        attributes={
            "screen_is_confirmed": True,
            "screen_fingerprint": "screen-a",
            "screen_confidence": 0.9,
        },
        is_uncertain=True,
    )

    events = manager.update([analysis], 0.0)

    assert [event.event_type for event in events] == [SCREEN_CHANGED]
    assert NarrationPolicy().narrate(events) == [expected]


class _RepeatingOcr:
    def __init__(self, text: str) -> None:
        self.result = OcrResult(
            lines=(OcrLine(text, 0.95, (0, 0, 180, 30)),),
            engine_name="fake",
        )

    def recognize(self, image: np.ndarray) -> OcrResult:
        del image
        return self.result


@pytest.mark.parametrize(
    ("class_name", "text", "expected"),
    [
        ("ticket_machine", "사용 불가", "발권기에 사용 불가라고 표시되어 있습니다."),
        ("bus_route_display", "3102", "버스 노선 표시기에 3102라고 표시되어 있습니다."),
    ],
)
def test_machine_text_classes_only_create_confirmed_text(
    class_name: str,
    text: str,
    expected: str,
) -> None:
    analyzer = TextObjectAnalyzer(
        _RepeatingOcr(text),
        minimum_confirmed_frames=1,
    )
    analysis = analyzer.analyze(
        detection(class_name),
        stable_id="stable-5",
        crop=np.zeros((100, 200, 3), dtype=np.uint8),
    )
    events = SceneEventManager(
        auto_presence=False,
        derive_state_changes=False,
    ).update([analysis], 0.0)

    assert analysis.object_type == class_name
    assert analysis.state is None
    assert [event.event_type for event in events] == [TEXT_CONFIRMED]
    assert NarrationPolicy().narrate(events) == [expected]


def test_reverse_vending_machine_never_creates_kiosk_screen_event() -> None:
    manager = SceneEventManager(auto_presence=False)
    unsafe_kiosk_like_result = AnalysisResult(
        object_type="reverse_vending_machine",
        stable_id="stable-6",
        state="PAYMENT",
        confidence=0.9,
        attributes={
            "screen_is_confirmed": True,
            "screen_fingerprint": "screen-a",
            "screen_confidence": 0.9,
        },
    )

    assert manager.update([unsafe_kiosk_like_result], 0.0) == []
    assert (
        NarrationPolicy().narrate(
            AnalysisEvent(
                event_type=SCREEN_CHANGED,
                object_type="reverse_vending_machine",
                stable_id="stable-6",
                timestamp_s=0.0,
                confidence=0.9,
            )
        )
        == []
    )


class _FakeTensor:
    def __init__(self, values: object) -> None:
        self.values = values

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> object:
        return self.values


class _FakeBoxes:
    def __init__(self, class_id: int) -> None:
        self.xyxy = _FakeTensor([[5.0, 2.0, 25.0, 38.0]])
        self.conf = _FakeTensor([0.9])
        self.cls = _FakeTensor([float(class_id)])
        self.id = _FakeTensor([1.0])

    def __len__(self) -> int:
        return 1


class _FakeResult:
    def __init__(self, class_id: int, class_name: str) -> None:
        self.boxes = _FakeBoxes(class_id)
        self.names = {class_id: class_name}


class _FakeModel:
    def __init__(self, class_id: int, class_name: str) -> None:
        self.result = _FakeResult(class_id, class_name)

    def track(self, frame: np.ndarray, **kwargs: object) -> list[_FakeResult]:
        del frame, kwargs
        return [self.result]


def _write_one_frame_video(path: Path, color: tuple[int, int, int]) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (40, 40),
    )
    assert writer.isOpened()
    writer.write(np.full((40, 40, 3), color, dtype=np.uint8))
    writer.release()


def _run_custom_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    class_id: int,
    class_name: str,
    color: tuple[int, int, int],
) -> tuple[dict[str, object], dict[str, object]]:
    source = tmp_path / f"{class_name}.mp4"
    output_dir = tmp_path / f"{class_name}-output"
    _write_one_frame_video(source, color)
    monkeypatch.setattr(
        "vision_agent.pipeline._load_yolo",
        lambda model_name: _FakeModel(class_id, class_name),
    )

    summary = run_video_pipeline(
        PipelineConfig(
            source=str(source),
            output_dir=output_dir,
            device="cpu",
            min_seen_frames=1,
            min_signal_state_frames=1,
            ocr_backend="none",
        )
    )
    jsonl_path = next(output_dir.glob("*_detections.jsonl"))
    with jsonl_path.open(encoding="utf-8") as file:
        frame = json.loads(file.readline())
    return summary, frame


def test_pipeline_does_not_drop_non_signal_with_custom_class_id_nine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary, frame = _run_custom_class(
        tmp_path,
        monkeypatch,
        class_id=9,
        class_name="ticket_machine",
        color=(127, 127, 127),
    )

    item = frame["detections"][0]
    assert summary["signal_targets"] == 0
    assert item["analysis"]["object_type"] == "ticket_machine"
    assert "is_signal_target" not in item
    assert frame["analysis_results"][0]["object_type"] == "ticket_machine"


def test_pipeline_analyzes_custom_signal_with_arbitrary_class_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary, frame = _run_custom_class(
        tmp_path,
        monkeypatch,
        class_id=37,
        class_name="pedestrian_signal",
        color=(0, 255, 0),
    )

    item = frame["detections"][0]
    assert summary["signal_targets"] == 1
    assert item["is_signal_target"] is True
    assert item["signal_state"] == "GREEN"
    assert item["analysis"]["object_type"] == "pedestrian_signal"
    assert item["analysis"]["attributes"]["signal_type"] == "PEDESTRIAN"
