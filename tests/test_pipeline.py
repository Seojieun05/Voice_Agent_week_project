from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_agent.event_manager import SceneEventManager
from vision_agent.narration import NarrationPolicy
from vision_agent.pipeline import (
    PipelineConfig,
    _build_object_router,
    _bus_overlay_parts,
    _build_ocr_engine,
    _build_detection_payload,
    _build_performance_summary,
    _calculate_frame_timestamp_s,
    _read_source_fps,
    _save_signal_crop,
)
from vision_agent.ocr import RapidOcrEngine, UnavailableOcrEngine
from vision_agent.signals import SignalStateResult
from vision_agent.types import AnalysisResult, Detection, SignalState
from vision_agent.vlm import TransformersVisionLanguageModel


class _FakeTensor:
    def __init__(self, values: object) -> None:
        self._values = values

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> object:
        return self._values


class _FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = _FakeTensor([[3.0, 4.0, 15.0, 14.0]])
        self.conf = _FakeTensor([0.9])
        self.cls = _FakeTensor([5.0])
        self.id = _FakeTensor([7.0])

    def __len__(self) -> int:
        return 1


class _FakeYoloResult:
    boxes = _FakeBoxes()
    names = {5: "bus"}


class _FakeYoloModel:
    def track(self, frame: np.ndarray, **kwargs: object) -> list[_FakeYoloResult]:
        return [_FakeYoloResult()]


class _EmptyBoxes:
    def __len__(self) -> int:
        return 0


class _EmptyYoloResult:
    boxes = _EmptyBoxes()
    names: dict[int, str] = {}


class _TransientYoloModel:
    def __init__(self) -> None:
        self.calls = 0

    def track(self, frame: np.ndarray, **kwargs: object) -> list[object]:
        self.calls += 1
        return [_FakeYoloResult() if self.calls == 1 else _EmptyYoloResult()]


class _CropRecordingRouter:
    def __init__(self) -> None:
        self.crops: list[np.ndarray | None] = []
        self.reset_calls: list[str | None] = []

    def route_detection(
        self,
        item: Detection,
        *,
        stable_id: str,
        crop: np.ndarray | None = None,
        precomputed_signal_result: SignalStateResult | None = None,
    ) -> AnalysisResult:
        self.crops.append(crop)
        return AnalysisResult(
            object_type="bus",
            stable_id=stable_id,
            state="UNKNOWN",
            confidence=0.0,
            attributes={},
            is_uncertain=True,
        )

    def reset(self, stable_id: str | None = None) -> None:
        self.reset_calls.append(stable_id)


class FakeCapture:
    def __init__(self, *positions_msec: object) -> None:
        self._positions_msec = iter(positions_msec)

    def get(self, _property_id: int) -> object:
        return next(self._positions_msec)


@pytest.mark.parametrize(
    ("reported_fps", "expected"),
    [
        (29.97, 29.97),
        (0.0, 0.0),
        (-1.0, 0.0),
        (math.nan, 0.0),
        (math.inf, 0.0),
        (None, 0.0),
    ],
)
def test_source_fps_reports_zero_when_metadata_is_unavailable(
    reported_fps: object,
    expected: float,
) -> None:
    capture = FakeCapture(reported_fps)

    assert _read_source_fps(capture) == expected


def test_timestamp_prefers_capture_position() -> None:
    capture = FakeCapture(1250.0)

    timestamp_s = _calculate_frame_timestamp_s(capture, 90, 30.0, 1.0)

    assert timestamp_s == pytest.approx(1.25)


def test_timestamp_accepts_zero_for_first_frame() -> None:
    capture = FakeCapture(0.0)

    assert _calculate_frame_timestamp_s(capture, 0, 30.0, 0.0) == 0.0


@pytest.mark.parametrize("position_msec", [0.0, -1.0, math.nan, math.inf, None])
def test_timestamp_falls_back_when_capture_position_is_unavailable(
    position_msec: object,
) -> None:
    capture = FakeCapture(position_msec)

    timestamp_s = _calculate_frame_timestamp_s(capture, 45, 30.0, 0.0)

    assert timestamp_s == pytest.approx(1.5)


def test_timestamp_never_moves_backwards() -> None:
    capture = FakeCapture(0.0, 100.0, 50.0, 300.0)
    timestamps: list[float] = []
    previous_timestamp_s = 0.0

    for frame_index in range(4):
        timestamp_s = _calculate_frame_timestamp_s(
            capture,
            frame_index,
            10.0,
            previous_timestamp_s,
        )
        timestamps.append(timestamp_s)
        previous_timestamp_s = timestamp_s

    assert timestamps == pytest.approx([0.0, 0.1, 0.1, 0.3])
    assert all(current >= previous for previous, current in zip(timestamps, timestamps[1:]))


def test_performance_summary_uses_last_timestamp_and_has_required_fields() -> None:
    summary = _build_performance_summary(
        frames=120,
        elapsed_s=6.0,
        source_fps=30.0,
        last_timestamp_s=3.0,
        inference_ms_values=[10.0, 20.0, 30.0],
    )

    assert summary == {
        "video_duration_s": 3.0,
        "source_fps": 30.0,
        "frames": 120,
        "elapsed_s": 6.0,
        "effective_fps": 20.0,
        "realtime_factor": 2.0,
        "average_inference_ms": 20.0,
    }


def test_performance_summary_falls_back_to_frames_over_fps() -> None:
    summary = _build_performance_summary(
        frames=60,
        elapsed_s=4.0,
        source_fps=20.0,
        last_timestamp_s=0.0,
        inference_ms_values=[],
    )

    assert summary["video_duration_s"] == 3.0
    assert summary["effective_fps"] == 15.0
    assert summary["realtime_factor"] == 1.333
    assert summary["average_inference_ms"] == 0.0


def test_performance_summary_handles_zero_denominators() -> None:
    summary = _build_performance_summary(
        frames=0,
        elapsed_s=0.0,
        source_fps=30.0,
        last_timestamp_s=None,
        inference_ms_values=[],
    )

    assert summary["video_duration_s"] == 0.0
    assert summary["effective_fps"] == 0.0
    assert summary["realtime_factor"] == 0.0


def test_pipeline_defaults_use_stable_signal_model() -> None:
    config = PipelineConfig(source="video.mp4")

    assert config.model == "yolo26s.pt"
    assert config.image_size == 640
    assert config.confidence == 0.10
    assert config.classify_signal_states is True
    assert config.signal_minimum_detection_confidence == 0.2
    assert config.signal_minimum_color_ratio == 0.015
    assert config.save_crops is False
    assert config.ocr_backend == "rapidocr"
    assert config.ocr_language == "default"
    assert config.allow_ocr_download is False
    assert config.bus_motion_window_frames == 9
    assert config.bus_minimum_detection_confidence == 0.3
    assert config.bus_minimum_area_change_ratio == 0.1
    assert config.bus_minimum_direction_consistency == 0.65
    assert config.bus_area_jitter_tolerance_ratio == 0.02
    assert config.bus_maximum_motion_frame_gap == 2
    assert config.bus_route_ocr_interval_frames == 7
    assert config.bus_route_ocr_requires_relevant_motion is True
    assert config.generic_vlm_model is None
    assert config.allow_vlm_download is False
    assert config.generic_vlm_classes == ("unknown", "unknown_object", "unknown_panel")


def test_pipeline_builds_one_lazy_ocr_engine_shared_by_domain_analyzers() -> None:
    config = PipelineConfig(
        source="video.mp4",
        ocr_model_path="/models/korean-rec.onnx",
        bus_motion_window_frames=7,
        bus_minimum_detection_confidence=0.23,
        bus_minimum_area_change_ratio=0.06,
        bus_maximum_motion_frame_gap=1,
        bus_route_ocr_interval_frames=4,
        bus_route_ocr_requires_relevant_motion=False,
    )

    router = _build_object_router(config, signal_classifier=None)

    ocr_engine = router.bus_analyzer.ocr_engine
    assert isinstance(ocr_engine, RapidOcrEngine)
    assert ocr_engine.params["Rec.model_path"] == "/models/korean-rec.onnx"
    assert router.kiosk_analyzer.ocr_engine is ocr_engine
    assert router.text_object_analyzer.ocr_engine is ocr_engine
    assert router.bus_analyzer.motion_window_frames == 7
    assert router.bus_analyzer.minimum_detection_confidence == 0.23
    assert router.bus_analyzer.minimum_area_change_ratio == 0.06
    assert router.bus_analyzer.maximum_motion_frame_gap == 1
    assert router.bus_analyzer.route_ocr_interval_frames == 4
    assert router.bus_analyzer.route_ocr_requires_relevant_motion is False
    assert router.generic_vision_analyzer.model is None


def test_pipeline_can_disable_ocr_explicitly() -> None:
    engine = _build_ocr_engine(PipelineConfig(source="video.mp4", ocr_backend="none"))

    assert isinstance(engine, UnavailableOcrEngine)


def test_pipeline_rejects_unknown_ocr_backend() -> None:
    with pytest.raises(ValueError, match="ocr_backend"):
        _build_ocr_engine(PipelineConfig(source="video.mp4", ocr_backend="cloud"))


def test_pipeline_configures_lazy_generic_vlm_without_loading_it() -> None:
    config = PipelineConfig(
        source="video.mp4",
        generic_vlm_model="/models/local-vlm",
        generic_vlm_device="cpu",
    )

    router = _build_object_router(config, signal_classifier=None)

    model = router.generic_vision_analyzer.model
    assert isinstance(model, TransformersVisionLanguageModel)
    assert model.model_name_or_path == "/models/local-vlm"
    assert model.device == "cpu"
    assert model.allow_download is False
    assert model._pipeline is None
    assert router.generic_vision_analyzer.allowed_object_types == frozenset(
        {"unknown", "unknown_object", "unknown_panel"}
    )
    assert router.generic_vision_analyzer.minimum_seen_frames_before_inference == 3
    assert router.traffic_light_analyzer.minimum_detection_confidence == 0.2


def test_video_pipeline_routes_a_non_signal_detection_with_its_crop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "one-frame.mp4"
    source_writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (20, 20),
    )
    assert source_writer.isOpened()
    source_writer.write(np.full((20, 20, 3), 127, dtype=np.uint8))
    source_writer.release()
    monkeypatch.setattr("vision_agent.pipeline._load_yolo", lambda model_name: _FakeYoloModel())
    manager_options: dict[str, object] = {}

    def build_manager(**kwargs: object) -> SceneEventManager:
        manager_options.update(kwargs)
        return SceneEventManager(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("vision_agent.pipeline.SceneEventManager", build_manager)
    router = _CropRecordingRouter()
    narration_policy = NarrationPolicy(minimum_confidence=0.8)

    from vision_agent.pipeline import run_video_pipeline

    summary = run_video_pipeline(
        PipelineConfig(
            source=str(source_path),
            output_dir=tmp_path / "output",
            device="cpu",
        ),
        object_router=router,  # type: ignore[arg-type]
        narration_policy=narration_policy,
    )

    assert summary["frames"] == 1
    assert summary["classes"] is None
    assert summary["ocr_language"] == "default"
    assert summary["bus_analysis_results"] == 1
    assert summary["bus_detection_frames"] == 1
    assert summary["bus_motion_state_counts"]["UNKNOWN"] == 1
    assert len(router.crops) == 1
    assert router.crops[0] is not None
    assert router.crops[0].shape == (10, 12, 3)
    assert manager_options["minimum_approach_confidence"] == 0.8
    assert manager_options["minimum_presence_confidence"] == 0.8
    assert manager_options["minimum_domain_confidence"] == 0.8


def test_video_pipeline_resets_transient_analyzer_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "two-frames.mp4"
    source_writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (20, 20),
    )
    assert source_writer.isOpened()
    for value in (80, 160):
        source_writer.write(np.full((20, 20, 3), value, dtype=np.uint8))
    source_writer.release()
    monkeypatch.setattr(
        "vision_agent.pipeline._load_yolo",
        lambda model_name: _TransientYoloModel(),
    )
    router = _CropRecordingRouter()

    from vision_agent.pipeline import run_video_pipeline

    run_video_pipeline(
        PipelineConfig(
            source=str(source_path),
            output_dir=tmp_path / "output",
            device="cpu",
            min_seen_frames=3,
            max_missed_frames=1,
        ),
        object_router=router,  # type: ignore[arg-type]
    )

    assert router.reset_calls == ["stable-1"]


def test_detection_payload_preserves_raw_id_and_adds_signal_metadata() -> None:
    detection = Detection(
        frame_index=2,
        timestamp_s=0.1,
        class_id=9,
        class_name="traffic light",
        confidence=0.7,
        xyxy=(1.0, 2.0, 3.0, 4.0),
        track_id=17,
    )
    result = SignalStateResult(
        state=SignalState.GREEN,
        confidence=0.9,
        red_ratio=0.01,
        green_ratio=0.2,
    )
    analysis = AnalysisResult(
        object_type="traffic_light",
        stable_id="stable-1",
        state="GREEN",
        confidence=0.9,
        attributes={"confirmed_frames": 3},
        is_uncertain=False,
    )

    payload = _build_detection_payload(
        detection,
        stable_object_key="traffic light:stable-1",
        is_signal_target=True,
        signal_result=result,
        analysis_result=analysis,
    )

    assert payload["track_id"] == 17
    assert payload["stable_object_key"] == "traffic light:stable-1"
    assert payload["is_signal_target"] is True
    assert payload["signal_state"] == "GREEN"
    assert payload["signal_green_ratio"] == 0.2
    assert payload["signal_yellow_ratio"] == 0.0
    assert payload["analysis"]["stable_id"] == "stable-1"
    assert payload["analysis"]["attributes"]["confirmed_frames"] == 3


def test_save_signal_crop_uses_required_filename_fields(tmp_path: Path) -> None:
    path = _save_signal_crop(
        tmp_path,
        crop=np.zeros((5, 5, 3), dtype=np.uint8),
        frame_index=12,
        stable_object_key="traffic light:stable-3",
        confidence=0.4567,
    )

    assert path.exists()
    assert path.name == "frame_000012__traffic-light-stable-3__conf_0.457.jpg"


def test_bus_overlay_exposes_motion_state_and_only_confirmed_route() -> None:
    approaching = AnalysisResult(
        object_type="bus",
        stable_id="stable-7",
        state="APPROACHING",
        confidence=0.9,
        attributes={"route_number": "3102"},
    )
    unknown = AnalysisResult(
        object_type="bus",
        stable_id="stable-8",
        state="UNKNOWN",
        confidence=0.0,
        attributes={"route_number": None},
        is_uncertain=True,
    )

    assert _bus_overlay_parts(approaching) == ("APPROACHING", "route=3102")
    assert _bus_overlay_parts(unknown) == ("UNKNOWN",)
    assert _bus_overlay_parts(None) == ()
