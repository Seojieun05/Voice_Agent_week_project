from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from .analyzers import (
    BusAnalyzer,
    GenericVisionAnalyzer,
    KioskAnalyzer,
    TextObjectAnalyzer,
    TrafficLightAnalyzer,
)
from .event_manager import OBJECT_APPROACHING, SceneEventManager
from .events import StableObjectEventEngine
from .io import JsonlWriter
from .narration import Narration, NarrationPolicy, NarrationScheduler
from .ocr import OcrEngine, RapidOcrEngine, UnavailableOcrEngine
from .router import ObjectRouter
from .signals import (
    HsvSignalClassifierConfig,
    HsvSignalStateClassifier,
    ImageArray,
    SignalStateClassifier,
    SignalStateResult,
    SignalTargetSelector,
    crop_frame_to_bbox,
)
from .types import AnalysisEvent, AnalysisResult, Detection, SceneEvent, SignalState
from .vlm import TransformersVisionLanguageModel


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    source: str
    model: str = "yolo26s.pt"
    output_dir: Path = Path("outputs")
    classes: tuple[int, ...] | None = None
    confidence: float = 0.10
    image_size: int = 640
    device: str | None = None
    track: bool = True
    tracker: str = "bytetrack.yaml"
    min_seen_frames: int = 3
    max_missed_frames: int = 8
    reconnect_iou_threshold: float = 0.3
    max_reconnect_frames: int = 3
    bus_motion_window_frames: int = 9
    bus_minimum_detection_confidence: float = 0.3
    bus_minimum_area_change_ratio: float = 0.1
    bus_minimum_direction_consistency: float = 0.65
    bus_area_jitter_tolerance_ratio: float = 0.02
    bus_maximum_motion_frame_gap: int = 2
    bus_route_ocr_interval_frames: int = 7
    bus_route_ocr_requires_relevant_motion: bool = True
    kiosk_ocr_interval_frames: int = 1
    text_ocr_interval_frames: int = 1
    classify_signal_states: bool = True
    min_signal_state_frames: int = 3
    signal_minimum_detection_confidence: float = 0.2
    signal_minimum_color_ratio: float = 0.015
    signal_minimum_score_margin: float = 0.015
    signal_minimum_dominance_ratio: float = 2.0
    save_crops: bool = False
    ocr_backend: str = "rapidocr"
    ocr_language: str = "default"
    ocr_model_path: str | None = None
    allow_ocr_download: bool = False
    generic_vlm_model: str | None = None
    generic_vlm_device: str | int | None = None
    allow_vlm_download: bool = False
    generic_vlm_classes: tuple[str, ...] = (
        "unknown",
        "unknown_object",
        "unknown_panel",
    )
    narration_presence_classes: tuple[str, ...] = ()
    narrate_bus_approach: bool = True
    narration_queue_size: int = 32
    narration_ttl_s: float = 5.0


@dataclass(frozen=True, slots=True)
class FrameContext:
    """Timing and ordering metadata for one frame accepted by a session."""

    source_sequence_id: int
    processed_index: int
    captured_at_s: float | None
    received_at_s: float
    processing_started_at_s: float
    dropped_frames: int = 0


@dataclass(slots=True)
class FrameAnalysis:
    """One shared frame result used by both MP4 and live transports."""

    source_sequence_id: int
    processed_index: int
    timestamp_s: float
    detections: list[Detection]
    scene_events: list[SceneEvent]
    analysis_results: list[AnalysisResult]
    analysis_events: list[AnalysisEvent]
    narrations: list[str]
    timings: dict[str, float]
    dropped_frames: int
    stable_keys_by_index: dict[int, str]
    signal_detection_indices: set[int]
    selected_signal_indices: set[int]
    signal_results_by_index: dict[int, SignalStateResult]
    analysis_results_by_index: dict[int, AnalysisResult]
    object_crops_by_index: dict[int, ImageArray]
    retired_object_keys: tuple[str, ...]
    selected_narrations: list[Narration]


def _build_ocr_engine(config: PipelineConfig) -> OcrEngine:
    """Build one lazy OCR engine shared by bus, kiosk, and text analyzers."""
    backend = config.ocr_backend.strip().lower()
    if backend in {"none", "disabled"}:
        return UnavailableOcrEngine("ocr_disabled")
    if backend != "rapidocr":
        raise ValueError("ocr_backend must be 'rapidocr' or 'none'")

    params: dict[str, object] = {}
    if config.ocr_model_path is not None and config.ocr_model_path.strip():
        params["Rec.model_path"] = config.ocr_model_path.strip()
    return RapidOcrEngine(
        language=config.ocr_language,
        allow_download=config.allow_ocr_download,
        params=params,
    )


def _build_object_router(
    config: PipelineConfig,
    signal_classifier: SignalStateClassifier | None,
) -> ObjectRouter:
    ocr_engine = _build_ocr_engine(config)
    generic_model = (
        TransformersVisionLanguageModel(
            config.generic_vlm_model,
            device=config.generic_vlm_device,
            allow_download=config.allow_vlm_download,
        )
        if config.generic_vlm_model is not None and config.generic_vlm_model.strip()
        else None
    )
    return ObjectRouter(
        traffic_light_analyzer=TrafficLightAnalyzer(
            classifier=signal_classifier,
            minimum_confirmed_frames=config.min_signal_state_frames,
            minimum_detection_confidence=config.signal_minimum_detection_confidence,
            enabled=config.classify_signal_states,
        ),
        bus_analyzer=BusAnalyzer(
            ocr_engine=ocr_engine,
            motion_window_frames=config.bus_motion_window_frames,
            minimum_detection_confidence=config.bus_minimum_detection_confidence,
            minimum_area_change_ratio=config.bus_minimum_area_change_ratio,
            minimum_direction_consistency=config.bus_minimum_direction_consistency,
            area_jitter_tolerance_ratio=config.bus_area_jitter_tolerance_ratio,
            maximum_motion_frame_gap=config.bus_maximum_motion_frame_gap,
            route_ocr_interval_frames=config.bus_route_ocr_interval_frames,
            route_ocr_requires_relevant_motion=config.bus_route_ocr_requires_relevant_motion,
        ),
        kiosk_analyzer=KioskAnalyzer(
            ocr_engine=ocr_engine,
            ocr_interval_frames=config.kiosk_ocr_interval_frames,
        ),
        text_object_analyzer=TextObjectAnalyzer(
            ocr_engine=ocr_engine,
            ocr_interval_frames=config.text_ocr_interval_frames,
        ),
        generic_vision_analyzer=GenericVisionAnalyzer(
            model=generic_model,
            allowed_object_types=config.generic_vlm_classes,
            minimum_seen_frames_before_inference=config.min_seen_frames,
        ),
    )


def _load_yolo(model_name: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics가 설치되지 않았습니다. `pip install -e .`를 실행하세요."
        ) from exc

    try:
        return YOLO(model_name)
    except Exception as exc:  # model download/version errors vary by runtime
        raise RuntimeError(
            f"모델 {model_name!r}을 불러오지 못했습니다. "
            "`pip install -U ultralytics` 후 다시 실행하세요."
        ) from exc


def _cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _requests_cuda(device: str) -> bool:
    normalized = device.lower()
    if normalized == "cuda" or normalized.startswith("cuda:"):
        return True
    return all(part.strip().isdigit() for part in normalized.split(","))


def resolve_device(
    requested_device: str | None,
    *,
    cuda_available: bool | None = None,
) -> str:
    """Resolve automatic device selection and reject unavailable explicit CUDA."""
    device = requested_device.strip() if requested_device is not None else ""
    if not device:
        available = _cuda_is_available() if cuda_available is None else cuda_available
        return "0" if available else "cpu"

    if _requests_cuda(device):
        available = _cuda_is_available() if cuda_available is None else cuda_available
        if not available:
            raise RuntimeError(
                "CUDA 장치를 사용할 수 없습니다. --device cpu를 지정하거나 --device를 생략하세요."
            )
    return device


def _read_source_fps(capture: cv2.VideoCapture) -> float:
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    except (TypeError, ValueError):
        return 0.0
    return fps if math.isfinite(fps) and fps > 0 else 0.0


def _calculate_frame_timestamp_s(
    capture: cv2.VideoCapture,
    frame_index: int,
    fps: float,
    previous_timestamp_s: float,
) -> float:
    """Return a non-negative, non-decreasing timestamp for the current frame."""
    if math.isfinite(fps) and fps > 0:
        fallback_timestamp_s = frame_index / fps
    else:
        fallback_timestamp_s = 0.0
    if not math.isfinite(fallback_timestamp_s) or fallback_timestamp_s < 0:
        fallback_timestamp_s = 0.0

    try:
        position_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    except (TypeError, ValueError):
        position_msec = math.nan

    position_is_available = (
        math.isfinite(position_msec)
        and position_msec >= 0
        and (frame_index == 0 or position_msec > 0)
    )
    candidate_timestamp_s = (
        position_msec / 1000.0 if position_is_available else fallback_timestamp_s
    )
    safe_previous_timestamp_s = (
        previous_timestamp_s
        if math.isfinite(previous_timestamp_s) and previous_timestamp_s >= 0
        else 0.0
    )
    return max(0.0, safe_previous_timestamp_s, candidate_timestamp_s)


def _build_performance_summary(
    *,
    frames: int,
    elapsed_s: float,
    source_fps: float,
    last_timestamp_s: float | None,
    inference_ms_values: Sequence[float],
) -> dict[str, float | int]:
    """Calculate video-time and processing-time metrics for the final summary."""
    if last_timestamp_s is not None and math.isfinite(last_timestamp_s) and last_timestamp_s > 0:
        video_duration_s = last_timestamp_s
    elif frames > 0 and math.isfinite(source_fps) and source_fps > 0:
        video_duration_s = frames / source_fps
    else:
        video_duration_s = 0.0

    safe_elapsed_s = elapsed_s if math.isfinite(elapsed_s) and elapsed_s > 0 else 0.0
    effective_fps = frames / safe_elapsed_s if safe_elapsed_s > 0 else 0.0
    realtime_factor = safe_elapsed_s / video_duration_s if video_duration_s > 0 else 0.0
    average_inference_ms = (
        sum(inference_ms_values) / len(inference_ms_values) if inference_ms_values else 0.0
    )

    return {
        "video_duration_s": round(video_duration_s, 3),
        "source_fps": round(source_fps, 3),
        "frames": frames,
        "elapsed_s": round(safe_elapsed_s, 3),
        "effective_fps": round(effective_fps, 3),
        "realtime_factor": round(realtime_factor, 3),
        "average_inference_ms": round(average_inference_ms, 3),
    }


def _build_detection_list(result: Any, frame_index: int, timestamp_s: float) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy_values = boxes.xyxy.detach().cpu().tolist()
    confidence_values = boxes.conf.detach().cpu().tolist()
    class_values = boxes.cls.detach().cpu().tolist()
    track_values = (
        boxes.id.detach().cpu().tolist()
        if getattr(boxes, "id", None) is not None
        else [None] * len(boxes)
    )

    names = result.names
    detections: list[Detection] = []
    for xyxy, confidence, class_id, track_id in zip(
        xyxy_values,
        confidence_values,
        class_values,
        track_values,
        strict=True,
    ):
        class_id_int = int(class_id)
        detections.append(
            Detection(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                class_id=class_id_int,
                class_name=str(names[class_id_int]),
                confidence=float(confidence),
                xyxy=tuple(float(value) for value in xyxy),
                track_id=int(track_id) if track_id is not None else None,
            )
        )
    return detections


def _build_detection_payload(
    detection: Detection,
    *,
    stable_object_key: str | None = None,
    is_signal_target: bool | None = None,
    signal_result: SignalStateResult | None = None,
    analysis_result: AnalysisResult | None = None,
) -> dict[str, Any]:
    payload = detection.to_dict()
    if stable_object_key is not None:
        payload["stable_object_key"] = stable_object_key
    if is_signal_target is not None:
        payload["is_signal_target"] = is_signal_target
    if signal_result is not None:
        payload.update(
            {
                "signal_state": signal_result.state.value,
                "signal_state_confidence": round(signal_result.confidence, 6),
                "signal_red_ratio": round(signal_result.red_ratio, 6),
                "signal_green_ratio": round(signal_result.green_ratio, 6),
                "signal_yellow_ratio": round(signal_result.yellow_ratio, 6),
            }
        )
    if analysis_result is not None:
        payload["analysis"] = analysis_result.to_dict()
    return payload


def _safe_filename_component(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "-" for character in value)
    return "-".join(part for part in normalized.split("-") if part) or "object"


def _save_signal_crop(
    crops_dir: Path,
    *,
    crop: ImageArray,
    frame_index: int,
    stable_object_key: str,
    confidence: float,
) -> Path:
    filename = (
        f"frame_{frame_index:06d}__{_safe_filename_component(stable_object_key)}"
        f"__conf_{confidence:.3f}.jpg"
    )
    path = crops_dir / filename
    if not cv2.imwrite(str(path), crop):
        raise RuntimeError(f"신호등 crop을 저장하지 못했습니다: {path}")
    return path


def _bus_overlay_parts(analysis_result: AnalysisResult | None) -> tuple[str, ...]:
    if analysis_result is None or analysis_result.object_type != "bus":
        return ()
    parts: list[str] = []
    state = str(analysis_result.state or "UNKNOWN").strip().upper()
    parts.append(state or "UNKNOWN")
    route_number = analysis_result.attributes.get("route_number")
    if route_number is not None and str(route_number).strip():
        parts.append(f"route={str(route_number).strip()}")
    return tuple(parts)


_BUS_STATE_COLORS = {
    "APPROACHING": (0, 165, 255),
    "STOPPED": (0, 200, 0),
    "RECEDING": (255, 128, 0),
    "UNKNOWN": (128, 128, 128),
}


def _draw_detection(
    frame: ImageArray,
    detection: Detection,
    *,
    stable_object_key: str | None,
    signal_result: SignalStateResult | None,
    analysis_result: AnalysisResult | None = None,
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection.xyxy
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(0, min(width - 1, int(round(x2))))
    bottom = max(0, min(height - 1, int(round(y2))))
    if right <= left or bottom <= top:
        return

    state_colors = {
        SignalState.GREEN: (0, 200, 0),
        SignalState.RED: (0, 0, 255),
        SignalState.YELLOW: (0, 220, 220),
        SignalState.UNKNOWN: (128, 128, 128),
    }
    if signal_result is not None:
        color = state_colors[signal_result.state]
    elif analysis_result is not None and analysis_result.object_type == "bus":
        bus_state = str(analysis_result.state or "UNKNOWN").strip().upper()
        color = _BUS_STATE_COLORS.get(bus_state, _BUS_STATE_COLORS["UNKNOWN"])
    else:
        color = (255, 0, 255)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    label_parts = [detection.class_name, f"{detection.confidence:.2f}"]
    if stable_object_key is not None:
        label_parts.append(stable_object_key.rsplit(":", maxsplit=1)[-1])
    if signal_result is not None:
        label_parts.append(signal_result.state.value)
    label_parts.extend(_bus_overlay_parts(analysis_result))
    label = " ".join(label_parts)
    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_top = max(0, top - text_size[1] - baseline - 4)
    cv2.rectangle(
        frame,
        (left, label_top),
        (min(width - 1, left + text_size[0] + 4), top),
        color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        label,
        (left + 2, max(text_size[1] + 1, top - baseline - 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _frame_timestamp_s(context: FrameContext) -> float:
    candidate = context.captured_at_s
    if candidate is None or not math.isfinite(candidate) or candidate < 0.0:
        candidate = context.received_at_s
    if not math.isfinite(candidate) or candidate < 0.0:
        return 0.0
    return float(candidate)


class VisionSession:
    """Stateful frame processor shared by file and live transports."""

    def __init__(
        self,
        *,
        model: Any,
        track: bool,
        tracker: str,
        predict_kwargs: dict[str, Any],
        event_engine: StableObjectEventEngine,
        signal_target_selector: SignalTargetSelector,
        signal_classifier: SignalStateClassifier | None,
        object_router: ObjectRouter,
        scene_event_manager: SceneEventManager,
        narration_scheduler: NarrationScheduler,
        maximum_state_gap_s: float | None = None,
        model_load_ms: float = 0.0,
    ) -> None:
        if maximum_state_gap_s is not None and (
            not math.isfinite(maximum_state_gap_s) or maximum_state_gap_s <= 0.0
        ):
            raise ValueError("maximum_state_gap_s must be positive when provided")
        self.model = model
        self.track = track
        self.tracker = tracker
        self.predict_kwargs = dict(predict_kwargs)
        self.event_engine = event_engine
        self.signal_target_selector = signal_target_selector
        self.signal_classifier = signal_classifier
        self.object_router = object_router
        self.scene_event_manager = scene_event_manager
        self.narration_scheduler = narration_scheduler
        self.maximum_state_gap_s = maximum_state_gap_s
        self.model_load_ms = model_load_ms
        self._last_processing_started_at_s: float | None = None
        self.processed_frames = 0

    def _reset_tracker(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", ()) if predictor is not None else ()
        for tracker in trackers or ():
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def _reset_state(self) -> None:
        self._reset_tracker()
        self.event_engine.reset()
        self.signal_target_selector.reset()
        self.object_router.reset()
        self.scene_event_manager.reset()
        self.narration_scheduler.reset()

    def reset(self) -> None:
        """Clear tracker, analyzer, event, and narration state for session reuse."""
        self._reset_state()
        self._last_processing_started_at_s = None
        self.processed_frames = 0

    def _reset_after_long_gap(self, processing_started_at_s: float) -> None:
        previous = self._last_processing_started_at_s
        if (
            previous is not None
            and self.maximum_state_gap_s is not None
            and processing_started_at_s - previous > self.maximum_state_gap_s
        ):
            self._reset_state()
        self._last_processing_started_at_s = processing_started_at_s

    def process_frame(
        self,
        frame: ImageArray,
        context: FrameContext,
    ) -> FrameAnalysis:
        """Analyze one decoded frame without owning its transport or persistence."""
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("frame must be a non-empty BGR image")
        if context.processed_index < 0:
            raise ValueError("processed_index must be non-negative")

        self._reset_after_long_gap(context.processing_started_at_s)
        timestamp_s = _frame_timestamp_s(context)
        processing_started_at = time.perf_counter()

        inference_started_at = time.perf_counter()
        if self.track:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker,
                **self.predict_kwargs,
            )
        else:
            results = self.model.predict(frame, **self.predict_kwargs)
        inference_ms = (time.perf_counter() - inference_started_at) * 1000.0

        analysis_started_at = time.perf_counter()
        result = results[0]
        detections = _build_detection_list(
            result,
            context.processed_index,
            timestamp_s,
        )
        signal_detection_indices = {
            index
            for index, detection in enumerate(detections)
            if self.signal_target_selector.is_signal_detection(detection)
        }
        selected_signal_indices = set(self.signal_target_selector.select_indices(detections))
        event_indices = [
            index
            for index in range(len(detections))
            if index not in signal_detection_indices or index in selected_signal_indices
        ]
        event_detections = [detections[index] for index in event_indices]

        signal_results_by_index: dict[int, SignalStateResult] = {}
        object_crops_by_index: dict[int, ImageArray] = {}
        signal_states: list[SignalState | None] = []
        for detection_index in event_indices:
            detection = detections[detection_index]
            crop = crop_frame_to_bbox(frame, detection.xyxy)
            if crop is not None:
                object_crops_by_index[detection_index] = crop
            if detection_index not in selected_signal_indices or self.signal_classifier is None:
                signal_states.append(None)
                continue

            signal_result = (
                SignalStateResult(
                    state=SignalState.UNKNOWN,
                    confidence=0.0,
                    red_ratio=0.0,
                    green_ratio=0.0,
                    yellow_ratio=0.0,
                )
                if crop is None
                else self.signal_classifier.classify(crop)
            )
            signal_results_by_index[detection_index] = signal_result
            signal_states.append(signal_result.state)

        frame_update = self.event_engine.update_frame(
            event_detections,
            timestamp_s,
            signal_states=signal_states,
        )
        scene_events = list(frame_update.events)
        stable_keys_by_index = {
            detection_index: stable_key
            for detection_index, stable_key in zip(
                event_indices,
                frame_update.object_keys,
                strict=True,
            )
        }
        analysis_results_by_index: dict[int, AnalysisResult] = {}
        for detection_index in event_indices:
            stable_object_key = stable_keys_by_index[detection_index]
            analysis_results_by_index[detection_index] = self.object_router.route_detection(
                detections[detection_index],
                stable_id=stable_object_key.rsplit(":", maxsplit=1)[-1],
                crop=object_crops_by_index.get(detection_index),
                precomputed_signal_result=signal_results_by_index.get(detection_index),
            )
        analysis_results = [analysis_results_by_index[index] for index in event_indices]
        analysis_events = self.scene_event_manager.update(
            analysis_results,
            timestamp_s,
            scene_events=scene_events,
        )

        self.narration_scheduler.enqueue(analysis_events, now_s=timestamp_s)
        selected = self.narration_scheduler.pop_next(now_s=timestamp_s)
        selected_narrations = [selected] if selected is not None else []
        narrations = [narration.message for narration in selected_narrations]

        for retired_object_key in frame_update.retired_object_keys:
            retired_stable_id = retired_object_key.rsplit(":", maxsplit=1)[-1]
            self.object_router.reset(retired_stable_id)
            self.scene_event_manager.reset(retired_stable_id)

        analysis_ms = (time.perf_counter() - analysis_started_at) * 1000.0
        total_processing_ms = (time.perf_counter() - processing_started_at) * 1000.0
        self.processed_frames += 1
        return FrameAnalysis(
            source_sequence_id=context.source_sequence_id,
            processed_index=context.processed_index,
            timestamp_s=timestamp_s,
            detections=detections,
            scene_events=scene_events,
            analysis_results=analysis_results,
            analysis_events=analysis_events,
            narrations=narrations,
            timings={
                "inference_ms": inference_ms,
                "analysis_ms": analysis_ms,
                "total_processing_ms": total_processing_ms,
            },
            dropped_frames=context.dropped_frames,
            stable_keys_by_index=stable_keys_by_index,
            signal_detection_indices=signal_detection_indices,
            selected_signal_indices=selected_signal_indices,
            signal_results_by_index=signal_results_by_index,
            analysis_results_by_index=analysis_results_by_index,
            object_crops_by_index=object_crops_by_index,
            retired_object_keys=frame_update.retired_object_keys,
            selected_narrations=selected_narrations,
        )


def create_vision_session(
    config: PipelineConfig,
    *,
    live_mode: bool = False,
    tracker_override: str | None = None,
    maximum_state_gap_s: float | None = None,
    narrate_bus_approach: bool | None = None,
    model: Any | None = None,
    signal_classifier: SignalStateClassifier | None = None,
    object_router: ObjectRouter | None = None,
    scene_event_manager: SceneEventManager | None = None,
    narration_policy: NarrationPolicy | None = None,
    narration_scheduler: NarrationScheduler | None = None,
) -> VisionSession:
    """Build one isolated stateful runtime without opening a video source."""
    resolved_device = resolve_device(config.device)
    model_load_ms = 0.0
    if model is None:
        model_load_started_at = time.perf_counter()
        model = _load_yolo(config.model)
        model_load_ms = (time.perf_counter() - model_load_started_at) * 1000.0

    if not config.classify_signal_states:
        signal_classifier = None
    elif signal_classifier is None:
        signal_classifier = HsvSignalStateClassifier(
            HsvSignalClassifierConfig(
                minimum_color_ratio=config.signal_minimum_color_ratio,
                minimum_score_margin=config.signal_minimum_score_margin,
                minimum_dominance_ratio=config.signal_minimum_dominance_ratio,
            )
        )
    if object_router is None:
        object_router = _build_object_router(config, signal_classifier)

    effective_bus_narration = (
        config.narrate_bus_approach if narrate_bus_approach is None else narrate_bus_approach
    )
    if narration_policy is None:
        narration_policy = NarrationPolicy(
            presence_narration_object_types=config.narration_presence_classes,
            allow_bus_approach=effective_bus_narration,
        )
    if narration_scheduler is None:
        narration_scheduler = NarrationScheduler(
            narration_policy,
            max_queue_size=config.narration_queue_size,
            default_ttl_s=config.narration_ttl_s,
        )
    if scene_event_manager is None:
        scene_event_manager = SceneEventManager(
            auto_presence=False,
            derive_state_changes=True,
            minimum_approach_confidence=narration_policy.minimum_confidence,
            minimum_presence_confidence=narration_policy.minimum_confidence,
            minimum_domain_confidence=narration_policy.minimum_confidence,
        )

    predict_kwargs: dict[str, Any] = {
        "conf": config.confidence,
        "imgsz": config.image_size,
        "device": resolved_device,
        "verbose": False,
    }
    if config.classes is not None:
        predict_kwargs["classes"] = list(config.classes)

    tracker = tracker_override or ("botsort.yaml" if live_mode else config.tracker)
    effective_gap_s = 2.0 if live_mode and maximum_state_gap_s is None else maximum_state_gap_s
    return VisionSession(
        model=model,
        track=config.track,
        tracker=tracker,
        predict_kwargs=predict_kwargs,
        event_engine=StableObjectEventEngine(
            min_seen_frames=config.min_seen_frames,
            max_missed_frames=config.max_missed_frames,
            reconnect_iou_threshold=config.reconnect_iou_threshold,
            max_reconnect_frames=config.max_reconnect_frames,
            min_signal_state_frames=config.min_signal_state_frames,
        ),
        signal_target_selector=SignalTargetSelector(),
        signal_classifier=signal_classifier,
        object_router=object_router,
        scene_event_manager=scene_event_manager,
        narration_scheduler=narration_scheduler,
        maximum_state_gap_s=effective_gap_s,
        model_load_ms=model_load_ms,
    )


def run_video_pipeline(
    config: PipelineConfig,
    *,
    signal_classifier: SignalStateClassifier | None = None,
    object_router: ObjectRouter | None = None,
    scene_event_manager: SceneEventManager | None = None,
    narration_policy: NarrationPolicy | None = None,
) -> dict[str, Any]:
    source_path = Path(config.source)
    if not source_path.exists():
        raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {source_path}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem
    output_video = config.output_dir / f"{stem}_annotated.mp4"
    output_jsonl = config.output_dir / f"{stem}_detections.jsonl"
    crops_dir = config.output_dir / f"{stem}_crops"
    if config.save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    session = create_vision_session(
        config,
        live_mode=False,
        signal_classifier=signal_classifier,
        object_router=object_router,
        scene_event_manager=scene_event_manager,
        narration_policy=narration_policy,
    )
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        session.reset()
        raise RuntimeError(f"영상을 열 수 없습니다: {source_path}")

    source_fps = _read_source_fps(capture)
    processing_fps = source_fps if source_fps > 0 else 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        session.reset()
        raise RuntimeError("영상의 해상도를 읽지 못했습니다.")

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        processing_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        session.reset()
        raise RuntimeError(f"결과 영상을 생성할 수 없습니다: {output_video}")

    frame_index = 0
    detection_count = 0
    event_count = 0
    analysis_event_count = 0
    narration_count = 0
    signal_change_count = 0
    signal_target_count = 0
    signal_state_counts = {state.value: 0 for state in SignalState}
    bus_analysis_count = 0
    bus_detection_frame_count = 0
    bus_approach_event_count = 0
    bus_motion_state_counts = {
        state: 0 for state in ("APPROACHING", "STOPPED", "RECEDING", "UNKNOWN")
    }
    bus_route_numbers: set[str] = set()
    inference_ms_values: list[float] = []
    previous_timestamp_s = 0.0
    last_timestamp_s: float | None = None
    started_at = time.perf_counter()

    try:
        with JsonlWriter(output_jsonl) as jsonl:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                timestamp_s = _calculate_frame_timestamp_s(
                    capture,
                    frame_index,
                    processing_fps,
                    previous_timestamp_s,
                )
                previous_timestamp_s = timestamp_s
                last_timestamp_s = timestamp_s
                received_at_s = time.perf_counter()
                frame_analysis = session.process_frame(
                    frame,
                    FrameContext(
                        source_sequence_id=frame_index,
                        processed_index=frame_index,
                        captured_at_s=timestamp_s,
                        received_at_s=received_at_s,
                        processing_started_at_s=received_at_s,
                    ),
                )

                detections = frame_analysis.detections
                events = frame_analysis.scene_events
                analysis_results = frame_analysis.analysis_results
                analysis_events = frame_analysis.analysis_events
                narrations = frame_analysis.narrations
                inference_ms = frame_analysis.timings["inference_ms"]
                inference_ms_values.append(inference_ms)

                bus_results = [result for result in analysis_results if result.object_type == "bus"]
                if bus_results:
                    bus_detection_frame_count += 1
                bus_analysis_count += len(bus_results)
                for bus_result in bus_results:
                    motion_state = str(bus_result.state or "UNKNOWN").strip().upper()
                    if motion_state not in bus_motion_state_counts:
                        motion_state = "UNKNOWN"
                    bus_motion_state_counts[motion_state] += 1
                    route_number = bus_result.attributes.get("route_number")
                    if route_number is not None and str(route_number).strip():
                        bus_route_numbers.add(str(route_number).strip())

                detection_count += len(detections)
                event_count += len(events)
                analysis_event_count += len(analysis_events)
                narration_count += len(narrations)
                signal_change_count += sum(event.event_type == "signal_changed" for event in events)
                signal_target_count += len(frame_analysis.selected_signal_indices)
                for signal_result in frame_analysis.signal_results_by_index.values():
                    signal_state_counts[signal_result.state.value] += 1
                bus_approach_event_count += sum(
                    event.event_type == OBJECT_APPROACHING and event.object_type == "bus"
                    for event in analysis_events
                )

                if config.save_crops:
                    for detection_index in frame_analysis.selected_signal_indices:
                        crop = frame_analysis.object_crops_by_index.get(detection_index)
                        stable_key = frame_analysis.stable_keys_by_index.get(detection_index)
                        if crop is None or stable_key is None:
                            continue
                        detection = detections[detection_index]
                        _save_signal_crop(
                            crops_dir,
                            crop=crop,
                            frame_index=frame_index,
                            stable_object_key=stable_key,
                            confidence=detection.confidence,
                        )

                jsonl.write(
                    {
                        "frame_index": frame_index,
                        "timestamp_s": timestamp_s,
                        "inference_ms": round(inference_ms, 3),
                        "detections": [
                            _build_detection_payload(
                                detection,
                                stable_object_key=(frame_analysis.stable_keys_by_index.get(index)),
                                is_signal_target=(
                                    index in frame_analysis.selected_signal_indices
                                    if index in frame_analysis.signal_detection_indices
                                    else None
                                ),
                                signal_result=(frame_analysis.signal_results_by_index.get(index)),
                                analysis_result=(
                                    frame_analysis.analysis_results_by_index.get(index)
                                ),
                            )
                            for index, detection in enumerate(detections)
                        ],
                        "events": [event.to_dict() for event in events],
                        "analysis_results": [result.to_dict() for result in analysis_results],
                        "analysis_events": [event.to_dict() for event in analysis_events],
                        "narrations": narrations,
                    }
                )

                for narration in frame_analysis.selected_narrations:
                    print(f"[{narration.event.timestamp_s:7.2f}s] {narration.message}")

                plotted = frame.copy()
                visible_indices = [
                    index
                    for index in range(len(detections))
                    if index not in frame_analysis.signal_detection_indices
                    or index in frame_analysis.selected_signal_indices
                ]
                for detection_index in visible_indices:
                    _draw_detection(
                        plotted,
                        detections[detection_index],
                        stable_object_key=(
                            frame_analysis.stable_keys_by_index.get(detection_index)
                        ),
                        signal_result=(frame_analysis.signal_results_by_index.get(detection_index)),
                        analysis_result=(
                            frame_analysis.analysis_results_by_index.get(detection_index)
                        ),
                    )
                writer.write(plotted)
                frame_index += 1
    finally:
        capture.release()
        writer.release()
        session.reset()

    elapsed_s = time.perf_counter() - started_at
    performance_summary = _build_performance_summary(
        frames=frame_index,
        elapsed_s=elapsed_s,
        source_fps=source_fps,
        last_timestamp_s=last_timestamp_s,
        inference_ms_values=inference_ms_values,
    )

    return {
        "source": str(source_path),
        "model": config.model,
        "classes": list(config.classes) if config.classes is not None else None,
        "tracker": config.tracker if config.track else None,
        "model_load_ms": round(session.model_load_ms, 3),
        "ocr_backend": config.ocr_backend,
        "ocr_language": config.ocr_language,
        "generic_vlm_enabled": bool(
            config.generic_vlm_model is not None and config.generic_vlm_model.strip()
        ),
        "generic_vlm_classes": list(config.generic_vlm_classes),
        **performance_summary,
        "detections": detection_count,
        "events": event_count,
        "analysis_events": analysis_event_count,
        "narrations": narration_count,
        "signal_targets": signal_target_count,
        "signal_changes": signal_change_count,
        "signal_state_counts": signal_state_counts,
        "bus_analysis_results": bus_analysis_count,
        "bus_detection_frames": bus_detection_frame_count,
        "bus_approach_events": bus_approach_event_count,
        "bus_motion_state_counts": bus_motion_state_counts,
        "bus_route_numbers": sorted(bus_route_numbers),
        "output_video": str(output_video),
        "output_jsonl": str(output_jsonl),
        "crops_dir": str(crops_dir) if config.save_crops else None,
    }
