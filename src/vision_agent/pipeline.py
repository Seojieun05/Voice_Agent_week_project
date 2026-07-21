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
from .narration import NarrationPolicy
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
from .types import AnalysisResult, Detection, SignalState
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
        kiosk_analyzer=KioskAnalyzer(ocr_engine=ocr_engine),
        text_object_analyzer=TextObjectAnalyzer(ocr_engine=ocr_engine),
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

    resolved_device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem
    output_video = config.output_dir / f"{stem}_annotated.mp4"
    output_jsonl = config.output_dir / f"{stem}_detections.jsonl"
    crops_dir = config.output_dir / f"{stem}_crops"
    if config.save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    model = _load_yolo(config.model)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {source_path}")

    source_fps = _read_source_fps(capture)
    processing_fps = source_fps if source_fps > 0 else 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("영상의 해상도를 읽지 못했습니다.")

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        processing_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"결과 영상을 생성할 수 없습니다: {output_video}")

    event_engine = StableObjectEventEngine(
        min_seen_frames=config.min_seen_frames,
        max_missed_frames=config.max_missed_frames,
        reconnect_iou_threshold=config.reconnect_iou_threshold,
        max_reconnect_frames=config.max_reconnect_frames,
        min_signal_state_frames=config.min_signal_state_frames,
    )
    signal_target_selector = SignalTargetSelector()
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
    if narration_policy is None:
        narration_policy = NarrationPolicy()
    if scene_event_manager is None:
        scene_event_manager = SceneEventManager(
            auto_presence=False,
            derive_state_changes=True,
            minimum_approach_confidence=narration_policy.minimum_confidence,
            minimum_presence_confidence=narration_policy.minimum_confidence,
            minimum_domain_confidence=narration_policy.minimum_confidence,
        )

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

    predict_kwargs: dict[str, Any] = {
        "conf": config.confidence,
        "imgsz": config.image_size,
        "device": resolved_device,
        "verbose": False,
    }
    if config.classes is not None:
        predict_kwargs["classes"] = list(config.classes)
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
                inference_started_at = time.perf_counter()
                if config.track:
                    results = model.track(
                        frame,
                        persist=True,
                        tracker=config.tracker,
                        **predict_kwargs,
                    )
                else:
                    results = model.predict(frame, **predict_kwargs)
                inference_ms = (time.perf_counter() - inference_started_at) * 1000
                inference_ms_values.append(inference_ms)

                result = results[0]
                detections = _build_detection_list(result, frame_index, timestamp_s)
                selected_signal_indices = set(signal_target_selector.select_indices(detections))
                event_indices = [
                    index
                    for index, detection in enumerate(detections)
                    if detection.class_id != 9 or index in selected_signal_indices
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
                    if detection_index not in selected_signal_indices:
                        signal_states.append(None)
                        continue

                    if signal_classifier is None:
                        signal_states.append(None)
                        continue

                    if crop is None:
                        signal_result = SignalStateResult(
                            state=SignalState.UNKNOWN,
                            confidence=0.0,
                            red_ratio=0.0,
                            green_ratio=0.0,
                            yellow_ratio=0.0,
                        )
                    else:
                        signal_result = signal_classifier.classify(crop)
                    signal_results_by_index[detection_index] = signal_result
                    signal_states.append(signal_result.state)
                    signal_state_counts[signal_result.state.value] += 1

                frame_update = event_engine.update_frame(
                    event_detections,
                    timestamp_s,
                    signal_states=signal_states,
                )
                events = list(frame_update.events)
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
                    analysis_results_by_index[detection_index] = object_router.route_detection(
                        detections[detection_index],
                        stable_id=stable_object_key.rsplit(":", maxsplit=1)[-1],
                        crop=object_crops_by_index.get(detection_index),
                        precomputed_signal_result=signal_results_by_index.get(detection_index),
                    )
                analysis_results = [analysis_results_by_index[index] for index in event_indices]
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
                analysis_events = scene_event_manager.update(
                    analysis_results,
                    timestamp_s,
                    scene_events=events,
                )
                selected_narrations = narration_policy.select(analysis_events)
                narrations = [narration.message for narration in selected_narrations]
                detection_count += len(detections)
                event_count += len(events)
                analysis_event_count += len(analysis_events)
                narration_count += len(narrations)
                signal_change_count += sum(event.event_type == "signal_changed" for event in events)
                signal_target_count += len(selected_signal_indices)
                bus_approach_event_count += sum(
                    event.event_type == OBJECT_APPROACHING and event.object_type == "bus"
                    for event in analysis_events
                )

                if config.save_crops:
                    for detection_index in selected_signal_indices:
                        crop = object_crops_by_index.get(detection_index)
                        if crop is None:
                            continue
                        detection = detections[detection_index]
                        _save_signal_crop(
                            crops_dir,
                            crop=crop,
                            frame_index=frame_index,
                            stable_object_key=stable_keys_by_index[detection_index],
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
                                stable_object_key=stable_keys_by_index.get(index),
                                is_signal_target=(
                                    index in selected_signal_indices
                                    if detection.class_id == 9
                                    else None
                                ),
                                signal_result=signal_results_by_index.get(index),
                                analysis_result=analysis_results_by_index.get(index),
                            )
                            for index, detection in enumerate(detections)
                        ],
                        "events": [event.to_dict() for event in events],
                        "analysis_results": [result.to_dict() for result in analysis_results],
                        "analysis_events": [event.to_dict() for event in analysis_events],
                        "narrations": narrations,
                    }
                )

                for retired_object_key in frame_update.retired_object_keys:
                    retired_stable_id = retired_object_key.rsplit(":", maxsplit=1)[-1]
                    object_router.reset(retired_stable_id)
                    scene_event_manager.reset(retired_stable_id)

                for narration in selected_narrations:
                    print(f"[{narration.event.timestamp_s:7.2f}s] {narration.message}")

                plotted = frame.copy()
                visible_indices = [
                    index
                    for index, detection in enumerate(detections)
                    if detection.class_id != 9 or index in selected_signal_indices
                ]
                for detection_index in visible_indices:
                    _draw_detection(
                        plotted,
                        detections[detection_index],
                        stable_object_key=stable_keys_by_index.get(detection_index),
                        signal_result=signal_results_by_index.get(detection_index),
                        analysis_result=analysis_results_by_index.get(detection_index),
                    )
                writer.write(plotted)
                frame_index += 1
    finally:
        capture.release()
        writer.release()

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
