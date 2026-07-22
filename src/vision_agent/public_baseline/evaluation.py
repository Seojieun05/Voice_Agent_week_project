from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..object_types import AnalyzerKind, SIGNAL_OBJECT_TYPES, object_class_spec
from .dataset import validate_annotation


class EvaluationError(ValueError):
    """Raised when an evaluation input cannot be interpreted safely."""


SIGNAL_STATES = ("RED", "GREEN", "YELLOW")
SIGNAL_GROUND_TRUTH_STATES = (*SIGNAL_STATES, "OFF", "UNKNOWN")
SIGNAL_PREDICTION_STATES = (*SIGNAL_STATES, "UNKNOWN")
BUS_STATES = ("APPROACHING", "STOPPED", "RECEDING", "UNKNOWN")
KIOSK_STAGES = ("ORDER_TYPE_SELECTION", "PAYMENT", "CONFIRMATION", "UNKNOWN")

METRIC_NAMES = (
    "processed_frame_count",
    "effective_fps",
    "realtime_factor",
    "average_yolo_inference_ms",
    "detection_frame_ratio",
    "stable_id_fragmentation_count",
    "event_count",
    "duplicate_event_count",
    "generated_narrations",
    "uncertain_unknown_ratio",
    "signal_state_frame_accuracy",
    "signal_state_accuracy_by_state",
    "signal_confusion_matrix",
    "transition_precision",
    "transition_recall",
    "transition_delay_frames",
    "transition_delay_ms",
    "transition_delay_details",
    "false_green_confirmation_count",
    "duplicate_transition_count",
    "unconfirmed_pedestrian_signal_narration_count",
    "bus_detection_frame_ratio",
    "bus_track_fragmentation_count",
    "bus_approaching_precision",
    "bus_approaching_recall",
    "duplicate_bus_approach_event_count",
    "route_number_exact_match",
    "wrong_confirmed_route_number_count",
    "route_number_confirmation_delay_frames",
    "route_number_confirmation_delay_ms",
    "routed_analyzer_types",
    "screen_stage_accuracy",
    "ocr_exact_match",
    "ocr_normalized_edit_distance",
    "screen_change_precision",
    "screen_change_recall",
    "reverse_vending_as_order_kiosk_count",
    "defective_screen_false_stage_count",
)

_SIGNAL_TYPES = {*SIGNAL_OBJECT_TYPES, "signal", "vehicle_signal"}
_BUS_TYPES = {"bus"}
_KIOSK_TYPES = {
    "kiosk",
    "screen",
    "display",
    "monitor",
    "tv",
    "ticket_machine",
    "reverse_vending_machine",
    "unknown_panel",
}


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"JSON을 읽지 못했습니다: {path}: {exc}") from exc


def _normalize_token(value: object) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def _normalize_state(value: object) -> str:
    if value is None:
        return "UNKNOWN"
    normalized = str(getattr(value, "value", value)).strip().upper()
    return normalized if normalized else "UNKNOWN"


def _finite_number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a public-baseline manifest.

    Full annotation validation belongs to ``annotation.schema.json``. Evaluation
    still rejects ambiguous manifests early so duplicate IDs cannot overwrite a
    report or silently associate predictions with the wrong video.
    """

    manifest_path = Path(path)
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict):
        raise EvaluationError("manifest 최상위 값은 JSON object여야 합니다.")
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise EvaluationError("manifest.videos는 배열이어야 합니다.")

    seen: set[str] = set()
    for index, raw_video in enumerate(videos):
        if not isinstance(raw_video, dict):
            raise EvaluationError(f"manifest.videos[{index}]는 object여야 합니다.")
        video_id = str(raw_video.get("id", raw_video.get("video_id", ""))).strip()
        if not video_id:
            raise EvaluationError(f"manifest.videos[{index}].id가 비어 있습니다.")
        if video_id in seen:
            raise EvaluationError(f"중복 video ID입니다: {video_id}")
        seen.add(video_id)
    return payload


def _load_run_rows(predictions_dir: Path) -> dict[str, dict[str, Any]]:
    summary_path = predictions_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    payload = _read_json(summary_path)
    rows = payload.get("videos", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise EvaluationError("run_summary.json의 videos는 배열이어야 합니다.")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        video_id = str(row.get("video_id", row.get("id", ""))).strip()
        if video_id:
            result[video_id] = row
    return result


def _resolve_annotation(
    manifest_dir: Path,
    video: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, Path | None]:
    video_id = str(video.get("id", video.get("video_id", ""))).strip()
    embedded = video.get("annotation")
    if isinstance(embedded, dict):
        payload = dict(embedded)
        validate_annotation(payload)
        if str(payload.get("video_id", "")).strip() != video_id:
            raise EvaluationError(
                f"annotation video_id 불일치: {video_id!r} != {payload.get('video_id')!r}"
            )
        return payload, None

    raw_path = video.get("annotation_path")
    if raw_path is None and isinstance(embedded, str):
        raw_path = embedded
    path = (
        manifest_dir / str(raw_path)
        if raw_path is not None and str(raw_path).strip()
        else manifest_dir / "annotations" / f"{video_id}.json"
    )
    if not path.exists():
        return None, path
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise EvaluationError(f"annotation은 JSON object여야 합니다: {path}")
    validate_annotation(payload)
    annotation_video_id = str(payload.get("video_id", video_id)).strip()
    if annotation_video_id and annotation_video_id != video_id:
        raise EvaluationError(
            f"annotation video_id 불일치: {video_id!r} != {annotation_video_id!r}"
        )
    objects = payload.get("objects", payload.get("ground_truth_objects", []))
    if objects is not None and not isinstance(objects, list):
        raise EvaluationError(f"annotation.objects는 배열이어야 합니다: {path}")
    return payload, path


def _review_status(annotation: Mapping[str, Any] | None) -> str:
    if annotation is None:
        return "missing"
    raw_status = annotation.get("review_status")
    if raw_status is not None:
        return _normalize_token(raw_status)
    review = annotation.get("review")
    if isinstance(review, Mapping):
        if review.get("reviewed") is True:
            return "reviewed"
        if review.get("needs_manual_review") is True:
            return "needs_manual_review"
    if annotation.get("reviewed") is True:
        return "reviewed"
    return "needs_manual_review"


def _candidate_path(base: Path, value: object) -> list[Path]:
    if value is None or not str(value).strip():
        return []
    path = Path(str(value))
    if path.is_absolute():
        return [path]
    return [path, base / path]


def _resolve_prediction_path(
    predictions_dir: Path,
    video: Mapping[str, Any],
    run_row: Mapping[str, Any],
) -> Path | None:
    raw_status = run_row.get("status")
    if raw_status is not None and _normalize_token(raw_status) not in {
        "success",
        "ok",
        "completed",
    }:
        return None

    candidates: list[Path] = []
    for value in (
        run_row.get("output_jsonl"),
        run_row.get("predictions"),
        video.get("prediction_path"),
    ):
        candidates.extend(_candidate_path(predictions_dir, value))
    video_id = str(video.get("id", video.get("video_id", ""))).strip()
    video_dir = predictions_dir / video_id
    if video_dir.exists():
        candidates.extend(sorted(video_dir.glob("*_detections.jsonl")))
        candidates.extend(sorted(video_dir.glob("*.jsonl")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(f"JSONL {path}:{line_number} 파싱 실패: {exc}") from exc
                if not isinstance(row, dict):
                    raise EvaluationError(f"JSONL {path}:{line_number}는 object여야 합니다.")
                rows.append(row)
    except OSError as exc:
        raise EvaluationError(f"JSONL을 읽지 못했습니다: {path}: {exc}") from exc
    rows.sort(key=lambda row: _integer(row.get("frame_index")) or 0)
    return rows


def _objects(annotation: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if annotation is None:
        return []
    raw_objects = annotation.get("objects", annotation.get("ground_truth_objects", []))
    return [dict(item) for item in raw_objects if isinstance(item, Mapping)]


def _object_family(value: object) -> str:
    normalized = _normalize_token(value)
    if normalized in _SIGNAL_TYPES or "signal" in normalized or normalized == "traffic_light":
        return "signal"
    if normalized in _BUS_TYPES or normalized.endswith("_bus"):
        return "bus"
    if normalized in _KIOSK_TYPES or any(
        token in normalized for token in ("kiosk", "screen", "display", "machine", "panel")
    ):
        return "kiosk"
    return normalized


def _annotation_families(annotation: Mapping[str, Any] | None, category: object) -> set[str]:
    families = {
        _object_family(obj.get("object_type", ""))
        for obj in _objects(annotation)
        if obj.get("object_type")
    }
    category_name = _normalize_token(category)
    if "signal" in category_name or "traffic_light" in category_name:
        families.add("signal")
    if "bus" in category_name:
        families.add("bus")
    if any(token in category_name for token in ("kiosk", "screen", "machine", "display")):
        families.add("kiosk")
    return families


def _frame_index(row: Mapping[str, Any], fallback: int = 0) -> int:
    value = _integer(row.get("frame_index"))
    return fallback if value is None else value


def _frame_analyses(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("analysis_results")
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    analyses: list[dict[str, Any]] = []
    detections = row.get("detections")
    if isinstance(detections, list):
        for detection in detections:
            if isinstance(detection, Mapping) and isinstance(detection.get("analysis"), Mapping):
                analyses.append(dict(detection["analysis"]))
    return analyses


def _all_analyses(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        frame_index = _frame_index(row, row_index)
        for analysis in _frame_analyses(row):
            analysis["_frame_index"] = frame_index
            analysis["_timestamp_s"] = row.get("timestamp_s")
            analyses.append(analysis)
    return analyses


def _all_events(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        raw_events = row.get("analysis_events")
        if not isinstance(raw_events, list):
            raw_events = row.get("events", [])
        if not isinstance(raw_events, list):
            continue
        for event in raw_events:
            if not isinstance(event, Mapping):
                continue
            record = dict(event)
            record["_frame_index"] = _frame_index(row, row_index)
            record["_timestamp_s"] = row.get("timestamp_s")
            events.append(record)
    return events


def _signal_events(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        frame = _frame_index(row, row_index)
        timestamp = row.get("timestamp_s")
        found_analysis_transition = False
        raw_analysis_events = row.get("analysis_events", [])
        if isinstance(raw_analysis_events, list):
            for event in raw_analysis_events:
                if not isinstance(event, Mapping):
                    continue
                if str(event.get("event_type", "")).upper() != "OBJECT_STATE_CHANGED":
                    continue
                if _object_family(event.get("object_type", "")) != "signal":
                    continue
                record = dict(event)
                record["_frame_index"] = frame
                record["_timestamp_s"] = timestamp
                events.append(record)
                found_analysis_transition = True
        if found_analysis_transition:
            continue
        raw_legacy_events = row.get("events", [])
        if not isinstance(raw_legacy_events, list):
            continue
        for event in raw_legacy_events:
            if not isinstance(event, Mapping):
                continue
            if _normalize_token(event.get("event_type", "")) != "signal_changed":
                continue
            record = dict(event)
            record.setdefault("object_type", event.get("class_name", "traffic_light"))
            record.setdefault("stable_id", str(event.get("object_key", "")).split(":")[-1])
            record["_frame_index"] = frame
            record["_timestamp_s"] = timestamp
            events.append(record)
    return events


def _narrations(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    messages: list[str] = []
    for row in rows:
        raw = row.get("narrations", [])
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            messages.extend(str(message) for message in raw if str(message).strip())
    return messages


def _event_signature(event: Mapping[str, Any]) -> tuple[object, ...]:
    attributes = event.get("attributes")
    attrs = attributes if isinstance(attributes, Mapping) else {}
    return (
        str(event.get("event_type", "")).upper(),
        _normalize_token(event.get("object_type", event.get("class_name", ""))),
        str(event.get("stable_id", event.get("object_key", ""))),
        _normalize_state(event.get("previous_state")),
        _normalize_state(event.get("current_state")),
        str(attrs.get("route_number", attrs.get("text", attrs.get("screen_fingerprint", "")))),
    )


def _duplicate_count(items: Iterable[object]) -> int:
    counts = Counter(items)
    return sum(count - 1 for count in counts.values() if count > 1)


def _false_state_episode_count(
    labels: Mapping[int, str],
    predictions: Mapping[int, str],
    predicted_state: str,
) -> int:
    """Count each contiguous false confirmation once, including an initial state."""

    count = 0
    previous_frame: int | None = None
    previous_was_false = False
    for frame in sorted(labels):
        expected = labels[frame]
        is_false = (
            predictions.get(frame, "UNKNOWN") == predicted_state
            and expected in {"RED", "GREEN", "YELLOW", "OFF"}
            and expected != predicted_state
        )
        is_contiguous = previous_frame is not None and frame == previous_frame + 1
        if is_false and (not previous_was_false or not is_contiguous):
            count += 1
        previous_was_false = is_false
        previous_frame = frame
    return count


def _duplicate_transition_count(predictions: Sequence[Mapping[str, Any]]) -> int:
    """Count same-current-state re-emission per stable ID, preserving valid cycles."""

    last_state_by_id: dict[str, str] = {}
    duplicates = 0
    ordered = sorted(predictions, key=lambda event: int(event.get("_frame_index", 0)))
    for event in ordered:
        stable_id = str(event.get("stable_id") or event.get("object_key") or "").strip()
        if not stable_id:
            continue
        current_state = _normalize_state(event.get("current_state"))
        if last_state_by_id.get(stable_id) == current_state:
            duplicates += 1
            continue
        last_state_by_id[stable_id] = current_state
    return duplicates


def _intervals(obj: Mapping[str, Any], *names: str) -> list[dict[str, Any]]:
    for name in names:
        raw = obj.get(name)
        if isinstance(raw, list):
            result: list[dict[str, Any]] = []
            for item in raw:
                if isinstance(item, Mapping):
                    result.append(dict(item))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    result.append({"start_frame": item[0], "end_frame": item[1]})
            return result
    return []


def _interval_bounds(interval: Mapping[str, Any]) -> tuple[int, int] | None:
    start = _integer(interval.get("start_frame", interval.get("start")))
    end = _integer(interval.get("end_frame", interval.get("end")))
    if start is None or end is None or start < 0 or end < start:
        return None
    return start, end


def _visible_frames(objects: Sequence[Mapping[str, Any]], family: str) -> set[int]:
    frames: set[int] = set()
    for obj in objects:
        if _object_family(obj.get("object_type", "")) != family:
            continue
        for interval in _intervals(obj, "visible_frame_ranges", "visible_ranges"):
            bounds = _interval_bounds(interval)
            if bounds is not None:
                frames.update(range(bounds[0], bounds[1] + 1))
    return frames


def _has_overlapping_visible_objects(objects: Sequence[Mapping[str, Any]], family: str) -> bool:
    ranges_by_object: list[list[tuple[int, int]]] = []
    for obj in objects:
        if _object_family(obj.get("object_type", "")) != family:
            continue
        ranges = [
            bounds
            for interval in _intervals(obj, "visible_frame_ranges", "visible_ranges")
            if (bounds := _interval_bounds(interval)) is not None
        ]
        if ranges:
            ranges_by_object.append(ranges)

    for object_index, ranges in enumerate(ranges_by_object):
        for other_ranges in ranges_by_object[object_index + 1 :]:
            if any(
                start <= other_end and other_start <= end
                for start, end in ranges
                for other_start, other_end in other_ranges
            ):
                return True
    return False


def _labeled_frames(
    objects: Sequence[Mapping[str, Any]],
    *,
    family: str,
    interval_names: tuple[str, ...],
    value_names: tuple[str, ...],
) -> dict[int, str]:
    labels: dict[int, str] = {}
    for obj in objects:
        if _object_family(obj.get("object_type", "")) != family:
            continue
        for interval in _intervals(obj, *interval_names):
            bounds = _interval_bounds(interval)
            if bounds is None:
                continue
            raw_value = next((interval.get(name) for name in value_names if name in interval), None)
            state = _normalize_state(raw_value)
            for frame in range(bounds[0], bounds[1] + 1):
                labels[frame] = state
    return labels


def _best_state_by_frame(
    rows: Sequence[Mapping[str, Any]],
    family: str,
) -> dict[int, str]:
    result: dict[int, str] = {}
    for row_index, row in enumerate(rows):
        candidates: list[tuple[float, str]] = []
        for analysis in _frame_analyses(row):
            if _object_family(analysis.get("object_type", "")) != family:
                continue
            confidence = _finite_number(analysis.get("confidence")) or 0.0
            state = (
                "UNKNOWN"
                if analysis.get("is_uncertain") is True
                else _normalize_state(analysis.get("state"))
            )
            candidates.append((confidence, state))
        if not candidates and family == "signal":
            detections = row.get("detections", [])
            if isinstance(detections, list):
                for detection in detections:
                    if not isinstance(detection, Mapping):
                        continue
                    if _object_family(detection.get("class_name", "")) != "signal":
                        continue
                    confidence = _finite_number(detection.get("signal_state_confidence")) or 0.0
                    is_uncertain = any(
                        detection.get(name) is True
                        for name in (
                            "signal_state_is_uncertain",
                            "signal_state_uncertain",
                            "is_uncertain",
                        )
                    )
                    state = (
                        "UNKNOWN"
                        if is_uncertain
                        else _normalize_state(detection.get("signal_state"))
                    )
                    candidates.append((confidence, state))
        if candidates:
            result[_frame_index(row, row_index)] = max(candidates, key=lambda item: item[0])[1]
    return result


def _detected_frames(rows: Sequence[Mapping[str, Any]], family: str | None = None) -> set[int]:
    frames: set[int] = set()
    for row_index, row in enumerate(rows):
        detections = row.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if not isinstance(detection, Mapping):
                continue
            if family is None or _object_family(detection.get("class_name", "")) == family:
                frames.add(_frame_index(row, row_index))
                break
    return frames


def _canonical_stable_id(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    return raw.rsplit(":", 1)[-1]


def _stable_ids_by_family(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        detections = row.get("detections", [])
        if isinstance(detections, list):
            for detection in detections:
                if not isinstance(detection, Mapping):
                    continue
                stable_id = _canonical_stable_id(detection.get("stable_object_key"))
                if stable_id:
                    result[_object_family(detection.get("class_name", ""))].add(stable_id)
        for analysis in _frame_analyses(row):
            stable_id = _canonical_stable_id(analysis.get("stable_id"))
            if stable_id:
                result[_object_family(analysis.get("object_type", ""))].add(stable_id)
    return result


def _metric(
    metrics: dict[str, Any],
    reasons: dict[str, str],
    name: str,
    value: Any,
    reason: str | None = None,
) -> None:
    metrics[name] = value
    if value is None:
        reasons[name] = reason or "metric_unavailable"
    else:
        reasons.pop(name, None)


def _initialize_metrics() -> tuple[dict[str, Any], dict[str, str]]:
    metrics = {name: None for name in METRIC_NAMES}
    reasons = {name: "metric_not_evaluated" for name in METRIC_NAMES}
    return metrics, reasons


def _performance_metrics(
    rows: Sequence[Mapping[str, Any]],
    run_row: Mapping[str, Any],
    metrics: dict[str, Any],
    reasons: dict[str, str],
) -> None:
    processed = len(rows)
    _metric(metrics, reasons, "processed_frame_count", processed)

    elapsed = _finite_number(run_row.get("elapsed_s"))
    effective_fps = _finite_number(run_row.get("effective_fps"))
    if effective_fps is None and elapsed is not None and elapsed > 0:
        effective_fps = processed / elapsed
    _metric(
        metrics,
        reasons,
        "effective_fps",
        round(effective_fps, 6) if effective_fps is not None else None,
        "run_elapsed_time_not_available",
    )

    realtime_factor = _finite_number(run_row.get("realtime_factor"))
    _metric(
        metrics,
        reasons,
        "realtime_factor",
        round(realtime_factor, 6) if realtime_factor is not None else None,
        "video_duration_or_run_elapsed_time_not_available",
    )

    inference_values = [
        value for row in rows if (value := _finite_number(row.get("inference_ms"))) is not None
    ]
    average_inference = _finite_number(
        run_row.get("average_inference_ms", run_row.get("average_yolo_inference_ms"))
    )
    if inference_values:
        average_inference = sum(inference_values) / len(inference_values)
    _metric(
        metrics,
        reasons,
        "average_yolo_inference_ms",
        round(average_inference, 6) if average_inference is not None else None,
        "per_frame_inference_time_not_available",
    )

    detected = len(_detected_frames(rows))
    _metric(
        metrics,
        reasons,
        "detection_frame_ratio",
        round(detected / processed, 6) if processed else None,
        "no_processed_frames",
    )

    events = _all_events(rows)
    _metric(metrics, reasons, "event_count", len(events))
    _metric(
        metrics,
        reasons,
        "duplicate_event_count",
        _duplicate_count(_event_signature(event) for event in events),
    )
    _metric(metrics, reasons, "generated_narrations", _narrations(rows))

    analyses = _all_analyses(rows)
    uncertain = sum(
        analysis.get("is_uncertain") is True or _normalize_state(analysis.get("state")) == "UNKNOWN"
        for analysis in analyses
    )
    _metric(
        metrics,
        reasons,
        "uncertain_unknown_ratio",
        round(uncertain / len(analyses), 6) if analyses else None,
        "no_analysis_results",
    )
    routing_counts = run_row.get(
        "inferred_analyzer_routing_counts",
        run_row.get("analyzer_routing_counts"),
    )
    if isinstance(routing_counts, Mapping):
        routed = sorted(
            str(name)
            for name, count in routing_counts.items()
            if (_finite_number(count) or 0.0) > 0
        )
    else:
        routed = sorted(
            {
                _analyzer_name(analysis.get("object_type", ""))
                for analysis in analyses
                if str(analysis.get("object_type", "")).strip()
            }
        )
    _metric(metrics, reasons, "routed_analyzer_types", routed)


def _analyzer_name(object_type: object) -> str:
    analyzer_kind = object_class_spec(str(object_type)).analyzer
    return {
        AnalyzerKind.TRAFFIC_LIGHT: "TrafficLightAnalyzer",
        AnalyzerKind.BUS: "BusAnalyzer",
        AnalyzerKind.KIOSK: "KioskAnalyzer",
        AnalyzerKind.TEXT: "TextObjectAnalyzer",
        AnalyzerKind.GENERIC: "GenericVisionAnalyzer",
    }[analyzer_kind]


def _fragmentation_metric(
    rows: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
    families: set[str],
) -> int | None:
    expected = Counter(
        _object_family(obj.get("object_type", ""))
        for obj in objects
        if _intervals(obj, "visible_frame_ranges", "visible_ranges")
    )
    if not expected:
        return None
    predicted = _stable_ids_by_family(rows)
    return sum(
        max(0, len(predicted.get(family, set())) - object_count)
        for family, object_count in expected.items()
        if family in families
    )


def _fps(annotation: Mapping[str, Any] | None, run_row: Mapping[str, Any]) -> float | None:
    metadata = annotation.get("video_metadata", {}) if annotation else {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    for value in (
        metadata.get("fps"),
        annotation.get("fps") if annotation else None,
        run_row.get("source_fps"),
    ):
        number = _finite_number(value)
        if number is not None and number > 0:
            return number
    return None


def _gt_transitions(objects: Sequence[Mapping[str, Any]], family: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for obj in objects:
        if _object_family(obj.get("object_type", "")) != family:
            continue
        raw = obj.get("transitions", obj.get("state_transitions", []))
        if not isinstance(raw, list):
            continue
        for transition in raw:
            if not isinstance(transition, Mapping) or transition.get("ambiguous") is True:
                continue
            frame = _integer(transition.get("frame", transition.get("frame_index")))
            if frame is None:
                continue
            result.append(
                {
                    "frame": frame,
                    "previous_state": _normalize_state(
                        transition.get("from_state", transition.get("previous_state"))
                    ),
                    "current_state": _normalize_state(
                        transition.get("to_state", transition.get("current_state"))
                    ),
                }
            )
    return result


def _match_transitions(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    tolerance_frames: int,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    candidates: list[tuple[int, int, int]] = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, prediction in enumerate(predictions):
            if _normalize_state(gt.get("previous_state")) != _normalize_state(
                prediction.get("previous_state")
            ):
                continue
            if _normalize_state(gt.get("current_state")) != _normalize_state(
                prediction.get("current_state")
            ):
                continue
            distance = abs(int(gt["frame"]) - int(prediction["_frame_index"]))
            if distance <= tolerance_frames:
                candidates.append((distance, gt_index, pred_index))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gt_index, pred_index in sorted(candidates):
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_index, pred_index))
    return matches, matched_gt, matched_pred


def _signal_metrics(
    rows: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
    fps: float | None,
    tolerance_frames: int,
    metrics: dict[str, Any],
    reasons: dict[str, str],
) -> None:
    predictions = _signal_events(rows)
    if _has_overlapping_visible_objects(objects, "signal"):
        for name in (
            "signal_state_frame_accuracy",
            "signal_state_accuracy_by_state",
            "signal_confusion_matrix",
            "transition_precision",
            "transition_recall",
            "transition_delay_frames",
            "transition_delay_ms",
            "transition_delay_details",
            "false_green_confirmation_count",
            "unconfirmed_pedestrian_signal_narration_count",
        ):
            _metric(
                metrics,
                reasons,
                name,
                None,
                "multiple_objects_require_prediction_association",
            )
        _metric(
            metrics,
            reasons,
            "duplicate_transition_count",
            _duplicate_transition_count(predictions),
        )
        return

    labels = _labeled_frames(
        objects,
        family="signal",
        interval_names=("signal_state_intervals", "signal_state_segments", "state_intervals"),
        value_names=("state", "signal_state"),
    )
    predictions_by_frame = _best_state_by_frame(rows, "signal")
    if labels:
        correct = sum(
            predictions_by_frame.get(frame, "UNKNOWN") == state for frame, state in labels.items()
        )
        _metric(metrics, reasons, "signal_state_frame_accuracy", round(correct / len(labels), 6))
        accuracy_by_state: dict[str, float | None] = {}
        for state in SIGNAL_GROUND_TRUTH_STATES:
            state_frames = [frame for frame, expected in labels.items() if expected == state]
            accuracy_by_state[state] = (
                round(
                    sum(
                        predictions_by_frame.get(frame, "UNKNOWN") == state
                        for frame in state_frames
                    )
                    / len(state_frames),
                    6,
                )
                if state_frames
                else None
            )
        _metric(metrics, reasons, "signal_state_accuracy_by_state", accuracy_by_state)
        if any(value is None for value in accuracy_by_state.values()):
            reasons["signal_state_accuracy_by_state"] = "states_without_ground_truth_are_null"
        matrix = {
            state: {predicted: 0 for predicted in SIGNAL_PREDICTION_STATES}
            for state in SIGNAL_STATES
        }
        for frame, expected in labels.items():
            if expected not in SIGNAL_STATES:
                continue
            predicted = predictions_by_frame.get(frame, "UNKNOWN")
            if predicted not in SIGNAL_PREDICTION_STATES:
                predicted = "UNKNOWN"
            matrix[expected][predicted] += 1
        _metric(metrics, reasons, "signal_confusion_matrix", matrix)
        _metric(
            metrics,
            reasons,
            "false_green_confirmation_count",
            _false_state_episode_count(labels, predictions_by_frame, "GREEN"),
        )
    else:
        for name in (
            "signal_state_frame_accuracy",
            "signal_state_accuracy_by_state",
            "signal_confusion_matrix",
            "false_green_confirmation_count",
        ):
            _metric(metrics, reasons, name, None, "signal_state_ground_truth_not_available")

    ground_truth = _gt_transitions(objects, "signal")
    matches, _, _ = _match_transitions(ground_truth, predictions, tolerance_frames)
    _metric(
        metrics,
        reasons,
        "transition_precision",
        round(len(matches) / len(predictions), 6) if predictions else None,
        "no_predicted_signal_transitions",
    )
    _metric(
        metrics,
        reasons,
        "transition_recall",
        round(len(matches) / len(ground_truth), 6) if ground_truth else None,
        "no_unambiguous_signal_transitions_in_ground_truth",
    )

    details: list[dict[str, Any]] = []
    for gt_index, prediction_index in matches:
        gt = ground_truth[gt_index]
        prediction = predictions[prediction_index]
        delay_frames = int(prediction["_frame_index"]) - int(gt["frame"])
        details.append(
            {
                "ground_truth_frame": int(gt["frame"]),
                "predicted_frame": int(prediction["_frame_index"]),
                "delay_frames": delay_frames,
                "delay_ms": round(delay_frames / fps * 1000.0, 3) if fps else None,
                "previous_state": gt["previous_state"],
                "current_state": gt["current_state"],
            }
        )
    _metric(
        metrics,
        reasons,
        "transition_delay_details",
        details if matches else None,
        "no_matched_signal_transitions",
    )
    _metric(
        metrics,
        reasons,
        "transition_delay_frames",
        round(sum(item["delay_frames"] for item in details) / len(details), 6) if details else None,
        "no_matched_signal_transitions",
    )
    delay_ms_values = [item["delay_ms"] for item in details if item["delay_ms"] is not None]
    _metric(
        metrics,
        reasons,
        "transition_delay_ms",
        round(sum(delay_ms_values) / len(delay_ms_values), 6) if delay_ms_values else None,
        "no_matched_signal_transitions_or_fps_not_available",
    )

    _metric(
        metrics,
        reasons,
        "duplicate_transition_count",
        _duplicate_transition_count(predictions),
    )

    pedestrian_narrations = 0
    for row in rows:
        subtype_confirmed = any(
            isinstance(analysis.get("attributes"), Mapping)
            and _normalize_token(analysis["attributes"].get("signal_type", ""))
            in {"ped", "pedestrian", "pedestrian_signal"}
            and analysis["attributes"].get("signal_type_is_uncertain") is not True
            for analysis in _frame_analyses(row)
        )
        raw_narrations = row.get("narrations", [])
        if isinstance(raw_narrations, str):
            raw_narrations = [raw_narrations]
        if isinstance(raw_narrations, list) and not subtype_confirmed:
            pedestrian_narrations += sum(
                "보행자 신호" in str(message) for message in raw_narrations
            )
    _metric(
        metrics,
        reasons,
        "unconfirmed_pedestrian_signal_narration_count",
        pedestrian_narrations,
    )


def _binary_precision_recall(
    labels: Mapping[int, str],
    predictions: Mapping[int, str],
    positive: str,
) -> tuple[float | None, float | None]:
    gt_positive = {frame for frame, state in labels.items() if state == positive}
    pred_positive = {frame for frame in labels if predictions.get(frame, "UNKNOWN") == positive}
    true_positive = len(gt_positive & pred_positive)
    precision = true_positive / len(pred_positive) if pred_positive else None
    recall = true_positive / len(gt_positive) if gt_positive else None
    return precision, recall


def _route_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for event in _all_events(rows):
        if str(event.get("event_type", "")).upper() != "TEXT_CONFIRMED":
            continue
        if _object_family(event.get("object_type", "")) != "bus":
            continue
        attributes = event.get("attributes", {})
        if not isinstance(attributes, Mapping):
            continue
        raw_number = attributes.get("route_number")
        if raw_number is None:
            continue
        number = str(raw_number).strip()
        if number:
            key = (str(event.get("stable_id", "")), number)
            if key in seen:
                continue
            seen.add(key)
            records.append({**event, "route_number": number})
    if records:
        return records

    for analysis in _all_analyses(rows):
        if _object_family(analysis.get("object_type", "")) != "bus":
            continue
        attributes = analysis.get("attributes", {})
        if not isinstance(attributes, Mapping):
            continue
        raw_number = attributes.get("route_number")
        if raw_number is None:
            continue
        number = str(raw_number).strip()
        stable_id = str(analysis.get("stable_id", ""))
        key = (stable_id, number)
        if not number or key in seen:
            continue
        seen.add(key)
        records.append({**analysis, "route_number": number})
    return records


def _bus_metrics(
    rows: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
    fps: float | None,
    metrics: dict[str, Any],
    reasons: dict[str, str],
) -> None:
    visible = _visible_frames(objects, "bus")
    bus_detected = _detected_frames(rows, "bus")
    _metric(
        metrics,
        reasons,
        "bus_detection_frame_ratio",
        round(len(visible & bus_detected) / len(visible), 6) if visible else None,
        "bus_visible_frame_ground_truth_not_available",
    )
    bus_objects = [obj for obj in objects if _object_family(obj.get("object_type", "")) == "bus"]
    stable_ids = _stable_ids_by_family(rows).get("bus", set())
    _metric(
        metrics,
        reasons,
        "bus_track_fragmentation_count",
        max(0, len(stable_ids) - len(bus_objects)) if bus_objects else None,
        "bus_object_ground_truth_not_available",
    )

    motion_labels = _labeled_frames(
        objects,
        family="bus",
        interval_names=("motion_intervals", "bus_motion_intervals", "motion_segments"),
        value_names=("state", "motion"),
    )
    motion_predictions = _best_state_by_frame(rows, "bus")
    if _has_overlapping_visible_objects(objects, "bus"):
        for name in ("bus_approaching_precision", "bus_approaching_recall"):
            _metric(
                metrics,
                reasons,
                name,
                None,
                "multiple_objects_require_prediction_association",
            )
    else:
        precision, recall = _binary_precision_recall(
            motion_labels, motion_predictions, "APPROACHING"
        )
        _metric(
            metrics,
            reasons,
            "bus_approaching_precision",
            round(precision, 6) if precision is not None else None,
            "no_predicted_approaching_frames_in_labeled_intervals",
        )
        _metric(
            metrics,
            reasons,
            "bus_approaching_recall",
            round(recall, 6) if recall is not None else None,
            "no_approaching_frames_in_ground_truth",
        )

    approach_events = [
        event
        for event in _all_events(rows)
        if str(event.get("event_type", "")).upper() == "OBJECT_APPROACHING"
        and _object_family(event.get("object_type", "")) == "bus"
    ]
    approach_stable_ids = [
        str(event.get("stable_id") or "").strip()
        for event in approach_events
        if str(event.get("stable_id") or "").strip()
    ]
    _metric(
        metrics,
        reasons,
        "duplicate_bus_approach_event_count",
        _duplicate_count(approach_stable_ids),
    )

    expected_routes = [
        str(obj.get("route_number")).strip()
        for obj in bus_objects
        if obj.get("route_number") is not None and str(obj.get("route_number")).strip()
    ]
    route_records = _route_records(rows)
    predicted_routes = [record["route_number"] for record in route_records]
    if expected_routes and len(bus_objects) > 1:
        for name in (
            "route_number_exact_match",
            "wrong_confirmed_route_number_count",
            "route_number_confirmation_delay_frames",
            "route_number_confirmation_delay_ms",
        ):
            _metric(
                metrics,
                reasons,
                name,
                None,
                "multiple_objects_require_prediction_association",
            )
        return

    if expected_routes:
        remaining = Counter(predicted_routes)
        correct = 0
        for expected in expected_routes:
            if remaining[expected] > 0:
                correct += 1
                remaining[expected] -= 1
        _metric(
            metrics, reasons, "route_number_exact_match", round(correct / len(expected_routes), 6)
        )
        _metric(
            metrics,
            reasons,
            "wrong_confirmed_route_number_count",
            sum(number not in set(expected_routes) for number in predicted_routes),
        )

        start_frames = [
            bounds[0]
            for obj in bus_objects
            if obj.get("route_number") is not None
            for interval in _intervals(obj, "visible_frame_ranges", "visible_ranges")
            if (bounds := _interval_bounds(interval)) is not None
        ]
        matching_records = [
            record for record in route_records if record["route_number"] in set(expected_routes)
        ]
        if start_frames and matching_records:
            confirmation_frame = min(int(record["_frame_index"]) for record in matching_records)
            delay_frames = confirmation_frame - min(start_frames)
            _metric(metrics, reasons, "route_number_confirmation_delay_frames", delay_frames)
            _metric(
                metrics,
                reasons,
                "route_number_confirmation_delay_ms",
                round(delay_frames / fps * 1000.0, 3) if fps else None,
                "video_fps_not_available",
            )
        else:
            _metric(
                metrics,
                reasons,
                "route_number_confirmation_delay_frames",
                None,
                "matching_route_confirmation_or_visible_start_not_available",
            )
            _metric(
                metrics,
                reasons,
                "route_number_confirmation_delay_ms",
                None,
                "matching_route_confirmation_or_video_fps_not_available",
            )
    else:
        for name in (
            "route_number_exact_match",
            "wrong_confirmed_route_number_count",
            "route_number_confirmation_delay_frames",
            "route_number_confirmation_delay_ms",
        ):
            _metric(metrics, reasons, name, None, "route_number_ground_truth_not_available")


def _normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in text if character.isalnum())


def _levenshtein(first: str, second: str) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for first_index, first_character in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_character in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1] + (first_character != second_character),
                )
            )
        previous = current
    return previous[-1]


def _ground_truth_texts(objects: Sequence[Mapping[str, Any]]) -> list[str]:
    texts: list[str] = []
    for obj in objects:
        if _object_family(obj.get("object_type", "")) != "kiosk":
            continue
        raw = obj.get("text_annotations", [])
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, str) and item.strip():
                texts.append(item.strip())
            elif isinstance(item, Mapping):
                if item.get("ambiguous") is True:
                    continue
                for name in ("text", "value", "content"):
                    if item.get(name) is not None and str(item[name]).strip():
                        texts.append(str(item[name]).strip())
                        break
    return texts


def _predicted_texts(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    texts: list[str] = []
    for event in _all_events(rows):
        if str(event.get("event_type", "")).upper() != "TEXT_CONFIRMED":
            continue
        if _object_family(event.get("object_type", "")) != "kiosk":
            continue
        attributes = event.get("attributes", {})
        if isinstance(attributes, Mapping) and attributes.get("text"):
            texts.append(str(attributes["text"]))
    for analysis in _all_analyses(rows):
        if _object_family(analysis.get("object_type", "")) != "kiosk":
            continue
        attributes = analysis.get("attributes", {})
        if not isinstance(attributes, Mapping) or attributes.get("screen_is_confirmed") is not True:
            continue
        visible = attributes.get("visible_text", [])
        if isinstance(visible, str):
            visible = [visible]
        if isinstance(visible, list):
            texts.extend(str(value) for value in visible if str(value).strip())
    return texts


def _derived_screen_changes(objects: Sequence[Mapping[str, Any]]) -> list[int]:
    frames: list[int] = []
    for obj in objects:
        if _object_family(obj.get("object_type", "")) != "kiosk":
            continue
        intervals = _intervals(obj, "screen_stage_intervals", "stage_intervals")
        ordered: list[tuple[int, int, str]] = []
        for interval in intervals:
            bounds = _interval_bounds(interval)
            if bounds is not None:
                ordered.append(
                    (
                        bounds[0],
                        bounds[1],
                        _normalize_state(interval.get("stage", interval.get("state"))),
                    )
                )
        ordered.sort()
        for previous, current in zip(ordered, ordered[1:]):
            if previous[2] != current[2]:
                frames.append(current[0])
    return frames


def _match_frames(expected: Sequence[int], predicted: Sequence[int], tolerance_frames: int) -> int:
    candidates = sorted(
        (abs(gt - pred), gt_index, pred_index)
        for gt_index, gt in enumerate(expected)
        for pred_index, pred in enumerate(predicted)
        if abs(gt - pred) <= tolerance_frames
    )
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for _, gt_index, pred_index in candidates:
        if gt_index not in matched_gt and pred_index not in matched_pred:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
    return len(matched_gt)


def _kiosk_metrics(
    video_id: str,
    rows: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
    tolerance_frames: int,
    metrics: dict[str, Any],
    reasons: dict[str, str],
) -> None:
    labels = _labeled_frames(
        objects,
        family="kiosk",
        interval_names=("screen_stage_intervals", "stage_intervals"),
        value_names=("stage", "state"),
    )
    predictions = _best_state_by_frame(rows, "kiosk")
    if labels:
        correct = sum(predictions.get(frame, "UNKNOWN") == stage for frame, stage in labels.items())
        _metric(metrics, reasons, "screen_stage_accuracy", round(correct / len(labels), 6))
    else:
        _metric(
            metrics,
            reasons,
            "screen_stage_accuracy",
            None,
            "screen_stage_ground_truth_not_available",
        )

    expected_texts = [_normalize_text(value) for value in _ground_truth_texts(objects)]
    predicted_texts = [_normalize_text(value) for value in _predicted_texts(rows)]
    expected_texts = [value for value in expected_texts if value]
    predicted_texts = [value for value in predicted_texts if value]
    if expected_texts:
        exact = float(set(expected_texts) == set(predicted_texts))
        _metric(metrics, reasons, "ocr_exact_match", exact)
        distances = [
            min(
                (_levenshtein(expected, predicted) / max(len(expected), len(predicted), 1))
                for predicted in predicted_texts
            )
            if predicted_texts
            else 1.0
            for expected in expected_texts
        ]
        _metric(
            metrics,
            reasons,
            "ocr_normalized_edit_distance",
            round(sum(distances) / len(distances), 6),
        )
    else:
        _metric(metrics, reasons, "ocr_exact_match", None, "ocr_text_ground_truth_not_available")
        _metric(
            metrics,
            reasons,
            "ocr_normalized_edit_distance",
            None,
            "ocr_text_ground_truth_not_available",
        )

    gt_changes = _derived_screen_changes(objects)
    pred_changes = [
        int(event["_frame_index"])
        for event in _all_events(rows)
        if str(event.get("event_type", "")).upper() == "SCREEN_CHANGED"
        and _object_family(event.get("object_type", "")) == "kiosk"
    ]
    matched = _match_frames(gt_changes, pred_changes, tolerance_frames)
    _metric(
        metrics,
        reasons,
        "screen_change_precision",
        round(matched / len(pred_changes), 6) if pred_changes else None,
        "no_predicted_screen_changes",
    )
    _metric(
        metrics,
        reasons,
        "screen_change_recall",
        round(matched / len(gt_changes), 6) if gt_changes else None,
        "no_screen_changes_in_ground_truth",
    )

    analyses = _all_analyses(rows)
    reverse_vending = "reverse_vending" in _normalize_token(video_id) or any(
        _normalize_token(obj.get("object_type", "")) == "reverse_vending_machine" for obj in objects
    )
    reverse_false_ids = {
        str(analysis.get("stable_id", ""))
        for analysis in analyses
        if reverse_vending
        and _normalize_state(analysis.get("state")) == "ORDER_TYPE_SELECTION"
        and analysis.get("is_uncertain") is not True
    }
    _metric(
        metrics,
        reasons,
        "reverse_vending_as_order_kiosk_count",
        len(reverse_false_ids) if reverse_vending else None,
        "metric_not_applicable_for_video",
    )

    defective = "defective_screen" in _normalize_token(video_id) or any(
        obj.get("is_defective") is True for obj in objects
    )
    false_stages = {
        (str(analysis.get("stable_id", "")), _normalize_state(analysis.get("state")))
        for analysis in analyses
        if defective
        and _normalize_state(analysis.get("state")) in {"PAYMENT", "CONFIRMATION"}
        and analysis.get("is_uncertain") is not True
    }
    _metric(
        metrics,
        reasons,
        "defective_screen_false_stage_count",
        len(false_stages) if defective else None,
        "metric_not_applicable_for_video",
    )


def _narration_constraints(annotation: Mapping[str, Any] | None) -> dict[str, Any]:
    if annotation is None:
        return {}
    raw = annotation.get("narration_constraints", annotation.get("safety_constraints", {}))
    return dict(raw) if isinstance(raw, Mapping) else {}


def _safety_constraints(
    video_id: str,
    rows: Sequence[Mapping[str, Any]],
    annotation: Mapping[str, Any] | None,
    *,
    prediction_available: bool = True,
) -> list[dict[str, Any]]:
    if not prediction_available:
        return [
            {
                "name": "prediction_based_safety_constraints",
                "status": "NOT_EVALUATED",
                "observed": None,
                "maximum": None,
                "reason": "prediction_not_available",
            }
        ]

    normalized_id = _normalize_token(video_id)
    narrations = _narrations(rows)
    signal_events = _signal_events(rows)
    all_events = _all_events(rows)
    constraints: list[dict[str, Any]] = []

    def add(name: str, observed: int, maximum: int) -> None:
        constraints.append(
            {
                "name": name,
                "status": "PASS" if observed <= maximum else "FAIL",
                "observed": observed,
                "maximum": maximum,
            }
        )

    if "signal_yellow_flicker_vertical" in normalized_id:
        red_green = sum(
            _normalize_state(event.get("current_state")) in {"RED", "GREEN"}
            for event in signal_events
        )
        add("no_confident_red_or_green_transition", red_green, 0)

    if "bus_london_pulls_in" in normalized_id:
        approaches = sum(
            str(event.get("event_type", "")).upper() == "OBJECT_APPROACHING"
            and _object_family(event.get("object_type", "")) == "bus"
            for event in all_events
        )
        add("at_most_one_bus_approach_event", approaches, 1)
        add("do_not_confirm_unreadable_route_number", len(_route_records(rows)), 0)

    if "bus_waiting_multiple_arrivals" in normalized_id:
        approach_events = [
            event
            for event in all_events
            if str(event.get("event_type", "")).upper() == "OBJECT_APPROACHING"
            and _object_family(event.get("object_type", "")) == "bus"
        ]
        approach_stable_ids = [
            str(event.get("stable_id") or "").strip()
            for event in approach_events
            if str(event.get("stable_id") or "").strip()
        ]
        duplicate_approaches = _duplicate_count(approach_stable_ids)
        maximum = _integer(_narration_constraints(annotation).get("maximum_duplicate_events"))
        add("duplicate_bus_approach_event_limit", duplicate_approaches, maximum or 0)
        stable_ids = sorted(_stable_ids_by_family(rows).get("bus", set()))
        constraints.append(
            {
                "name": "bus_stable_id_lifecycle_observation",
                "status": "OBSERVED",
                "observed": stable_ids,
                "maximum": None,
                "reason": "distinct_arrivals_require_manual_track_association",
            }
        )

    if "kiosk_like_reverse_vending_machine" in normalized_id:
        false_ids = {
            str(analysis.get("stable_id", ""))
            for analysis in _all_analyses(rows)
            if _normalize_state(analysis.get("state")) == "ORDER_TYPE_SELECTION"
            and analysis.get("is_uncertain") is not True
        }
        add("do_not_classify_reverse_vending_as_order_kiosk", len(false_ids), 0)

    if "ticket_machine_defective_screen" in normalized_id:
        false_stages = {
            (str(analysis.get("stable_id", "")), _normalize_state(analysis.get("state")))
            for analysis in _all_analyses(rows)
            if _normalize_state(analysis.get("state")) in {"PAYMENT", "CONFIRMATION"}
            and analysis.get("is_uncertain") is not True
        }
        false_narrations = sum(
            any(
                token in message.casefold() for token in ("payment", "confirmation", "결제", "확인")
            )
            for message in narrations
        )
        add(
            "do_not_confirm_payment_or_confirmation_on_defective_screen",
            len(false_stages) + false_narrations,
            0,
        )

    narration_rules = _narration_constraints(annotation)
    phrases = narration_rules.get("forbidden_phrases", [])
    if not isinstance(phrases, list):
        phrases = []
    phrase_violations = sum(
        str(phrase).casefold() in message.casefold()
        for phrase in phrases
        for message in narrations
        if str(phrase)
    )
    patterns = narration_rules.get("forbidden_patterns", [])
    if not isinstance(patterns, list):
        patterns = []
    pattern_violations = 0
    for pattern in patterns:
        try:
            compiled = re.compile(str(pattern), re.IGNORECASE)
        except re.error:
            compiled = re.compile(re.escape(str(pattern)), re.IGNORECASE)
        pattern_violations += sum(compiled.search(message) is not None for message in narrations)
    if phrases or patterns:
        add("forbidden_narration", phrase_violations + pattern_violations, 0)
    return constraints


def _qualitative_summary(
    rows: Sequence[Mapping[str, Any]],
    prediction_path: Path | None,
) -> dict[str, Any]:
    analyses = _all_analyses(rows)
    return {
        "prediction_available": prediction_path is not None,
        "routed_analyzer_types": sorted(
            {
                _normalize_token(analysis.get("object_type", ""))
                for analysis in analyses
                if str(analysis.get("object_type", "")).strip()
            }
        ),
        "observed_states": dict(
            sorted(
                Counter(_normalize_state(analysis.get("state")) for analysis in analyses).items()
            )
        ),
        "confirmed_route_numbers": sorted(
            {record["route_number"] for record in _route_records(rows)}
        ),
        "narrations": _narrations(rows),
    }


def evaluate_video(
    *,
    video: Mapping[str, Any],
    annotation: Mapping[str, Any] | None,
    annotation_path: Path | None,
    prediction_path: Path | None,
    run_row: Mapping[str, Any],
    transition_tolerance_frames: int,
) -> dict[str, Any]:
    video_id = str(video.get("id", video.get("video_id", ""))).strip()
    category = str(video.get("category", "unknown"))
    review_status = _review_status(annotation)
    reviewed = review_status == "reviewed"
    raw_run_status = run_row.get("status")
    normalized_run_status = _normalize_token(raw_run_status) if raw_run_status is not None else ""
    run_failed = bool(normalized_run_status) and normalized_run_status not in {
        "success",
        "ok",
        "completed",
    }
    prediction_available = prediction_path is not None and not run_failed
    rows = _load_jsonl(prediction_path) if prediction_available else []
    metrics, reasons = _initialize_metrics()
    if prediction_available:
        _performance_metrics(rows, run_row, metrics, reasons)
    else:
        reasons = {name: "prediction_not_available" for name in METRIC_NAMES}

    objects = _objects(annotation)
    families = _annotation_families(annotation, category)
    if reviewed and prediction_available:
        fragmentation = _fragmentation_metric(rows, objects, families)
        _metric(
            metrics,
            reasons,
            "stable_id_fragmentation_count",
            fragmentation,
            "visible_object_ground_truth_not_available",
        )
        video_fps = _fps(annotation, run_row)
        if "signal" in families:
            _signal_metrics(
                rows,
                objects,
                video_fps,
                transition_tolerance_frames,
                metrics,
                reasons,
            )
        if "bus" in families:
            _bus_metrics(rows, objects, video_fps, metrics, reasons)
        if "kiosk" in families:
            _kiosk_metrics(
                video_id,
                rows,
                objects,
                transition_tolerance_frames,
                metrics,
                reasons,
            )
        for name in METRIC_NAMES:
            if metrics[name] is None and reasons[name] == "metric_not_evaluated":
                reasons[name] = "metric_not_applicable_for_video"
    elif prediction_available:
        for name in METRIC_NAMES:
            if name in {
                "processed_frame_count",
                "effective_fps",
                "realtime_factor",
                "average_yolo_inference_ms",
                "detection_frame_ratio",
                "event_count",
                "duplicate_event_count",
                "generated_narrations",
                "uncertain_unknown_ratio",
                "routed_analyzer_types",
            }:
                continue
            _metric(metrics, reasons, name, None, "annotation_not_reviewed")

    status = "ok" if prediction_available else "prediction_missing"
    if run_failed:
        status = str(raw_run_status)
    safety = _safety_constraints(
        video_id, rows, annotation, prediction_available=prediction_available
    )
    return {
        "schema_version": "1.0",
        "video_id": video_id,
        "category": category,
        "status": status,
        "review_status": review_status,
        "quantitative_evaluation": reviewed and prediction_available,
        "annotation_path": str(annotation_path) if annotation_path is not None else None,
        "prediction_jsonl": str(prediction_path) if prediction_path is not None else None,
        "metrics": metrics,
        "metric_reasons": reasons,
        "qualitative": _qualitative_summary(
            rows, prediction_path if prediction_available else None
        ),
        "safety_constraints": safety,
        "safety_failure_count": sum(item.get("status") == "FAIL" for item in safety),
        "limitations": [
            "stable ID fragmentation is estimated by object-family counts; no box-level GT association is available",
            "multiple ground-truth buses cannot be associated with predicted routes without track-level GT links",
            "only non-ambiguous, human-reviewed intervals and transitions are scored",
        ],
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report['video_id']} evaluation",
        "",
        f"- category: `{report['category']}`",
        f"- status: `{report['status']}`",
        f"- review status: `{report['review_status']}`",
        f"- quantitative evaluation: `{str(report['quantitative_evaluation']).lower()}`",
        "",
        "## Metrics",
        "",
        "| metric | value | unavailable reason |",
        "|---|---:|---|",
    ]
    metrics = report["metrics"]
    reasons = report["metric_reasons"]
    for name in METRIC_NAMES:
        value = metrics[name]
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        lines.append(f"| `{name}` | `{rendered}` | {reasons.get(name, '')} |")
    lines.extend(["", "## Safety constraints", ""])
    safety = report["safety_constraints"]
    if safety:
        lines.extend(["| constraint | status | observed | maximum |", "|---|---|---:|---:|"])
        for item in safety:
            lines.append(
                f"| `{item['name']}` | `{item['status']}` | "
                f"`{json.dumps(item.get('observed'), ensure_ascii=False)}` | "
                f"`{json.dumps(item.get('maximum'), ensure_ascii=False)}` |"
            )
    else:
        lines.append("No video-specific safety constraint was configured.")
    lines.extend(["", "## Qualitative observations", "", "```json"])
    lines.append(json.dumps(report["qualitative"], ensure_ascii=False, indent=2, sort_keys=True))
    lines.extend(["```", ""])
    return "\n".join(lines)


def _summary_row(report: Mapping[str, Any], json_path: Path, markdown_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "video_id": report["video_id"],
        "category": report["category"],
        "status": report["status"],
        "review_status": report["review_status"],
        "quantitative_evaluation": report["quantitative_evaluation"],
        "prediction_jsonl": report["prediction_jsonl"],
        "report_json": str(json_path),
        "report_markdown": str(markdown_path),
        "safety_failure_count": report["safety_failure_count"],
    }
    metrics = report["metrics"]
    reasons = report["metric_reasons"]
    for name in METRIC_NAMES:
        row[name] = metrics[name]
        row[f"{name}_reason"] = reasons.get(name)
    return row


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return str(value).lower()
    return value


def evaluate_public_baseline(
    manifest_path: str | Path,
    predictions_dir: str | Path,
    output_dir: str | Path,
    *,
    transition_tolerance_frames: int = 30,
) -> dict[str, Any]:
    """Evaluate all manifest videos without turning predictions into ground truth."""

    if transition_tolerance_frames < 0:
        raise EvaluationError("transition_tolerance_frames는 0 이상이어야 합니다.")
    manifest_path = Path(manifest_path).resolve()
    predictions_dir = Path(predictions_dir).resolve()
    output_dir = Path(output_dir).resolve()
    manifest = load_manifest(manifest_path)
    run_rows = _load_run_rows(predictions_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for raw_video in manifest["videos"]:
        video = dict(raw_video)
        video_id = str(video.get("id", video.get("video_id", ""))).strip()
        run_row = run_rows.get(video_id, {})
        try:
            annotation, annotation_path = _resolve_annotation(manifest_path.parent, video)
            prediction_path = _resolve_prediction_path(predictions_dir, video, run_row)
            report = evaluate_video(
                video=video,
                annotation=annotation,
                annotation_path=annotation_path,
                prediction_path=prediction_path,
                run_row=run_row,
                transition_tolerance_frames=transition_tolerance_frames,
            )
        except (EvaluationError, OSError, ValueError) as exc:
            metrics, reasons = _initialize_metrics()
            reasons = {name: "video_evaluation_failed" for name in METRIC_NAMES}
            report = {
                "schema_version": "1.0",
                "video_id": video_id,
                "category": str(video.get("category", "unknown")),
                "status": "evaluation_failed",
                "review_status": "unknown",
                "quantitative_evaluation": False,
                "annotation_path": None,
                "prediction_jsonl": None,
                "metrics": metrics,
                "metric_reasons": reasons,
                "qualitative": {"error": str(exc)},
                "safety_constraints": [],
                "safety_failure_count": 0,
                "limitations": ["evaluation failed before metrics could be computed"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

        json_path = reports_dir / f"{video_id}.json"
        markdown_path = reports_dir / f"{video_id}.md"
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(_markdown_report(report), encoding="utf-8")
        summary_rows.append(_summary_row(report, json_path, markdown_path))

    summary = {
        "schema_version": "1.0",
        "manifest": str(manifest_path),
        "predictions": str(predictions_dir),
        "video_count": len(summary_rows),
        "quantitatively_evaluated_video_count": sum(
            row["quantitative_evaluation"] is True for row in summary_rows
        ),
        "qualitative_only_video_count": sum(
            row["quantitative_evaluation"] is not True for row in summary_rows
        ),
        "failed_video_count": sum(row["status"] == "evaluation_failed" for row in summary_rows),
        "videos": summary_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = (
        list(summary_rows[0])
        if summary_rows
        else [
            "video_id",
            "category",
            "status",
            "review_status",
            "quantitative_evaluation",
        ]
    )
    with summary_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
    return summary
