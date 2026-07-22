"""Dataset helpers for the manually reviewed public baseline.

This module deliberately validates structure only.  It never promotes model
predictions to ground truth or infers labels from video pixels.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = "1.0"
ANNOTATION_SCHEMA_VERSION = "1.0"
REVIEW_STATUSES = frozenset({"needs_manual_review", "reviewed"})
SIGNAL_STATES = frozenset({"RED", "GREEN", "YELLOW", "OFF", "UNKNOWN"})
MOTION_STATES = frozenset({"APPROACHING", "STOPPED", "RECEDING", "UNKNOWN"})
SCREEN_STAGES = frozenset({"ORDER_TYPE_SELECTION", "PAYMENT", "CONFIRMATION", "UNKNOWN"})


class DatasetValidationError(ValueError):
    """Raised when a public-baseline JSON document is structurally invalid."""


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetValidationError(f"Expected a JSON object in {path}")
    return payload


def _require_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{location} must be a non-empty string")
    return value


def _require_relative_path(value: object, location: str) -> str:
    raw = _require_string(value, location)
    if Path(raw).is_absolute():
        raise DatasetValidationError(f"{location} must be relative to the manifest")
    return raw


def _require_keys(value: Mapping[str, object], required: set[str], location: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise DatasetValidationError(f"{location} is missing required fields: {', '.join(missing)}")


def _reject_extra_keys(value: Mapping[str, object], allowed: set[str], location: str) -> None:
    extras = sorted(set(value) - allowed)
    if extras:
        raise DatasetValidationError(f"{location} has unsupported fields: {', '.join(extras)}")


def validate_manifest(payload: Mapping[str, object]) -> None:
    """Validate the canonical public-baseline manifest structure."""

    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise DatasetValidationError(f"schema_version must be {MANIFEST_SCHEMA_VERSION!r}")
    _require_string(payload.get("dataset_id"), "dataset_id")
    _require_relative_path(payload.get("source_catalog"), "source_catalog")
    videos = payload.get("videos")
    if not isinstance(videos, list) or not videos:
        raise DatasetValidationError("videos must be a non-empty array")

    seen_ids: set[str] = set()
    for index, item in enumerate(videos):
        location = f"videos[{index}]"
        if not isinstance(item, dict):
            raise DatasetValidationError(f"{location} must be an object")
        video_id = _require_string(item.get("id"), f"{location}.id")
        if video_id in seen_ids:
            raise DatasetValidationError(f"duplicate video id: {video_id}")
        seen_ids.add(video_id)
        _require_string(item.get("category"), f"{location}.category")
        candidates = item.get("video_paths")
        if not isinstance(candidates, list) or not candidates:
            raise DatasetValidationError(f"{location}.video_paths must be non-empty")
        for path_index, candidate in enumerate(candidates):
            _require_relative_path(candidate, f"{location}.video_paths[{path_index}]")
        _require_relative_path(item.get("annotation_path"), f"{location}.annotation_path")


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a public-baseline manifest."""

    manifest_path = Path(path)
    payload = _read_json_object(manifest_path)
    validate_manifest(payload)
    return payload


def resolve_manifest_path(manifest_path: str | Path, relative_path: str) -> Path:
    """Resolve a path stored relative to the manifest directory."""

    if Path(relative_path).is_absolute():
        raise DatasetValidationError("manifest paths must be relative")
    return (Path(manifest_path).resolve().parent / relative_path).resolve()


def resolve_video_path(
    manifest_path: str | Path,
    video: Mapping[str, object],
    *,
    require_exists: bool = True,
) -> Path | None:
    """Return the first existing video candidate (standardized MP4 comes first).

    When ``require_exists`` is false and no candidate exists, the first candidate
    is returned so callers can report its expected location.  ``None`` is only
    returned for malformed/empty candidates in this non-strict mode.
    """

    candidates = video.get("video_paths")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        if require_exists:
            raise DatasetValidationError("video_paths must be an array")
        return None

    resolved: list[Path] = []
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            if require_exists:
                raise DatasetValidationError("video_paths entries must be strings")
            continue
        path = resolve_manifest_path(manifest_path, candidate)
        resolved.append(path)
        if path.is_file():
            return path

    if require_exists:
        video_id = video.get("id", "<unknown>")
        searched = ", ".join(str(path) for path in resolved) or "<none>"
        raise FileNotFoundError(f"No video file found for {video_id}; searched: {searched}")
    return resolved[0] if resolved else None


def resolve_annotation_path(manifest_path: str | Path, video: Mapping[str, object]) -> Path:
    """Resolve one video's canonical annotation path."""

    relative_path = _require_relative_path(video.get("annotation_path"), "annotation_path")
    return resolve_manifest_path(manifest_path, relative_path)


def _validate_nullable_number(value: object, location: str, *, integer: bool = False) -> None:
    if value is None:
        return
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected):
        kind = "integer" if integer else "number"
        raise DatasetValidationError(f"{location} must be null or a positive {kind}")
    if not math.isfinite(float(value)) or value <= 0:
        raise DatasetValidationError(f"{location} must be null or a positive number")


def _validate_frame_range(value: object, location: str) -> None:
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"{location} must be an object")
    _require_keys(value, {"start_frame", "end_frame"}, location)
    start = value.get("start_frame")
    end = value.get("end_frame")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end < start
    ):
        raise DatasetValidationError(f"{location} requires 0 <= start_frame <= end_frame")


def _validate_enum_intervals(
    value: object,
    location: str,
    *,
    field_name: str,
    allowed: frozenset[str],
) -> None:
    if not isinstance(value, list):
        raise DatasetValidationError(f"{location} must be an array")
    for index, interval in enumerate(value):
        item_location = f"{location}[{index}]"
        if not isinstance(interval, Mapping):
            raise DatasetValidationError(f"{item_location} must be an object")
        interval_fields = {"start_frame", "end_frame", field_name}
        _require_keys(interval, interval_fields, item_location)
        _reject_extra_keys(interval, interval_fields, item_location)
        _validate_frame_range(interval, item_location)
        if interval.get(field_name) not in allowed:
            raise DatasetValidationError(
                f"{item_location}.{field_name} must be one of {sorted(allowed)}"
            )


def validate_annotation(payload: Mapping[str, object]) -> None:
    """Validate an annotation against ``annotation.schema.json`` constraints."""

    required_fields = {
        "schema_version",
        "video_id",
        "review_status",
        "video_metadata",
        "objects",
        "narration_constraints",
    }
    _require_keys(payload, required_fields, "annotation")
    _reject_extra_keys(payload, required_fields | {"review_notes"}, "annotation")

    if payload.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise DatasetValidationError(f"schema_version must be {ANNOTATION_SCHEMA_VERSION!r}")
    _require_string(payload.get("video_id"), "video_id")
    if payload.get("review_status") not in REVIEW_STATUSES:
        raise DatasetValidationError(f"review_status must be one of {sorted(REVIEW_STATUSES)}")

    metadata = payload.get("video_metadata")
    if not isinstance(metadata, Mapping):
        raise DatasetValidationError("video_metadata must be an object")
    _require_keys(metadata, {"fps", "frame_count", "width", "height"}, "video_metadata")
    _reject_extra_keys(metadata, {"fps", "frame_count", "width", "height"}, "video_metadata")
    _validate_nullable_number(metadata.get("fps"), "video_metadata.fps")
    for field_name in ("frame_count", "width", "height"):
        _validate_nullable_number(
            metadata.get(field_name), f"video_metadata.{field_name}", integer=True
        )

    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise DatasetValidationError("objects must be an array")
    object_fields = {
        "ground_truth_id",
        "object_type",
        "visible_frame_ranges",
        "signal_state_intervals",
        "transitions",
        "motion_intervals",
        "route_number",
        "screen_stage_intervals",
        "text_annotations",
    }
    seen_ids: set[str] = set()
    for index, item in enumerate(objects):
        location = f"objects[{index}]"
        if not isinstance(item, Mapping):
            raise DatasetValidationError(f"{location} must be an object")
        _require_keys(item, object_fields, location)
        _reject_extra_keys(item, object_fields, location)
        object_id = _require_string(item.get("ground_truth_id"), f"{location}.ground_truth_id")
        if object_id in seen_ids:
            raise DatasetValidationError(f"duplicate ground_truth_id: {object_id}")
        seen_ids.add(object_id)
        _require_string(item.get("object_type"), f"{location}.object_type")

        visible_ranges = item.get("visible_frame_ranges")
        if not isinstance(visible_ranges, list):
            raise DatasetValidationError(f"{location}.visible_frame_ranges must be an array")
        for range_index, frame_range in enumerate(visible_ranges):
            range_location = f"{location}.visible_frame_ranges[{range_index}]"
            _validate_frame_range(frame_range, range_location)
            assert isinstance(frame_range, Mapping)
            _reject_extra_keys(frame_range, {"start_frame", "end_frame"}, range_location)
        _validate_enum_intervals(
            item.get("signal_state_intervals"),
            f"{location}.signal_state_intervals",
            field_name="state",
            allowed=SIGNAL_STATES,
        )
        _validate_enum_intervals(
            item.get("motion_intervals"),
            f"{location}.motion_intervals",
            field_name="state",
            allowed=MOTION_STATES,
        )
        _validate_enum_intervals(
            item.get("screen_stage_intervals"),
            f"{location}.screen_stage_intervals",
            field_name="stage",
            allowed=SCREEN_STAGES,
        )
        route_number = item.get("route_number")
        if route_number is not None and not isinstance(route_number, str):
            raise DatasetValidationError(f"{location}.route_number must be a string or null")
        transitions = item.get("transitions")
        if not isinstance(transitions, list):
            raise DatasetValidationError(f"{location}.transitions must be an array")
        for transition_index, transition in enumerate(transitions):
            transition_location = f"{location}.transitions[{transition_index}]"
            if not isinstance(transition, Mapping):
                raise DatasetValidationError(f"{transition_location} must be an object")
            transition_fields = {"frame", "from_state", "to_state", "ambiguous"}
            _require_keys(transition, transition_fields, transition_location)
            _reject_extra_keys(transition, transition_fields, transition_location)
            frame = transition.get("frame")
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                raise DatasetValidationError(f"{transition_location}.frame must be >= 0")
            for state_field in ("from_state", "to_state"):
                if transition.get(state_field) not in SIGNAL_STATES:
                    raise DatasetValidationError(
                        f"{transition_location}.{state_field} must be a signal state"
                    )
            if not isinstance(transition.get("ambiguous"), bool):
                raise DatasetValidationError(f"{transition_location}.ambiguous must be a boolean")

        text_annotations = item.get("text_annotations")
        if not isinstance(text_annotations, list):
            raise DatasetValidationError(f"{location}.text_annotations must be an array")
        text_annotation_fields = {"start_frame", "end_frame", "text", "ambiguous"}
        for text_index, text_annotation in enumerate(text_annotations):
            text_location = f"{location}.text_annotations[{text_index}]"
            if not isinstance(text_annotation, Mapping):
                raise DatasetValidationError(f"{text_location} must be an object")
            _require_keys(text_annotation, text_annotation_fields, text_location)
            _reject_extra_keys(text_annotation, text_annotation_fields, text_location)
            _validate_frame_range(text_annotation, text_location)
            text_value = text_annotation.get("text")
            if text_value is not None and not isinstance(text_value, str):
                raise DatasetValidationError(f"{text_location}.text must be a string or null")
            if not isinstance(text_annotation.get("ambiguous"), bool):
                raise DatasetValidationError(f"{text_location}.ambiguous must be a boolean")

    constraints = payload.get("narration_constraints")
    if not isinstance(constraints, Mapping):
        raise DatasetValidationError("narration_constraints must be an object")
    constraint_fields = {
        "forbidden_phrases",
        "forbidden_patterns",
        "maximum_duplicate_events",
        "notes",
    }
    _require_keys(constraints, constraint_fields, "narration_constraints")
    _reject_extra_keys(constraints, constraint_fields, "narration_constraints")
    for field_name in ("forbidden_phrases", "forbidden_patterns"):
        values = constraints.get(field_name)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise DatasetValidationError(
                f"narration_constraints.{field_name} must be an array of strings"
            )
    maximum_duplicates = constraints.get("maximum_duplicate_events")
    if maximum_duplicates is not None and (
        isinstance(maximum_duplicates, bool)
        or not isinstance(maximum_duplicates, int)
        or maximum_duplicates < 0
    ):
        raise DatasetValidationError(
            "narration_constraints.maximum_duplicate_events must be null or an integer >= 0"
        )
    notes = constraints.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise DatasetValidationError("narration_constraints.notes must be a string or null")

    review_notes = payload.get("review_notes")
    if review_notes is not None and not isinstance(review_notes, str):
        raise DatasetValidationError("review_notes must be a string or null")


def load_annotation(path: str | Path) -> dict[str, Any]:
    """Load and validate one annotation document."""

    annotation_path = Path(path)
    payload = _read_json_object(annotation_path)
    validate_annotation(payload)
    return payload


def build_annotation_draft(
    video_id: str,
    *,
    fps: float | None,
    frame_count: int | None,
    width: int | None,
    height: int | None,
) -> dict[str, Any]:
    """Build an unlabeled draft; no object or state ground truth is inferred."""

    draft: dict[str, Any] = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "video_id": video_id,
        "review_status": "needs_manual_review",
        "video_metadata": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        },
        "objects": [],
        "narration_constraints": {
            "forbidden_phrases": [],
            "forbidden_patterns": [],
            "maximum_duplicate_events": 0,
            "notes": (
                "Human review required. Do not use model predictions as ground truth or "
                "invent unreadable text, route numbers, signal subtype, or crossing advice."
            ),
        },
    }
    validate_annotation(draft)
    return draft
