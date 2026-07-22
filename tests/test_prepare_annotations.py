from __future__ import annotations

from collections.abc import Callable
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from scripts.prepare_annotations import prepare_annotations, sample_frame_indices
from vision_agent.public_baseline.dataset import (
    DatasetValidationError,
    build_annotation_draft,
    load_annotation,
    load_manifest,
    resolve_video_path,
    validate_annotation,
    validate_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPOSITORY_ROOT / "datasets" / "public_baseline"


def _write_synthetic_video(path: Path, *, frame_count: int = 7) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (96, 64),
    )
    assert writer.isOpened()
    try:
        for frame_index in range(frame_count):
            frame = np.full((64, 96, 3), 20 + frame_index * 20, dtype=np.uint8)
            cv2.circle(frame, (12 + frame_index * 8, 40), 5, (0, 255, 0), -1)
            writer.write(frame)
    finally:
        writer.release()


def _manifest(videos: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset_id": "synthetic_public_baseline",
        "source_catalog": "sources.json",
        "videos": videos,
    }


def _video_entry(video_id: str, paths: list[str]) -> dict[str, object]:
    return {
        "id": video_id,
        "category": "traffic_light",
        "video_paths": paths,
        "annotation_path": f"annotations/{video_id}.json",
    }


def _annotation_with_all_object_fields() -> dict[str, Any]:
    annotation = build_annotation_draft(
        "clip",
        fps=30.0,
        frame_count=21,
        width=96,
        height=64,
    )
    annotation["objects"].append(
        {
            "ground_truth_id": "signal-1",
            "object_type": "traffic_light",
            "visible_frame_ranges": [{"start_frame": 0, "end_frame": 20}],
            "signal_state_intervals": [{"start_frame": 0, "end_frame": 20, "state": "GREEN"}],
            "transitions": [
                {
                    "frame": 10,
                    "from_state": "GREEN",
                    "to_state": "RED",
                    "ambiguous": False,
                }
            ],
            "motion_intervals": [{"start_frame": 0, "end_frame": 20, "state": "APPROACHING"}],
            "route_number": None,
            "screen_stage_intervals": [{"start_frame": 0, "end_frame": 20, "stage": "UNKNOWN"}],
            "text_annotations": [
                {"start_frame": 0, "end_frame": 20, "text": None, "ambiguous": True}
            ],
        }
    )
    return annotation


def test_committed_manifest_and_unreviewed_annotations_are_valid() -> None:
    manifest_path = BASELINE_ROOT / "manifest.json"
    manifest = load_manifest(manifest_path)

    assert len(manifest["videos"]) == 7
    assert len({video["id"] for video in manifest["videos"]}) == 7
    for video in manifest["videos"]:
        candidates = video["video_paths"]
        assert candidates[0].endswith(".mp4")
        assert not Path(candidates[0]).is_absolute()
        assert not Path(candidates[1]).is_absolute()
        annotation = load_annotation(BASELINE_ROOT / video["annotation_path"])
        assert annotation["video_id"] == video["id"]
        assert annotation["review_status"] == "needs_manual_review"
        assert annotation["objects"] == []
        assert set(annotation["video_metadata"].values()) == {None}

    schema = json.loads((BASELINE_ROOT / "annotation.schema.json").read_text(encoding="utf-8"))
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["properties"]["review_status"]["enum"] == [
        "needs_manual_review",
        "reviewed",
    ]
    required_object_fields = set(schema["$defs"]["object_annotation"]["required"])
    assert {
        "ground_truth_id",
        "object_type",
        "visible_frame_ranges",
        "signal_state_intervals",
        "transitions",
        "motion_intervals",
        "route_number",
        "screen_stage_intervals",
    } <= required_object_fields


def test_manifest_and_annotation_validation_reject_invalid_ground_truth() -> None:
    entry = _video_entry("clip", ["videos/clip.mp4"])
    manifest = _manifest([entry])
    validate_manifest(manifest)

    duplicate_manifest = copy.deepcopy(manifest)
    duplicate_manifest["videos"].append(copy.deepcopy(entry))
    with pytest.raises(DatasetValidationError, match="duplicate video id"):
        validate_manifest(duplicate_manifest)

    absolute_manifest = copy.deepcopy(manifest)
    absolute_manifest["videos"][0]["video_paths"] = ["/tmp/clip.mp4"]
    with pytest.raises(DatasetValidationError, match="must be relative"):
        validate_manifest(absolute_manifest)

    annotation = build_annotation_draft("clip", fps=30.0, frame_count=21, width=96, height=64)
    annotation["objects"].append(
        {
            "ground_truth_id": "signal-1",
            "object_type": "traffic_light",
            "visible_frame_ranges": [{"start_frame": 0, "end_frame": 20}],
            "signal_state_intervals": [
                {"start_frame": 0, "end_frame": 9, "state": "GREEN"},
                {"start_frame": 10, "end_frame": 20, "state": "RED"},
            ],
            "transitions": [
                {
                    "frame": 10,
                    "from_state": "GREEN",
                    "to_state": "RED",
                    "ambiguous": False,
                }
            ],
            "motion_intervals": [],
            "route_number": None,
            "screen_stage_intervals": [],
            "text_annotations": [],
        }
    )
    validate_annotation(annotation)

    invalid_annotation = copy.deepcopy(annotation)
    invalid_annotation["objects"][0]["signal_state_intervals"][0]["state"] = "BLUE"
    with pytest.raises(DatasetValidationError, match="must be one of"):
        validate_annotation(invalid_annotation)

    invalid_annotation = copy.deepcopy(annotation)
    invalid_annotation["objects"][0]["visible_frame_ranges"] = [{"start_frame": 5, "end_frame": 4}]
    with pytest.raises(DatasetValidationError, match="start_frame <= end_frame"):
        validate_annotation(invalid_annotation)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        pytest.param(
            lambda annotation: annotation.pop("objects"),
            "missing required fields",
            id="missing-top-level-field",
        ),
        pytest.param(
            lambda annotation: annotation.__setitem__("unexpected", True),
            "unsupported fields",
            id="extra-top-level-field",
        ),
        pytest.param(
            lambda annotation: annotation["video_metadata"].pop("width"),
            "missing required fields",
            id="missing-metadata-field",
        ),
        pytest.param(
            lambda annotation: annotation["video_metadata"].__setitem__("unexpected", True),
            "unsupported fields",
            id="extra-metadata-field",
        ),
        pytest.param(
            lambda annotation: annotation["objects"][0].pop("route_number"),
            "missing required fields",
            id="missing-object-field",
        ),
        pytest.param(
            lambda annotation: annotation["objects"][0].pop("text_annotations"),
            "missing required fields",
            id="missing-text-annotations",
        ),
        pytest.param(
            lambda annotation: annotation["objects"][0].__setitem__("unexpected", True),
            "unsupported fields",
            id="extra-object-field",
        ),
        pytest.param(
            lambda annotation: annotation["objects"][0]["visible_frame_ranges"][0].__setitem__(
                "unexpected", True
            ),
            "unsupported fields",
            id="extra-visible-range-field",
        ),
        pytest.param(
            lambda annotation: annotation["objects"][0]["signal_state_intervals"][0].__setitem__(
                "unexpected", True
            ),
            "unsupported fields",
            id="extra-enum-interval-field",
        ),
        pytest.param(
            lambda annotation: annotation["objects"][0]["transitions"][0].pop("ambiguous"),
            "missing required fields",
            id="missing-transition-field",
        ),
        pytest.param(
            lambda annotation: annotation["objects"][0]["transitions"][0].__setitem__(
                "unexpected", True
            ),
            "unsupported fields",
            id="extra-transition-field",
        ),
        pytest.param(
            lambda annotation: annotation["objects"][0]["text_annotations"][0].pop("ambiguous"),
            "missing required fields",
            id="missing-text-field",
        ),
        pytest.param(
            lambda annotation: annotation["objects"][0]["text_annotations"][0].__setitem__(
                "text", 42
            ),
            "text must be a string or null",
            id="invalid-text-type",
        ),
        pytest.param(
            lambda annotation: annotation["objects"][0]["text_annotations"][0].__setitem__(
                "ambiguous", "yes"
            ),
            "ambiguous must be a boolean",
            id="invalid-text-ambiguity",
        ),
        pytest.param(
            lambda annotation: annotation["objects"][0]["text_annotations"][0].__setitem__(
                "end_frame", -1
            ),
            "start_frame <= end_frame",
            id="invalid-text-range",
        ),
        pytest.param(
            lambda annotation: annotation["objects"][0]["text_annotations"][0].__setitem__(
                "unexpected", True
            ),
            "unsupported fields",
            id="extra-text-field",
        ),
        pytest.param(
            lambda annotation: annotation["narration_constraints"].pop("maximum_duplicate_events"),
            "missing required fields",
            id="missing-maximum-duplicates",
        ),
        pytest.param(
            lambda annotation: annotation["narration_constraints"].pop("notes"),
            "missing required fields",
            id="missing-constraint-notes",
        ),
        pytest.param(
            lambda annotation: annotation["narration_constraints"].__setitem__("unexpected", True),
            "unsupported fields",
            id="extra-constraint-field",
        ),
        pytest.param(
            lambda annotation: annotation["narration_constraints"].__setitem__(
                "maximum_duplicate_events", True
            ),
            "maximum_duplicate_events",
            id="boolean-maximum-duplicates",
        ),
        pytest.param(
            lambda annotation: annotation["narration_constraints"].__setitem__(
                "maximum_duplicate_events", 1.5
            ),
            "maximum_duplicate_events",
            id="float-maximum-duplicates",
        ),
        pytest.param(
            lambda annotation: annotation["narration_constraints"].__setitem__(
                "maximum_duplicate_events", -1
            ),
            "maximum_duplicate_events",
            id="negative-maximum-duplicates",
        ),
        pytest.param(
            lambda annotation: annotation["narration_constraints"].__setitem__("notes", []),
            "notes must be a string or null",
            id="invalid-constraint-notes",
        ),
        pytest.param(
            lambda annotation: annotation.__setitem__("review_notes", 1),
            "review_notes must be a string or null",
            id="invalid-review-notes",
        ),
    ],
)
def test_annotation_validation_rejects_schema_violations(
    mutation: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    annotation = _annotation_with_all_object_fields()
    mutation(annotation)

    with pytest.raises(DatasetValidationError, match=message):
        validate_annotation(annotation)


def test_annotation_validation_accepts_nullable_schema_fields() -> None:
    annotation = _annotation_with_all_object_fields()
    annotation["narration_constraints"]["maximum_duplicate_events"] = None
    annotation["narration_constraints"]["notes"] = None
    annotation["review_notes"] = None

    validate_annotation(annotation)


def test_sample_frame_indices_are_even_and_include_endpoints() -> None:
    assert sample_frame_indices(7, 3) == [0, 3, 6]
    assert sample_frame_indices(3, 12) == [0, 1, 2]
    assert sample_frame_indices(20, 1) == [0]
    with pytest.raises(ValueError, match="sample_count"):
        sample_frame_indices(3, 0)


def test_prepare_annotations_creates_review_aids_and_preserves_existing_draft(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    video_path = dataset_dir / "videos" / "original" / "available.avi"
    _write_synthetic_video(video_path)
    manifest = _manifest(
        [
            _video_entry(
                "available",
                ["videos/mp4/available.mp4", "videos/original/available.avi"],
            ),
            _video_entry("missing", ["videos/mp4/missing.mp4"]),
        ]
    )
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_dir = tmp_path / "review"
    assert resolve_video_path(manifest_path, manifest["videos"][0]) == video_path.resolve()

    summary = prepare_annotations(manifest_path, output_dir, sample_count=3, columns=2)

    assert summary["videos_total"] == 2
    assert summary["videos_prepared"] == 1
    assert summary["videos_missing"] == 1
    assert [result["status"] for result in summary["results"]] == [
        "prepared",
        "missing_video",
    ]
    assert summary["results"][0]["sampled_frames"] == [0, 3, 6]
    assert summary["results"][0]["video_path"] == str(video_path.resolve())

    available_annotation_path = output_dir / "annotations" / "available.json"
    available = load_annotation(available_annotation_path)
    assert available["review_status"] == "needs_manual_review"
    assert available["objects"] == []
    assert available["video_metadata"] == {
        "fps": 5.0,
        "frame_count": 7,
        "width": 96,
        "height": 64,
    }
    missing = load_annotation(output_dir / "annotations" / "missing.json")
    assert set(missing["video_metadata"].values()) == {None}

    frame_paths = sorted((output_dir / "frames" / "available").glob("*.jpg"))
    assert [path.name for path in frame_paths] == [
        "frame_000000.jpg",
        "frame_000003.jpg",
        "frame_000006.jpg",
    ]
    numbered_frame = cv2.imread(str(frame_paths[0]))
    assert numbered_frame is not None
    assert int(numbered_frame[:30, :80].max()) > 240
    contact_sheet = cv2.imread(str(output_dir / "contact_sheets" / "available.jpg"))
    assert contact_sheet is not None
    assert contact_sheet.shape[:2] == (480, 640)
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "uses detector, OCR, or VLM output as labels" in readme
    assert "ambiguous" in readme

    available["review_notes"] = "human work must survive reruns"
    available_annotation_path.write_text(
        json.dumps(available, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    original_bytes = available_annotation_path.read_bytes()
    second_summary = prepare_annotations(manifest_path, output_dir, sample_count=2, columns=2)
    assert available_annotation_path.read_bytes() == original_bytes
    assert second_summary["results"][0]["annotation_created"] is False


def test_video_assets_and_generated_review_outputs_are_not_tracked() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "*.mp4", "*.ogv", "*.webm"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert tracked == []
    assert "review/" in (BASELINE_ROOT / ".gitignore").read_text(encoding="utf-8")
