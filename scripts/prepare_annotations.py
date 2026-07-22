#!/usr/bin/env python3
"""Prepare human-review artifacts for the public baseline videos."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_agent.public_baseline.dataset import (
    DatasetValidationError,
    build_annotation_draft,
    load_manifest,
    resolve_video_path,
)


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int


def extract_video_metadata(path: Path) -> VideoMetadata:
    """Read basic video metadata without interpreting video content."""

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid FPS metadata for {path}: {fps}")
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid frame metadata for {path}: frames={frame_count}, size={width}x{height}"
        )
    return VideoMetadata(
        fps=round(fps, 6),
        frame_count=frame_count,
        width=width,
        height=height,
    )


def sample_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    """Choose deterministic, evenly spaced frame indices including both ends."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if frame_count <= sample_count:
        return list(range(frame_count))
    if sample_count == 1:
        return [0]
    last = frame_count - 1
    return sorted({round(index * last / (sample_count - 1)) for index in range(sample_count)})


def _frame_with_label(frame: np.ndarray, frame_index: int) -> np.ndarray:
    labeled = frame.copy()
    label = f"frame {frame_index}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(labeled.shape[:2]) / 700)
    thickness = max(1, round(font_scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(
        labeled,
        (0, 0),
        (text_width + 18, text_height + baseline + 16),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        labeled,
        label,
        (8, text_height + 7),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return labeled


def _contact_sheet_cell(frame: np.ndarray, *, width: int = 320, height: int = 240) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_width = max(1, round(frame.shape[1] * scale))
    resized_height = max(1, round(frame.shape[0] * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    cell = np.zeros((height, width, 3), dtype=np.uint8)
    x_offset = (width - resized_width) // 2
    y_offset = (height - resized_height) // 2
    cell[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return cell


def create_review_images(
    video_path: Path,
    *,
    frame_count: int,
    sample_count: int,
    columns: int,
    frames_dir: Path,
    contact_sheet_path: Path,
) -> list[int]:
    """Create numbered sample frames and a contact sheet for manual review."""

    indices = sample_frame_indices(frame_count, sample_count)
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    cells: list[np.ndarray] = []
    written_indices: list[int] = []
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            labeled = _frame_with_label(frame, frame_index)
            frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(frame_path), labeled):
                raise RuntimeError(f"Could not write review frame: {frame_path}")
            cells.append(_contact_sheet_cell(labeled))
            written_indices.append(frame_index)
    finally:
        capture.release()

    if not cells:
        raise RuntimeError(f"Could not decode any sampled frame from {video_path}")
    blank = np.zeros_like(cells[0])
    rows: list[np.ndarray] = []
    for start in range(0, len(cells), columns):
        row = cells[start : start + columns]
        row.extend(blank.copy() for _ in range(columns - len(row)))
        rows.append(np.hstack(row))
    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(contact_sheet_path), np.vstack(rows)):
        raise RuntimeError(f"Could not write contact sheet: {contact_sheet_path}")
    return written_indices


def _write_json_if_missing(path: Path, payload: dict[str, Any]) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def _write_review_readme(
    output_dir: Path,
    *,
    manifest_path: Path,
    results: list[dict[str, Any]],
) -> None:
    rows = []
    for result in results:
        detail = str(result.get("error") or result.get("video_path") or "-").replace("|", "\\|")
        rows.append(
            f"| `{result['video_id']}` | `{result['status']}` | "
            f"`{result['annotation']}` | {detail} |"
        )
    content = """# Public baseline annotation review

This directory contains generated review aids, not verified ground truth. The tool never
uses detector, OCR, or VLM output as labels. Every draft remains `needs_manual_review`
until a person has checked object identity, visibility, state boundaries, and text.

## Review workflow

1. Open the contact sheet and numbered frames for one video. Inspect the original video
   around every possible boundary; sampled frames alone are not sufficient.
2. Edit `annotations/<video-id>.json`. Add only observations a person can verify. Keep
   unavailable route numbers/text as `null`, and mark uncertain transitions with
   `"ambiguous": true`.
3. Check all frame ranges are inclusive and within `0..frame_count-1`.
4. Add prohibited narration phrases or regular-expression patterns where the source's
   safety constraint requires them.
5. Validate the file, then change `review_status` to `reviewed`. Copy the reviewed file
   to the canonical path listed in the manifest. Quantitative evaluation ignores drafts.

An object entry uses this shape (remove sections that do not apply only if the schema
allows it; empty arrays are preferable):

```json
{
  "ground_truth_id": "object-1",
  "object_type": "traffic_light",
  "visible_frame_ranges": [{"start_frame": 0, "end_frame": 20}],
  "signal_state_intervals": [
    {"start_frame": 0, "end_frame": 20, "state": "UNKNOWN"}
  ],
  "transitions": [
    {"frame": 10, "from_state": "GREEN", "to_state": "RED", "ambiguous": true}
  ],
  "motion_intervals": [],
  "route_number": null,
  "screen_stage_intervals": [],
  "text_annotations": []
}
```

Allowed signal states: `RED`, `GREEN`, `YELLOW`, `OFF`, `UNKNOWN`.
Allowed bus motion states: `APPROACHING`, `STOPPED`, `RECEDING`, `UNKNOWN`.
Allowed screen stages: `ORDER_TYPE_SELECTION`, `PAYMENT`, `CONFIRMATION`, `UNKNOWN`.

## Preparation results

| Video | Status | Draft | Video/error |
|---|---|---|---|
"""
    content += "\n".join(rows)
    content += f"\n\nManifest: `{manifest_path}`\n"
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def prepare_annotations(
    manifest_path: Path,
    output_dir: Path,
    *,
    sample_count: int = 12,
    columns: int = 4,
) -> dict[str, Any]:
    """Prepare drafts and visual aids for every manifest entry."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if columns <= 0:
        raise ValueError("columns must be positive")
    manifest = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for video in manifest["videos"]:
        video_id = str(video["id"])
        annotation_path = output_dir / "annotations" / f"{video_id}.json"
        result: dict[str, Any] = {
            "video_id": video_id,
            "status": "pending",
            "annotation": str(annotation_path.relative_to(output_dir)),
            "annotation_created": False,
            "sampled_frames": [],
        }
        try:
            video_path = resolve_video_path(manifest_path, video)
            assert video_path is not None
            metadata = extract_video_metadata(video_path)
            draft = build_annotation_draft(video_id, **asdict(metadata))
            result["annotation_created"] = _write_json_if_missing(annotation_path, draft)
            sampled = create_review_images(
                video_path,
                frame_count=metadata.frame_count,
                sample_count=sample_count,
                columns=columns,
                frames_dir=output_dir / "frames" / video_id,
                contact_sheet_path=output_dir / "contact_sheets" / f"{video_id}.jpg",
            )
            result.update(
                {
                    "status": "prepared",
                    "video_path": str(video_path),
                    "metadata": asdict(metadata),
                    "sampled_frames": sampled,
                    "contact_sheet": f"contact_sheets/{video_id}.jpg",
                }
            )
        except FileNotFoundError as exc:
            draft = build_annotation_draft(
                video_id, fps=None, frame_count=None, width=None, height=None
            )
            result["annotation_created"] = _write_json_if_missing(annotation_path, draft)
            result.update({"status": "missing_video", "error": str(exc)})
        except Exception as exc:  # keep remaining videos reviewable
            result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        results.append(result)

    summary = {
        "schema_version": "1.0",
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "videos_total": len(results),
        "videos_prepared": sum(result["status"] == "prepared" for result in results),
        "videos_missing": sum(result["status"] == "missing_video" for result in results),
        "videos_failed": sum(result["status"] == "error" for result in results),
        "results": results,
    }
    (output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_review_readme(
        output_dir,
        manifest_path=manifest_path,
        results=results,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="공개 baseline 영상의 사람 검수용 annotation 초안과 contact sheet를 생성합니다."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="dataset manifest JSON")
    parser.add_argument("--output-dir", required=True, type=Path, help="검수 자료 출력 폴더")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=12,
        help="영상마다 균등하게 추출할 최대 프레임 수 (기본: 12)",
    )
    parser.add_argument("--columns", type=int, default=4, help="contact sheet 열 수 (기본: 4)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = prepare_annotations(
            args.manifest,
            args.output_dir,
            sample_count=args.sample_count,
            columns=args.columns,
        )
    except (DatasetValidationError, OSError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
