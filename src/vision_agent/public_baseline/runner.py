from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vision_agent.pipeline import PipelineConfig, run_video_pipeline
from vision_agent.router import ObjectRouter

from .dataset import load_manifest, resolve_video_path


PipelineRunner = Callable[[PipelineConfig], Mapping[str, Any]]

CATEGORY_CLASS_FILTERS: dict[str, tuple[int, ...] | None] = {
    "traffic_light": (9,),
    "bus": (5,),
    "kiosk_like_machine": None,
    "ticket_machine_screen": None,
}

REQUIRED_CUSTOM_CLASSES_BY_CATEGORY = {
    "kiosk_like_machine": frozenset({"reverse_vending_machine"}),
    "ticket_machine_screen": frozenset({"ticket_machine"}),
}

PIPELINE_SUMMARY_FIELDS = (
    "video_duration_s",
    "source_fps",
    "frames",
    "elapsed_s",
    "effective_fps",
    "realtime_factor",
    "average_inference_ms",
    "detections",
    "events",
    "analysis_events",
    "narrations",
    "signal_targets",
    "signal_changes",
    "bus_analysis_results",
    "bus_detection_frames",
    "bus_approach_events",
)

ANALYZER_NAMES = (
    "TrafficLightAnalyzer",
    "BusAnalyzer",
    "KioskAnalyzer",
    "TextObjectAnalyzer",
    "GenericVisionAnalyzer",
)

SUMMARY_FIELDS = (
    "video_id",
    "category",
    "status",
    "source",
    "classes",
    "git_commit_sha",
    "settings_path",
    "settings",
    "output_dir",
    "pipeline_summary_path",
    "output_video",
    "output_jsonl",
    "detection_class_counts",
    "analysis_object_type_counts",
    "inferred_analyzer_routing_counts",
    "limitations",
    *PIPELINE_SUMMARY_FIELDS,
    "error",
    "error_type",
    "error_message",
)


@dataclass(frozen=True, slots=True)
class BaselineSettings:
    """Settings shared by every video except its documented category class filter."""

    model: str = "yolo26s.pt"
    confidence: float = 0.10
    image_size: int = 640
    device: str | None = None
    track: bool = True
    tracker: str = "bytetrack.yaml"
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


@dataclass(frozen=True, slots=True)
class BaselineRunResult:
    summary: dict[str, Any]
    json_path: Path
    csv_path: Path

    @property
    def failed_videos(self) -> int:
        return int(self.summary["failed_videos"])


def classes_for_category(category: str) -> tuple[int, ...] | None:
    """Return the fixed baseline COCO filter for a public-pack category."""
    normalized = re.sub(r"[^a-z0-9]+", "_", category.strip().lower()).strip("_")
    return CATEGORY_CLASS_FILTERS.get(normalized)


def get_git_commit_sha(repository: str | Path | None = None) -> str:
    """Read the current commit without making Git availability a batch blocker."""
    working_directory = Path(repository) if repository is not None else Path.cwd()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit_sha = completed.stdout.strip()
    return commit_sha or "unknown"


def build_pipeline_config(
    *,
    source: str | Path,
    output_dir: str | Path,
    category: str,
    settings: BaselineSettings,
) -> PipelineConfig:
    """Create one config from shared settings and the category-only class policy."""
    config = PipelineConfig(source=str(source), output_dir=Path(output_dir))
    return replace(
        config,
        model=settings.model,
        classes=classes_for_category(category),
        confidence=settings.confidence,
        image_size=settings.image_size,
        device=settings.device,
        track=settings.track,
        tracker=settings.tracker,
        ocr_backend=settings.ocr_backend,
        ocr_language=settings.ocr_language,
        ocr_model_path=settings.ocr_model_path,
        allow_ocr_download=settings.allow_ocr_download,
        generic_vlm_model=settings.generic_vlm_model,
        generic_vlm_device=settings.generic_vlm_device,
        allow_vlm_download=settings.allow_vlm_download,
        generic_vlm_classes=settings.generic_vlm_classes,
    )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_compatible(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _classes_label(classes: tuple[int, ...] | None) -> str:
    if classes is None:
        return "all"
    return ",".join(str(class_id) for class_id in classes)


def _empty_summary_row(
    *,
    video_id: str,
    category: str,
    source: Path | None,
    classes: tuple[int, ...] | None,
    git_commit_sha: str,
    video_output_dir: Path,
) -> dict[str, Any]:
    row = {field: None for field in SUMMARY_FIELDS}
    row.update(
        {
            "video_id": video_id,
            "category": category,
            "status": "failed",
            "source": str(source) if source is not None else None,
            "classes": _classes_label(classes),
            "git_commit_sha": git_commit_sha,
            "output_dir": str(video_output_dir),
            "detection_class_counts": {},
            "analysis_object_type_counts": {},
            "inferred_analyzer_routing_counts": {name: 0 for name in ANALYZER_NAMES},
            "limitations": [],
        }
    )
    return row


def _validate_video_id(raw_video_id: object) -> str:
    video_id = str(raw_video_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", video_id):
        raise ValueError(f"안전하지 않은 video ID입니다: {video_id!r}")
    return video_id


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in SUMMARY_FIELDS})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_json_compatible(value), ensure_ascii=False, sort_keys=True)
    return value


def summarize_pipeline_jsonl(path: str | Path) -> dict[str, dict[str, int]]:
    """Count detector outputs and analyzer calls from the preserved frame JSONL."""
    detection_counts: Counter[str] = Counter()
    analysis_counts: Counter[str] = Counter()
    analyzer_counts: Counter[str] = Counter({name: 0 for name in ANALYZER_NAMES})
    router = ObjectRouter()
    jsonl_path = Path(path)
    with jsonl_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_number}: 올바르지 않은 JSONL입니다.") from exc
            if not isinstance(frame, Mapping):
                raise ValueError(f"{jsonl_path}:{line_number}: frame 레코드는 객체여야 합니다.")

            detections = frame.get("detections", [])
            if isinstance(detections, list):
                for detection in detections:
                    if not isinstance(detection, Mapping):
                        continue
                    class_name = str(detection.get("class_name", "unknown")).strip() or "unknown"
                    detection_counts[class_name] += 1
                    if isinstance(detection.get("analysis"), Mapping):
                        analyzer_name = type(router.analyzer_for(class_name)).__name__
                        analyzer_counts[analyzer_name] += 1

            analysis_results = frame.get("analysis_results", [])
            if isinstance(analysis_results, list):
                for analysis in analysis_results:
                    if not isinstance(analysis, Mapping):
                        continue
                    object_type = str(analysis.get("object_type", "unknown")).strip() or "unknown"
                    analysis_counts[object_type] += 1

    return {
        "detection_class_counts": dict(sorted(detection_counts.items())),
        "analysis_object_type_counts": dict(sorted(analysis_counts.items())),
        "inferred_analyzer_routing_counts": dict(sorted(analyzer_counts.items())),
    }


def run_public_baseline(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    settings: BaselineSettings | None = None,
    pipeline_runner: PipelineRunner | None = None,
    git_commit_sha: str | None = None,
) -> BaselineRunResult:
    """Run every manifest video while isolating errors to the affected video."""
    manifest_file = Path(manifest_path).resolve()
    destination = Path(output_dir).resolve()
    manifest = load_manifest(manifest_file)
    videos = manifest.get("videos")
    if not isinstance(videos, list):
        raise ValueError("manifest의 videos는 배열이어야 합니다.")

    baseline_settings = settings or BaselineSettings()
    execute_pipeline = pipeline_runner or run_video_pipeline
    commit_sha = git_commit_sha or get_git_commit_sha()
    destination.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []

    for index, raw_video in enumerate(videos):
        video = raw_video if isinstance(raw_video, Mapping) else {}
        raw_video_id = video.get("id", f"invalid-video-{index + 1}")
        category = str(video.get("category", "unknown")).strip() or "unknown"
        source: Path | None = None
        source_error: Exception | None = None
        try:
            source = resolve_video_path(manifest_file, video, require_exists=False)
        except Exception as exc:
            source_error = exc
        classes = classes_for_category(category)
        try:
            video_id = _validate_video_id(raw_video_id)
        except ValueError:
            video_id = f"invalid-video-{index + 1}"
        video_output_dir = destination / video_id
        row = _empty_summary_row(
            video_id=str(raw_video_id),
            category=category,
            source=source,
            classes=classes,
            git_commit_sha=commit_sha,
            video_output_dir=video_output_dir,
        )
        settings_path = video_output_dir / "run_settings.json"
        preliminary_settings = {
            **asdict(baseline_settings),
            "classes": classes,
        }
        try:
            video_output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                settings_path,
                {
                    "schema_version": "1.0",
                    "video_id": str(raw_video_id),
                    "category": category,
                    "manifest": str(manifest_file),
                    "git_commit_sha": commit_sha,
                    "baseline_settings": asdict(baseline_settings),
                    "category_class_filter": classes,
                    "pipeline_config": None,
                },
            )
            row["settings_path"] = str(settings_path)
            row["settings"] = _json_compatible(preliminary_settings)
        except OSError as exc:
            row["limitations"].append(
                f"Per-video settings unavailable: {type(exc).__name__}: {exc}"
            )

        try:
            _validate_video_id(raw_video_id)
            if source_error is not None:
                raise source_error
            if source is None:
                raise FileNotFoundError(f"{video_id}: video_paths 후보가 없습니다.")
            config = build_pipeline_config(
                source=source,
                output_dir=video_output_dir,
                category=category,
                settings=baseline_settings,
            )
            serialized_config = _json_compatible(asdict(config))
            _write_json(
                settings_path,
                {
                    "schema_version": "1.0",
                    "video_id": video_id,
                    "category": category,
                    "manifest": str(manifest_file),
                    "git_commit_sha": commit_sha,
                    "baseline_settings": asdict(baseline_settings),
                    "category_class_filter": classes,
                    "pipeline_config": serialized_config,
                },
            )
            row["settings_path"] = str(settings_path)
            row["settings"] = serialized_config

            result = dict(execute_pipeline(config))
            pipeline_summary_path = video_output_dir / "pipeline_summary.json"
            _write_json(pipeline_summary_path, result)
            output_video = result.get("output_video")
            output_jsonl = result.get("output_jsonl")
            for output_name, output_value in (
                ("annotated MP4", output_video),
                ("prediction JSONL", output_jsonl),
            ):
                if not isinstance(output_value, (str, Path)) or not str(output_value).strip():
                    raise FileNotFoundError(f"Pipeline returned no {output_name} output path.")
                if not Path(output_value).is_file():
                    raise FileNotFoundError(
                        f"Pipeline {output_name} output does not exist: {output_value}"
                    )
            row.update(
                {
                    "status": "success",
                    "pipeline_summary_path": str(pipeline_summary_path),
                    "output_video": output_video,
                    "output_jsonl": output_jsonl,
                    "error_type": None,
                    "error_message": None,
                }
            )
            try:
                row.update(summarize_pipeline_jsonl(output_jsonl))
            except Exception as exc:
                raise ValueError(
                    f"Prediction JSONL could not be summarized: {type(exc).__name__}: {exc}"
                ) from exc
            for field in PIPELINE_SUMMARY_FIELDS:
                row[field] = result.get(field)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)

        normalized_category = re.sub(r"[^a-z0-9]+", "_", category.strip().lower()).strip("_")
        required_custom_classes = REQUIRED_CUSTOM_CLASSES_BY_CATEGORY.get(
            normalized_category,
            frozenset(),
        )
        observed_classes = {
            re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
            for name, count in row["detection_class_counts"].items()
            if count
        }
        if required_custom_classes and observed_classes.isdisjoint(required_custom_classes):
            row["limitations"].append(
                "Required custom detector class was not observed: "
                + ", ".join(sorted(required_custom_classes))
                + ". The baseline checkpoint cannot enter the intended safe route."
            )
        rows.append(row)

    finished_at = datetime.now(timezone.utc)
    succeeded = sum(row["status"] == "success" for row in rows)
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "manifest": str(manifest_file),
        "output_dir": str(destination),
        "git_commit_sha": commit_sha,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "baseline_settings": asdict(baseline_settings),
        "category_class_filters": {
            category: _json_compatible(classes)
            for category, classes in CATEGORY_CLASS_FILTERS.items()
        },
        "total_videos": len(rows),
        "succeeded_videos": succeeded,
        "failed_videos": len(rows) - succeeded,
        "summary_fields": list(SUMMARY_FIELDS),
        "videos": rows,
    }
    json_path = destination / "run_summary.json"
    csv_path = destination / "run_summary.csv"
    _write_json(json_path, summary)
    _write_summary_csv(csv_path, rows)
    return BaselineRunResult(summary=summary, json_path=json_path, csv_path=csv_path)
