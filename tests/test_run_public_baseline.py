from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import run_public_baseline as cli
from vision_agent.pipeline import PipelineConfig
from vision_agent.public_baseline import runner
from vision_agent.public_baseline.runner import (
    SUMMARY_FIELDS,
    BaselineRunResult,
    BaselineSettings,
)


def _videos() -> list[dict[str, object]]:
    return [
        {
            "id": "signal_clip",
            "category": "traffic_light",
            "video_paths": ["videos/signal.mp4"],
        },
        {
            "id": "broken_bus",
            "category": "bus",
            "video_paths": ["videos/broken.mp4"],
        },
        {
            "id": "machine_clip",
            "category": "kiosk_like_machine",
            "video_paths": ["videos/machine.mp4"],
        },
    ]


def test_category_filters_only_known_coco_categories() -> None:
    assert runner.classes_for_category("traffic-light") == (9,)
    assert runner.classes_for_category("BUS") == (5,)
    assert runner.classes_for_category("kiosk_like_machine") is None
    assert runner.classes_for_category("ticket_machine_screen") is None
    assert runner.classes_for_category("unlisted") is None


def test_batch_continues_after_one_video_failure_and_writes_matching_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "dataset" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "load_manifest", lambda _path: {"videos": _videos()})

    def fake_resolve(
        manifest: str | Path,
        video: dict[str, object],
        *,
        require_exists: bool = True,
    ) -> Path:
        del require_exists
        return Path(manifest).parent / str(video["video_paths"][0])

    monkeypatch.setattr(runner, "resolve_video_path", fake_resolve)
    received: list[PipelineConfig] = []

    def fake_pipeline(config: PipelineConfig) -> dict[str, object]:
        received.append(config)
        if Path(config.source).name == "broken.mp4":
            raise RuntimeError("synthetic decode failure")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        video_path = config.output_dir / f"{Path(config.source).stem}_annotated.mp4"
        jsonl_path = config.output_dir / f"{Path(config.source).stem}_detections.jsonl"
        video_path.write_bytes(b"synthetic-video")
        class_name = "traffic light" if config.classes == (9,) else "person"
        frame_payload = {
            "frame_index": 0,
            "detections": [
                {
                    "class_name": class_name,
                    "analysis": {"object_type": class_name.replace(" ", "_")},
                }
            ],
            "analysis_results": [{"object_type": class_name.replace(" ", "_")}],
        }
        jsonl_path.write_text(json.dumps(frame_payload) + "\n", encoding="utf-8")

        return {
            "frames": 10,
            "effective_fps": 20.0,
            "average_inference_ms": 4.5,
            "output_video": str(video_path),
            "output_jsonl": str(jsonl_path),
        }

    settings = BaselineSettings(
        model="same-model.pt",
        confidence=0.25,
        image_size=512,
        device="cpu",
        ocr_backend="none",
        generic_vlm_model="local-vlm",
    )
    result = runner.run_public_baseline(
        manifest_path,
        tmp_path / "outputs",
        settings=settings,
        pipeline_runner=fake_pipeline,
        git_commit_sha="abc123",
    )

    assert len(received) == 3
    assert [config.classes for config in received] == [(9,), (5,), None]
    assert {config.model for config in received} == {"same-model.pt"}
    assert {config.confidence for config in received} == {0.25}
    assert {config.image_size for config in received} == {512}
    assert {config.device for config in received} == {"cpu"}
    assert result.summary["total_videos"] == 3
    assert result.summary["succeeded_videos"] == 2
    assert result.summary["failed_videos"] == 1

    rows = result.summary["videos"]
    assert [row["status"] for row in rows] == ["success", "failed", "success"]
    assert rows[0]["classes"] == "9"
    assert rows[1]["classes"] == "5"
    assert rows[2]["classes"] == "all"
    assert rows[1]["error_type"] == "RuntimeError"
    assert rows[1]["error_message"] == "synthetic decode failure"
    assert rows[1]["error"] == {"type": "RuntimeError", "message": "synthetic decode failure"}
    assert rows[0]["settings"]["model"] == "same-model.pt"
    assert Path(rows[0]["output_video"]).is_file()
    assert Path(rows[0]["output_jsonl"]).is_file()
    assert rows[0]["detection_class_counts"] == {"traffic light": 1}
    assert rows[0]["analysis_object_type_counts"] == {"traffic_light": 1}
    assert rows[0]["inferred_analyzer_routing_counts"]["TrafficLightAnalyzer"] == 1
    assert rows[2]["detection_class_counts"] == {"person": 1}
    assert rows[2]["inferred_analyzer_routing_counts"]["GenericVisionAnalyzer"] == 1
    assert rows[2]["inferred_analyzer_routing_counts"]["KioskAnalyzer"] == 0
    assert any(
        "Required custom detector class was not observed: reverse_vending_machine" in item
        for item in rows[2]["limitations"]
    )

    persisted = json.loads(result.json_path.read_text(encoding="utf-8"))
    with result.csv_path.open(encoding="utf-8", newline="") as file:
        csv_reader = csv.DictReader(file)
        csv_rows = list(csv_reader)
    assert tuple(csv_reader.fieldnames or ()) == SUMMARY_FIELDS
    assert persisted["summary_fields"] == list(SUMMARY_FIELDS)
    assert all(tuple(row) == SUMMARY_FIELDS for row in persisted["videos"])
    assert len(csv_rows) == len(persisted["videos"])
    for json_row, csv_row in zip(persisted["videos"], csv_rows, strict=True):
        assert set(json_row) == set(csv_row)
        assert json_row["video_id"] == csv_row["video_id"]
        assert json_row["status"] == csv_row["status"]

    settings_payload = json.loads(
        (tmp_path / "outputs" / "signal_clip" / "run_settings.json").read_text(encoding="utf-8")
    )
    assert settings_payload["git_commit_sha"] == "abc123"
    assert settings_payload["pipeline_config"]["model"] == "same-model.pt"
    assert settings_payload["pipeline_config"]["classes"] == [9]
    assert settings_payload["pipeline_config"]["ocr_backend"] == "none"
    assert settings_payload["pipeline_config"]["generic_vlm_model"] == "local-vlm"


def test_source_resolution_failure_is_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    videos = _videos()[:2]
    monkeypatch.setattr(runner, "load_manifest", lambda _path: {"videos": videos})

    def fake_resolve(
        manifest: str | Path,
        video: dict[str, object],
        *,
        require_exists: bool = True,
    ) -> Path:
        del manifest, require_exists
        if video["id"] == "signal_clip":
            raise FileNotFoundError("missing source")
        return tmp_path / "bus.mp4"

    monkeypatch.setattr(runner, "resolve_video_path", fake_resolve)
    calls: list[str] = []

    def fake_pipeline(config: PipelineConfig) -> dict[str, object]:
        calls.append(config.source)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        output_video = config.output_dir / "bus_annotated.mp4"
        output_jsonl = config.output_dir / "bus_detections.jsonl"
        output_video.write_bytes(b"synthetic-video")
        output_jsonl.write_text("", encoding="utf-8")
        return {
            "frames": 0,
            "output_video": str(output_video),
            "output_jsonl": str(output_jsonl),
        }

    result = runner.run_public_baseline(
        manifest_path,
        tmp_path / "out",
        pipeline_runner=fake_pipeline,
        git_commit_sha="deadbeef",
    )

    assert calls == [str(tmp_path / "bus.mp4")]
    assert result.summary["failed_videos"] == 1
    assert result.summary["succeeded_videos"] == 1
    assert result.summary["videos"][0]["error_type"] == "FileNotFoundError"
    failed_settings = Path(result.summary["videos"][0]["settings_path"])
    assert failed_settings.is_file()
    failed_payload = json.loads(failed_settings.read_text(encoding="utf-8"))
    assert failed_payload["pipeline_config"] is None
    assert failed_payload["category_class_filter"] == [9]


def test_cli_forwards_reproducible_baseline_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[BaselineSettings] = []

    def fake_run(
        manifest_path: Path,
        output_dir: Path,
        *,
        settings: BaselineSettings,
    ) -> BaselineRunResult:
        assert manifest_path == Path("manifest.json")
        assert output_dir == Path("batch-out")
        captured.append(settings)
        return BaselineRunResult(
            summary={"total_videos": 2, "succeeded_videos": 2, "failed_videos": 0},
            json_path=tmp_path / "run_summary.json",
            csv_path=tmp_path / "run_summary.csv",
        )

    monkeypatch.setattr(cli, "run_public_baseline", fake_run)
    exit_code = cli.main(
        [
            "--manifest",
            "manifest.json",
            "--output-dir",
            "batch-out",
            "--device",
            "cpu",
            "--model",
            "baseline.pt",
            "--imgsz",
            "512",
            "--conf",
            "0.2",
            "--ocr-backend",
            "none",
            "--vlm-model",
            "local-vlm",
            "--vlm-classes",
            "unknown panel,machine",
        ]
    )

    assert exit_code == 0
    assert captured == [
        BaselineSettings(
            model="baseline.pt",
            confidence=0.2,
            image_size=512,
            device="cpu",
            ocr_backend="none",
            generic_vlm_model="local-vlm",
            generic_vlm_classes=("unknown panel", "machine"),
        )
    ]


def test_cli_returns_failure_after_writing_batch_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "run_public_baseline",
        lambda *_args, **_kwargs: BaselineRunResult(
            summary={"total_videos": 2, "succeeded_videos": 1, "failed_videos": 1},
            json_path=tmp_path / "run_summary.json",
            csv_path=tmp_path / "run_summary.csv",
        ),
    )

    assert cli.main(["--manifest", "manifest.json", "--output-dir", "out"]) == 1


def test_missing_pipeline_artifact_is_a_video_failure_without_stopping_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    videos = _videos()
    monkeypatch.setattr(runner, "load_manifest", lambda _path: {"videos": videos})
    monkeypatch.setattr(
        runner,
        "resolve_video_path",
        lambda manifest, video, require_exists=False: (
            Path(manifest).parent / str(video["video_paths"][0])
        ),
    )

    def fake_pipeline(config: PipelineConfig) -> dict[str, object]:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        output_video = config.output_dir / "annotated.mp4"
        output_video.write_bytes(b"synthetic-video")
        if Path(config.source).name == "signal.mp4":
            return {"output_video": str(output_video), "output_jsonl": None}
        output_jsonl = config.output_dir / "detections.jsonl"
        output_jsonl.write_text(
            "{not-json\n" if Path(config.source).name == "machine.mp4" else "",
            encoding="utf-8",
        )
        return {
            "output_video": str(output_video),
            "output_jsonl": str(output_jsonl),
        }

    result = runner.run_public_baseline(
        manifest_path,
        tmp_path / "out",
        pipeline_runner=fake_pipeline,
        git_commit_sha="deadbeef",
    )

    assert [row["status"] for row in result.summary["videos"]] == ["failed", "success", "failed"]
    assert result.summary["videos"][0]["error_type"] == "FileNotFoundError"
    assert result.summary["videos"][2]["error_type"] == "ValueError"
    assert result.summary["failed_videos"] == 2
