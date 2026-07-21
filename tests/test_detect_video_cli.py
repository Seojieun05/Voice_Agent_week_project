from __future__ import annotations

import sys

import pytest

from scripts import detect_video
from vision_agent.pipeline import PipelineConfig


def test_cli_forwards_ocr_and_vlm_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[PipelineConfig] = []

    def fake_run(config: PipelineConfig) -> dict[str, object]:
        captured.append(config)
        return {"frames": 0}

    monkeypatch.setattr(detect_video, "run_video_pipeline", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "detect_video.py",
            "--source",
            "input.mp4",
            "--ocr-backend",
            "rapidocr",
            "--ocr-language",
            "japan",
            "--ocr-model-path",
            "/models/japan.onnx",
            "--allow-ocr-download",
            "--vlm-model",
            "/models/vlm",
            "--vlm-device",
            "cpu",
            "--vlm-classes",
            "unknown panel,vending machine",
            "--allow-vlm-download",
            "--signal-minimum-detection-confidence",
            "0.33",
            "--bus-motion-window-frames",
            "7",
            "--bus-minimum-detection-confidence",
            "0.23",
            "--bus-minimum-area-change-ratio",
            "0.06",
            "--bus-max-motion-frame-gap",
            "1",
            "--bus-route-ocr-interval-frames",
            "4",
        ],
    )

    assert detect_video.main() == 0
    config = captured[0]
    assert config.ocr_language == "japan"
    assert config.ocr_model_path == "/models/japan.onnx"
    assert config.allow_ocr_download is True
    assert config.generic_vlm_model == "/models/vlm"
    assert config.generic_vlm_device == "cpu"
    assert config.generic_vlm_classes == ("unknown panel", "vending machine")
    assert config.allow_vlm_download is True
    assert config.signal_minimum_detection_confidence == pytest.approx(0.33)
    assert config.bus_motion_window_frames == 7
    assert config.bus_minimum_detection_confidence == pytest.approx(0.23)
    assert config.bus_minimum_area_change_ratio == pytest.approx(0.06)
    assert config.bus_route_ocr_interval_frames == 4
    assert config.bus_maximum_motion_frame_gap == 1


def test_cli_defaults_do_not_filter_out_bus_and_use_offline_ocr() -> None:
    args = detect_video.build_parser().parse_args(["--source", "samples/visiontest.mp4"])

    assert detect_video._parse_classes(args.classes) is None
    assert detect_video._parse_classes("5") == (5,)
    assert args.ocr_language == "default"
    assert args.bus_motion_window_frames == 9
    assert args.bus_minimum_detection_confidence == pytest.approx(0.3)
    assert args.bus_minimum_area_change_ratio == pytest.approx(0.1)
    assert args.bus_max_motion_frame_gap == 2
    assert args.bus_route_ocr_interval_frames == 7
