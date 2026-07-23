from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vision_agent.signals import (
    HsvSignalClassifierConfig,
    HsvSignalStateClassifier,
    SignalTargetSelector,
    SignalTargetSelectorConfig,
    crop_frame_to_bbox,
)
from vision_agent.types import Detection, SignalState


def signal_crop(color: tuple[int, int, int] | None = None) -> np.ndarray:
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    if color is not None:
        crop[3:97, 8:92] = color
    return crop


def detection(
    track_id: int | None,
    xyxy: tuple[float, float, float, float],
    *,
    confidence: float,
    class_id: int = 9,
) -> Detection:
    return Detection(
        frame_index=0,
        timestamp_s=0.0,
        class_id=class_id,
        class_name="traffic light" if class_id == 9 else "person",
        confidence=confidence,
        xyxy=xyxy,
        track_id=track_id,
    )


def test_classifies_red_signal() -> None:
    result = HsvSignalStateClassifier().classify(signal_crop((0, 0, 255)))

    assert result.state is SignalState.RED
    assert result.red_ratio == pytest.approx(1.0)
    assert result.green_ratio == 0.0
    assert result.confidence == pytest.approx(1.0)


def test_classifies_green_signal() -> None:
    result = HsvSignalStateClassifier().classify(signal_crop((0, 255, 0)))

    assert result.state is SignalState.GREEN
    assert result.green_ratio == pytest.approx(1.0)
    assert result.red_ratio == 0.0
    assert result.confidence == pytest.approx(1.0)


def test_classifies_yellow_signal() -> None:
    result = HsvSignalStateClassifier().classify(signal_crop((0, 255, 255)))

    assert result.state is SignalState.YELLOW
    assert result.yellow_ratio == pytest.approx(1.0)
    assert result.red_ratio == 0.0
    assert result.green_ratio == 0.0
    assert result.confidence == pytest.approx(1.0)


def test_orange_halo_does_not_turn_strong_red_evidence_unknown() -> None:
    crop = signal_crop()
    crop[3:43, 8:50] = (0, 0, 255)
    crop[43:67, 8:50] = (0, 128, 255)

    result = HsvSignalStateClassifier().classify(crop)

    assert result.state is SignalState.RED
    assert result.red_ratio > 0.20
    assert result.yellow_ratio == 0.0


def test_returns_unknown_without_color_evidence() -> None:
    result = HsvSignalStateClassifier().classify(signal_crop())

    assert result.state is SignalState.UNKNOWN
    assert result.confidence == 0.0
    assert result.red_ratio == 0.0
    assert result.green_ratio == 0.0


def test_returns_unknown_for_competing_red_and_green() -> None:
    crop = signal_crop()
    crop[3:70, 8:50] = (0, 0, 255)
    crop[3:70, 50:92] = (0, 255, 0)

    result = HsvSignalStateClassifier().classify(crop)

    assert result.state is SignalState.UNKNOWN
    assert result.confidence == 0.0
    assert result.red_ratio == pytest.approx(result.green_ratio)


def test_ignores_color_outside_pedestrian_signal_roi() -> None:
    crop = signal_crop()
    crop[98:, :] = (0, 255, 0)

    result = HsvSignalStateClassifier().classify(crop)

    assert result.state is SignalState.UNKNOWN


def test_classifies_sparse_green_signal_in_lower_lamp() -> None:
    crop = signal_crop()
    crop[75:95, 40:48] = (0, 255, 0)

    result = HsvSignalStateClassifier().classify(crop)

    assert result.state is SignalState.GREEN
    assert result.green_ratio > 0.015


def test_returns_unknown_below_minimum_sparse_color_ratio() -> None:
    crop = signal_crop()
    crop[75:85, 40:50] = (0, 255, 0)

    result = HsvSignalStateClassifier().classify(crop)

    assert result.state is SignalState.UNKNOWN
    assert 0.0 < result.green_ratio < 0.015


@pytest.mark.parametrize(
    "crop",
    [
        np.empty((0, 0, 3), dtype=np.uint8),
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
    ],
)
def test_returns_unknown_for_invalid_crop(crop: np.ndarray) -> None:
    assert HsvSignalStateClassifier().classify(crop).state is SignalState.UNKNOWN


def test_returns_unknown_when_roi_has_too_few_pixels() -> None:
    tiny_red_crop = np.full((1, 1, 3), (0, 0, 255), dtype=np.uint8)

    assert HsvSignalStateClassifier().classify(tiny_red_crop).state is SignalState.UNKNOWN


def test_thresholds_are_configurable() -> None:
    crop = signal_crop()
    crop[3:70, 8:25] = (0, 0, 255)
    strict = HsvSignalStateClassifier(HsvSignalClassifierConfig(minimum_color_ratio=0.25))
    permissive = HsvSignalStateClassifier(HsvSignalClassifierConfig(minimum_color_ratio=0.10))

    assert strict.classify(crop).state is SignalState.UNKNOWN
    assert permissive.classify(crop).state is SignalState.RED


def test_crop_frame_to_bbox_clamps_coordinates() -> None:
    frame = np.arange(5 * 6 * 3, dtype=np.uint8).reshape((5, 6, 3))

    crop = crop_frame_to_bbox(frame, (-3.0, -2.0, 3.2, 2.2))

    assert crop is not None
    assert np.array_equal(crop, frame[0:3, 0:4])


@pytest.mark.parametrize(
    "xyxy",
    [
        (10.0, 10.0, 20.0, 20.0),
        (4.0, 4.0, 2.0, 2.0),
        (0.0, 0.0, float("nan"), 2.0),
        (0.0, 0.0, float("inf"), 2.0),
    ],
)
def test_crop_frame_to_bbox_returns_none_for_empty_or_invalid_box(
    xyxy: tuple[float, float, float, float],
) -> None:
    frame = np.zeros((5, 6, 3), dtype=np.uint8)

    assert crop_frame_to_bbox(frame, xyxy) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("roi_x_start_ratio", -0.1),
        ("roi_x_end_ratio", 1.1),
        ("roi_y_start_ratio", 0.97),
        ("red_low_hue_max", 180),
        ("red_high_hue_min", -1),
        ("red_min_saturation", 256),
        ("green_hue_min", -1),
        ("green_hue_max", 180),
        ("green_min_value", 256),
        ("yellow_hue_min", -1),
        ("yellow_hue_max", 180),
        ("yellow_min_value", 256),
        ("minimum_roi_pixels", 0),
        ("minimum_roi_pixels", True),
        ("minimum_color_ratio", 0.0),
        ("minimum_score_margin", 1.1),
        ("minimum_dominance_ratio", 0.9),
        ("minimum_dominance_ratio", float("nan")),
    ],
)
def test_invalid_classifier_configuration_is_rejected(field: str, value: object) -> None:
    defaults = HsvSignalClassifierConfig()

    with pytest.raises(ValueError):
        replace(defaults, **{field: value})


def test_target_selector_suppresses_stacked_component_boxes() -> None:
    selector = SignalTargetSelector()
    detections = [
        detection(1, (10.0, 0.0, 30.0, 40.0), confidence=0.7),
        detection(2, (10.0, 42.0, 30.0, 62.0), confidence=0.9),
        detection(3, (9.0, 0.0, 31.0, 63.0), confidence=0.6),
    ]

    assert selector.select_indices(detections) == [0]


def test_target_selector_keeps_previous_raw_track() -> None:
    selector = SignalTargetSelector()
    first = [
        detection(1, (10.0, 0.0, 30.0, 40.0), confidence=0.7),
        detection(2, (10.0, 42.0, 30.0, 62.0), confidence=0.5),
    ]
    second = [
        detection(1, (11.0, 0.0, 31.0, 40.0), confidence=0.2),
        detection(4, (10.0, 0.0, 30.0, 42.0), confidence=0.95),
    ]

    assert selector.select_indices(first) == [0]
    assert selector.select_indices(second) == [0]


def test_target_selector_reset_forgets_previous_raw_track() -> None:
    selector = SignalTargetSelector()
    first = [
        detection(1, (10.0, 0.0, 30.0, 40.0), confidence=0.7),
        detection(2, (10.0, 42.0, 30.0, 62.0), confidence=0.5),
    ]
    second = [
        detection(1, (11.0, 0.0, 31.0, 40.0), confidence=0.2),
        detection(4, (10.0, 0.0, 30.0, 42.0), confidence=0.95),
    ]

    assert selector.select_indices(first) == [0]
    selector.reset()

    assert selector.select_indices(second) == [1]


def test_target_selector_keeps_spatially_separate_signals() -> None:
    selector = SignalTargetSelector()
    detections = [
        detection(1, (0.0, 0.0, 20.0, 40.0), confidence=0.8),
        detection(2, (100.0, 0.0, 120.0, 40.0), confidence=0.7),
        detection(None, (0.0, 42.0, 20.0, 62.0), confidence=0.9),
        detection(7, (0.0, 0.0, 10.0, 20.0), confidence=0.99, class_id=0),
    ]

    assert selector.select_indices(detections) == [0, 1]


def test_target_selector_does_not_merge_vertically_separate_signals() -> None:
    selector = SignalTargetSelector()
    detections = [
        detection(1, (0.0, 0.0, 20.0, 40.0), confidence=0.8),
        detection(2, (0.0, 42.0, 20.0, 82.0), confidence=0.7),
    ]

    assert selector.select_indices(detections) == [0, 1]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("traffic_light_class_id", -1),
        ("preferred_min_aspect_ratio", 0.0),
        ("preferred_max_aspect_ratio", float("inf")),
        ("maximum_component_aspect_ratio", 0.0),
        ("minimum_horizontal_overlap_ratio", 0.0),
        ("minimum_box_overlap_ratio", 0.0),
        ("maximum_vertical_gap_width_ratio", -0.1),
    ],
)
def test_invalid_target_selector_configuration_is_rejected(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        replace(SignalTargetSelectorConfig(), **{field: value})
