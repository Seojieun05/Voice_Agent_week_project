from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from .object_types import is_signal_object_type, normalize_object_type
from .types import Detection, SignalState


ImageArray = NDArray[np.uint8]
BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SignalStateResult:
    """One conservative signal-color classification and its HSV evidence."""

    state: SignalState
    confidence: float
    red_ratio: float
    green_ratio: float
    yellow_ratio: float = 0.0


class SignalStateClassifier(Protocol):
    """Interface for classifiers operating on one clamped traffic-light crop."""

    def classify(self, crop: ImageArray) -> SignalStateResult:
        """Classify a BGR crop, returning UNKNOWN when evidence is insufficient."""
        ...


@dataclass(frozen=True, slots=True)
class HsvSignalClassifierConfig:
    """Configuration for the experimental, video-specific HSV classifier."""

    roi_x_start_ratio: float = 0.08
    roi_x_end_ratio: float = 0.92
    roi_y_start_ratio: float = 0.03
    roi_y_end_ratio: float = 0.97
    red_low_hue_max: int = 12
    red_high_hue_min: int = 168
    red_min_saturation: int = 90
    red_min_value: int = 100
    green_hue_min: int = 35
    green_hue_max: int = 100
    green_min_saturation: int = 80
    green_min_value: int = 90
    yellow_hue_min: int = 18
    yellow_hue_max: int = 34
    yellow_min_saturation: int = 80
    yellow_min_value: int = 100
    minimum_roi_pixels: int = 64
    minimum_color_ratio: float = 0.015
    minimum_score_margin: float = 0.015
    minimum_dominance_ratio: float = 2.0

    def __post_init__(self) -> None:
        _validate_interval(
            "roi_x",
            self.roi_x_start_ratio,
            self.roi_x_end_ratio,
            upper_bound=1.0,
        )
        _validate_interval(
            "roi_y",
            self.roi_y_start_ratio,
            self.roi_y_end_ratio,
            upper_bound=1.0,
        )
        _validate_integer_range("red_low_hue_max", self.red_low_hue_max, 0, 179)
        _validate_integer_range("red_high_hue_min", self.red_high_hue_min, 0, 179)
        if self.red_low_hue_max >= self.red_high_hue_min:
            raise ValueError("red hue ranges must not overlap")
        _validate_integer_range("red_min_saturation", self.red_min_saturation, 0, 255)
        _validate_integer_range("red_min_value", self.red_min_value, 0, 255)
        _validate_integer_range("green_hue_min", self.green_hue_min, 0, 179)
        _validate_integer_range("green_hue_max", self.green_hue_max, 0, 179)
        if self.green_hue_min > self.green_hue_max:
            raise ValueError("green_hue_min must not exceed green_hue_max")
        _validate_integer_range("green_min_saturation", self.green_min_saturation, 0, 255)
        _validate_integer_range("green_min_value", self.green_min_value, 0, 255)
        _validate_integer_range("yellow_hue_min", self.yellow_hue_min, 0, 179)
        _validate_integer_range("yellow_hue_max", self.yellow_hue_max, 0, 179)
        if self.yellow_hue_min > self.yellow_hue_max:
            raise ValueError("yellow_hue_min must not exceed yellow_hue_max")
        _validate_integer_range("yellow_min_saturation", self.yellow_min_saturation, 0, 255)
        _validate_integer_range("yellow_min_value", self.yellow_min_value, 0, 255)
        if (
            not isinstance(self.minimum_roi_pixels, int)
            or isinstance(self.minimum_roi_pixels, bool)
            or self.minimum_roi_pixels < 1
        ):
            raise ValueError("minimum_roi_pixels must be a positive integer")
        _validate_fraction("minimum_color_ratio", self.minimum_color_ratio, positive=True)
        _validate_fraction("minimum_score_margin", self.minimum_score_margin)
        if not math.isfinite(self.minimum_dominance_ratio) or self.minimum_dominance_ratio < 1.0:
            raise ValueError("minimum_dominance_ratio must be finite and at least 1")


@dataclass(frozen=True, slots=True)
class SignalTargetSelectorConfig:
    """Rules for choosing one stable pedestrian-signal box per spatial group."""

    traffic_light_class_id: int = 9
    preferred_min_aspect_ratio: float = 1.5
    preferred_max_aspect_ratio: float = 2.3
    maximum_component_aspect_ratio: float = 1.25
    minimum_horizontal_overlap_ratio: float = 0.5
    minimum_box_overlap_ratio: float = 0.5
    maximum_vertical_gap_width_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.traffic_light_class_id < 0:
            raise ValueError("traffic_light_class_id must be non-negative")
        if (
            not math.isfinite(self.preferred_min_aspect_ratio)
            or not math.isfinite(self.preferred_max_aspect_ratio)
            or not 0 < self.preferred_min_aspect_ratio <= self.preferred_max_aspect_ratio
        ):
            raise ValueError("preferred aspect ratios must be finite, positive, and ordered")
        if (
            not math.isfinite(self.maximum_component_aspect_ratio)
            or self.maximum_component_aspect_ratio <= 0
        ):
            raise ValueError("maximum_component_aspect_ratio must be finite and positive")
        _validate_fraction(
            "minimum_horizontal_overlap_ratio",
            self.minimum_horizontal_overlap_ratio,
            positive=True,
        )
        _validate_fraction(
            "minimum_box_overlap_ratio",
            self.minimum_box_overlap_ratio,
            positive=True,
        )
        if (
            not math.isfinite(self.maximum_vertical_gap_width_ratio)
            or self.maximum_vertical_gap_width_ratio < 0
        ):
            raise ValueError("maximum_vertical_gap_width_ratio must be finite and non-negative")


def _validate_interval(name: str, start: float, end: float, *, upper_bound: float) -> None:
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError(f"{name} ratios must be finite")
    if not 0.0 <= start < end <= upper_bound:
        raise ValueError(f"{name} ratios must satisfy 0 <= start < end <= {upper_bound}")


def _validate_integer_range(name: str, value: int, lower: int, upper: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer from {lower} through {upper}")


def _validate_fraction(name: str, value: float, *, positive: bool = False) -> None:
    lower_is_valid = value > 0.0 if positive else value >= 0.0
    if not math.isfinite(value) or not lower_is_valid or value > 1.0:
        qualifier = "0 < value <= 1" if positive else "0 <= value <= 1"
        raise ValueError(f"{name} must be finite and satisfy {qualifier}")


def crop_frame_to_bbox(frame: ImageArray, xyxy: BBox) -> ImageArray | None:
    """Clamp an xyxy box to a frame and return its crop, or None when empty."""
    if frame.ndim < 2 or frame.size == 0:
        return None

    try:
        x1, y1, x2, y2 = (float(value) for value in xyxy)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None

    height, width = frame.shape[:2]
    left = max(0, min(width, math.floor(x1)))
    top = max(0, min(height, math.floor(y1)))
    right = max(0, min(width, math.ceil(x2)))
    bottom = max(0, min(height, math.ceil(y2)))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def _box_dimensions(xyxy: BBox) -> tuple[float, float]:
    return max(0.0, xyxy[2] - xyxy[0]), max(0.0, xyxy[3] - xyxy[1])


def _box_aspect_ratio(xyxy: BBox) -> float:
    width, height = _box_dimensions(xyxy)
    return height / width if width > 0 else math.inf


def _is_preferred_signal_box(
    xyxy: BBox,
    config: SignalTargetSelectorConfig,
) -> bool:
    aspect_ratio = _box_aspect_ratio(xyxy)
    return config.preferred_min_aspect_ratio <= aspect_ratio <= config.preferred_max_aspect_ratio


def _boxes_belong_to_same_signal(
    first: BBox,
    second: BBox,
    config: SignalTargetSelectorConfig,
) -> bool:
    first_width, first_height = _box_dimensions(first)
    second_width, second_height = _box_dimensions(second)
    minimum_width = min(first_width, second_width)
    first_area = first_width * first_height
    second_area = second_width * second_height
    minimum_area = min(first_area, second_area)
    if minimum_width <= 0 or minimum_area <= 0:
        return False

    horizontal_overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    overlap_ratio = horizontal_overlap / minimum_width
    if overlap_ratio < config.minimum_horizontal_overlap_ratio:
        return False

    vertical_overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    box_overlap_ratio = horizontal_overlap * vertical_overlap / minimum_area
    if box_overlap_ratio >= config.minimum_box_overlap_ratio:
        return True

    vertical_gap = max(0.0, max(first[1], second[1]) - min(first[3], second[3]))
    boxes_are_display_and_component = (
        _is_preferred_signal_box(first, config)
        and _box_aspect_ratio(second) <= config.maximum_component_aspect_ratio
    ) or (
        _is_preferred_signal_box(second, config)
        and _box_aspect_ratio(first) <= config.maximum_component_aspect_ratio
    )
    return (
        boxes_are_display_and_component
        and vertical_gap <= minimum_width * config.maximum_vertical_gap_width_ratio
    )


class SignalTargetSelector:
    """Suppress component duplicates while retaining previously selected raw tracks."""

    def __init__(self, config: SignalTargetSelectorConfig | None = None) -> None:
        self.config = config or SignalTargetSelectorConfig()
        self._selected_track_ids: set[int] = set()

    def reset(self) -> None:
        """Forget raw-track preferences from the previous stream or session."""
        self._selected_track_ids.clear()

    def _preferred(self, detection: Detection) -> bool:
        return _is_preferred_signal_box(detection.xyxy, self.config)

    def is_signal_detection(self, detection: Detection) -> bool:
        """Match model-independent signal names, with blank-name legacy fallback."""
        return is_signal_object_type(detection.class_name) or (
            not detection.class_name.strip()
            and detection.class_id == self.config.traffic_light_class_id
        )

    def select_indices(self, detections: list[Detection]) -> list[int]:
        """Return one selected traffic-light index per connected spatial group."""
        traffic_indices = [
            index
            for index, detection in enumerate(detections)
            if self.is_signal_detection(detection)
        ]
        groups: list[list[int]] = []
        for detection_index in traffic_indices:
            matching_groups = [
                group_index
                for group_index, group in enumerate(groups)
                if any(
                    normalize_object_type(detections[detection_index].class_name)
                    == normalize_object_type(detections[other_index].class_name)
                    and _boxes_belong_to_same_signal(
                        detections[detection_index].xyxy,
                        detections[other_index].xyxy,
                        self.config,
                    )
                    for other_index in group
                )
            ]
            if not matching_groups:
                groups.append([detection_index])
                continue

            target_group = groups[matching_groups[0]]
            target_group.append(detection_index)
            for group_index in reversed(matching_groups[1:]):
                target_group.extend(groups.pop(group_index))

        selected_indices: list[int] = []
        selected_track_ids: set[int] = set()
        for group in groups:
            locked = [
                index for index in group if detections[index].track_id in self._selected_track_ids
            ]
            preferred = [index for index in group if self._preferred(detections[index])]
            candidates = locked or preferred or group
            selected_index = max(
                candidates,
                key=lambda index: (
                    detections[index].confidence,
                    _box_dimensions(detections[index].xyxy)[0]
                    * _box_dimensions(detections[index].xyxy)[1],
                    -index,
                ),
            )
            selected_indices.append(selected_index)
            track_id = detections[selected_index].track_id
            if track_id is not None:
                selected_track_ids.add(track_id)

        self._selected_track_ids = selected_track_ids
        return sorted(selected_indices)


class HsvSignalStateClassifier:
    """Experimental RED/GREEN/YELLOW classifier tuned from supplied videos.

    This rule-based implementation consumes OpenCV BGR uint8 crops. It deliberately
    returns UNKNOWN for weak or competing color evidence.
    """

    def __init__(self, config: HsvSignalClassifierConfig | None = None) -> None:
        self.config = config or HsvSignalClassifierConfig()

    @staticmethod
    def _unknown(
        *,
        red_ratio: float = 0.0,
        green_ratio: float = 0.0,
        yellow_ratio: float = 0.0,
    ) -> SignalStateResult:
        return SignalStateResult(
            state=SignalState.UNKNOWN,
            confidence=0.0,
            red_ratio=red_ratio,
            green_ratio=green_ratio,
            yellow_ratio=yellow_ratio,
        )

    def classify(self, crop: ImageArray) -> SignalStateResult:
        if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3 or crop.size == 0:
            return self._unknown()

        height, width = crop.shape[:2]
        config = self.config
        left = math.floor(width * config.roi_x_start_ratio)
        right = math.ceil(width * config.roi_x_end_ratio)
        top = math.floor(height * config.roi_y_start_ratio)
        bottom = math.ceil(height * config.roi_y_end_ratio)
        roi = crop[top:bottom, left:right]
        if roi.size == 0 or roi.shape[0] * roi.shape[1] < config.minimum_roi_pixels:
            return self._unknown()

        hue, saturation, value = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
        red_mask = (
            ((hue <= config.red_low_hue_max) | (hue >= config.red_high_hue_min))
            & (saturation >= config.red_min_saturation)
            & (value >= config.red_min_value)
        )
        green_mask = (
            (hue >= config.green_hue_min)
            & (hue <= config.green_hue_max)
            & (saturation >= config.green_min_saturation)
            & (value >= config.green_min_value)
        )
        yellow_mask = (
            (hue >= config.yellow_hue_min)
            & (hue <= config.yellow_hue_max)
            & (saturation >= config.yellow_min_saturation)
            & (value >= config.yellow_min_value)
        )
        red_ratio = float(np.count_nonzero(red_mask) / red_mask.size)
        green_ratio = float(np.count_nonzero(green_mask) / green_mask.size)
        yellow_ratio = float(np.count_nonzero(yellow_mask) / yellow_mask.size)

        color_ratios = {
            SignalState.RED: red_ratio,
            SignalState.GREEN: green_ratio,
            SignalState.YELLOW: yellow_ratio,
        }
        state = max(color_ratios, key=color_ratios.__getitem__)
        winner_ratio = color_ratios[state]
        loser_ratio = max(
            ratio for candidate, ratio in color_ratios.items() if candidate is not state
        )

        has_enough_evidence = winner_ratio >= config.minimum_color_ratio
        has_enough_margin = winner_ratio - loser_ratio >= config.minimum_score_margin
        is_dominant = winner_ratio >= config.minimum_dominance_ratio * loser_ratio
        if not (has_enough_evidence and has_enough_margin and is_dominant):
            return self._unknown(
                red_ratio=red_ratio,
                green_ratio=green_ratio,
                yellow_ratio=yellow_ratio,
            )

        confidence = winner_ratio / sum(color_ratios.values())
        return SignalStateResult(
            state=state,
            confidence=confidence,
            red_ratio=red_ratio,
            green_ratio=green_ratio,
            yellow_ratio=yellow_ratio,
        )
