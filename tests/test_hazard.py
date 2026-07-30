from __future__ import annotations

import math

import pytest

from vision_agent.hazard import (
    ApproachMonitor,
    HazardLevel,
    HazardZone,
)
from vision_agent.types import Detection

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def _box(area_ratio: float, *, center_x: float | None = None) -> tuple[float, float, float, float]:
    """Box occupying ``area_ratio`` of the frame, centered unless told otherwise."""
    scale = math.sqrt(area_ratio)
    width = FRAME_WIDTH * scale
    height = FRAME_HEIGHT * scale
    cx = FRAME_WIDTH / 2.0 if center_x is None else center_x
    cy = FRAME_HEIGHT / 2.0
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def _detection(
    area_ratio: float,
    *,
    class_name: str = "bicycle",
    confidence: float = 0.9,
    center_x: float | None = None,
) -> Detection:
    return Detection(
        frame_index=0,
        timestamp_s=0.0,
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        xyxy=_box(area_ratio, center_x=center_x),
    )


def _area_for_ttc(ttc_s: float, elapsed_s: float, *, initial: float = 0.02) -> float:
    """Area ratio at ``elapsed_s`` for a constant time-to-contact trajectory.

    ``TTC = 2 / growth_rate``, so a fixed TTC means a fixed exponential rate.
    """
    return initial * math.exp((2.0 / ttc_s) * elapsed_s)


def _feed(
    monitor: ApproachMonitor,
    areas: list[float],
    *,
    dt_s: float = 0.2,
    class_name: str = "bicycle",
    confidence: float = 0.9,
    center_x: float | None = None,
    stable_id: str = "obj-1",
) -> list[tuple[float, list]]:
    """Push one area per frame and collect (timestamp, assessments) per frame."""
    emitted = []
    for index, area in enumerate(areas):
        timestamp_s = index * dt_s
        detection = _detection(
            area,
            class_name=class_name,
            confidence=confidence,
            center_x=center_x,
        )
        assessments = monitor.update(
            [(stable_id, detection)],
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            timestamp_s=timestamp_s,
        )
        emitted.append((timestamp_s, assessments))
    return emitted


def _first_hit(frames: list[tuple[float, list]]):
    for timestamp_s, assessments in frames:
        if assessments:
            return timestamp_s, assessments[0]
    return None, None


def test_fast_approach_is_imminent():
    monitor = ApproachMonitor()
    areas = [_area_for_ttc(1.0, index * 0.2) for index in range(3)]

    timestamp_s, hazard = _first_hit(_feed(monitor, areas))

    assert hazard is not None
    # 3개 샘플이 모여 최소 시간 간격을 넘긴 프레임에서 처음 판정된다
    assert timestamp_s == pytest.approx(0.4)
    assert hazard.level is HazardLevel.IMMINENT
    assert hazard.time_to_contact_s == pytest.approx(1.0, rel=1e-6)
    assert hazard.zone is HazardZone.CENTER
    assert hazard.in_path is True


def test_slow_approach_is_only_a_warning():
    monitor = ApproachMonitor()
    areas = [_area_for_ttc(3.0, index * 0.2) for index in range(3)]

    _, hazard = _first_hit(_feed(monitor, areas))

    assert hazard is not None
    assert hazard.level is HazardLevel.WARNING
    assert hazard.time_to_contact_s == pytest.approx(3.0, rel=1e-6)


def test_object_moving_away_is_not_a_hazard():
    monitor = ApproachMonitor()
    areas = [_area_for_ttc(1.0, (4 - index) * 0.2) for index in range(5)]

    frames = _feed(monitor, areas)

    assert all(not assessments for _, assessments in frames)


def test_steady_object_is_not_a_hazard():
    monitor = ApproachMonitor()

    frames = _feed(monitor, [0.05] * 6)

    assert all(not assessments for _, assessments in frames)


def test_distant_approach_beyond_the_warning_horizon_is_ignored():
    monitor = ApproachMonitor(warning_ttc_s=3.5)
    areas = [_area_for_ttc(8.0, index * 0.2) for index in range(6)]

    frames = _feed(monitor, areas)

    assert all(not assessments for _, assessments in frames)


def test_tiny_far_box_is_ignored():
    monitor = ApproachMonitor(minimum_area_ratio=0.005)
    # 빠르게 커지고 있지만 여전히 프레임의 0.5% 미만 — 성장분이 대부분 박스 떨림이다
    areas = [_area_for_ttc(1.0, index * 0.2, initial=0.0005) for index in range(4)]

    frames = _feed(monitor, areas)

    assert all(not assessments for _, assessments in frames)


def test_jittery_growth_is_rejected():
    monitor = ApproachMonitor()
    # 전체 추세는 증가지만 프레임 간 방향이 엇갈린다
    areas = [0.02, 0.05, 0.03, 0.06]

    frames = _feed(monitor, areas)

    assert all(not assessments for _, assessments in frames)


def test_static_obstacle_warns_only_when_contact_is_imminent():
    slow = ApproachMonitor()
    slow_frames = _feed(
        slow,
        [_area_for_ttc(3.0, index * 0.2) for index in range(4)],
        class_name="bollard",
    )
    assert all(not assessments for _, assessments in slow_frames)

    fast = ApproachMonitor()
    _, hazard = _first_hit(
        _feed(
            fast,
            [_area_for_ttc(1.0, index * 0.2) for index in range(3)],
            class_name="bollard",
        )
    )
    assert hazard is not None
    assert hazard.level is HazardLevel.IMMINENT
    assert hazard.object_type == "bollard"


def test_off_path_object_warns_only_when_imminent():
    slow = ApproachMonitor()
    slow_frames = _feed(
        slow,
        [_area_for_ttc(3.0, index * 0.2) for index in range(4)],
        center_x=60.0,
    )
    assert all(not assessments for _, assessments in slow_frames)

    fast = ApproachMonitor()
    _, hazard = _first_hit(
        _feed(
            fast,
            [_area_for_ttc(1.0, index * 0.2) for index in range(3)],
            center_x=60.0,
        )
    )
    assert hazard is not None
    assert hazard.level is HazardLevel.IMMINENT
    assert hazard.in_path is False
    assert hazard.zone is HazardZone.LEFT


def test_zone_follows_horizontal_position():
    right = ApproachMonitor()
    _, hazard = _first_hit(
        _feed(
            right,
            [_area_for_ttc(1.0, index * 0.2) for index in range(3)],
            center_x=580.0,
        )
    )
    assert hazard is not None
    assert hazard.zone is HazardZone.RIGHT


def test_low_confidence_detection_is_dropped():
    monitor = ApproachMonitor(minimum_confidence=0.35)

    frames = _feed(
        monitor,
        [_area_for_ttc(1.0, index * 0.2) for index in range(4)],
        confidence=0.3,
    )

    assert all(not assessments for _, assessments in frames)


def test_repeat_is_throttled_while_the_hazard_persists():
    monitor = ApproachMonitor(repeat_interval_s=1.0)
    areas = [_area_for_ttc(3.0, index * 0.2) for index in range(12)]

    hits = [timestamp_s for timestamp_s, assessments in _feed(monitor, areas) if assessments]

    assert hits, "지속되는 위험은 최소 한 번은 알려야 한다"
    gaps = [later - earlier for earlier, later in zip(hits, hits[1:])]
    assert all(gap >= 1.0 for gap in gaps), f"반복 간격이 좁다: {hits}"


def test_escalation_to_imminent_skips_the_repeat_interval():
    monitor = ApproachMonitor(repeat_interval_s=2.0)
    warning_areas = [_area_for_ttc(3.0, index * 0.2) for index in range(3)]
    frames = _feed(monitor, warning_areas)
    warning_at, warning = _first_hit(frames)
    assert warning is not None
    assert warning.level is HazardLevel.WARNING

    # 0.2초 뒤 급격히 커진다. 반복 간격(2초)은 안 지났지만 등급이 올라갔으므로 바로 알린다.
    escalated = monitor.update(
        [("obj-1", _detection(0.06))],
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        timestamp_s=warning_at + 0.2,
    )

    assert escalated
    assert escalated[0].level is HazardLevel.IMMINENT


def test_most_urgent_hazard_comes_first():
    monitor = ApproachMonitor()
    slow = [_area_for_ttc(3.0, index * 0.2) for index in range(3)]
    fast = [_area_for_ttc(1.0, index * 0.2) for index in range(3)]

    assessments: list = []
    for index in range(3):
        assessments = monitor.update(
            [
                ("slow", _detection(slow[index], center_x=300.0)),
                ("fast", _detection(fast[index], center_x=340.0)),
            ],
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            timestamp_s=index * 0.2,
        )

    assert [item.stable_id for item in assessments] == ["fast", "slow"]


def test_history_is_dropped_when_the_object_leaves_the_frame():
    monitor = ApproachMonitor()
    _feed(monitor, [_area_for_ttc(1.0, index * 0.2) for index in range(2)])

    # 빈 프레임이 지나가면 이전 이력은 버려진다
    monitor.update([], frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT, timestamp_s=0.4)

    # 같은 stable_id로 큰 박스가 돌아와도 곧바로 위험이 되지 않는다
    reappeared = monitor.update(
        [("obj-1", _detection(0.3))],
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        timestamp_s=0.6,
    )

    assert not reappeared


def test_reset_clears_history():
    monitor = ApproachMonitor()
    _feed(monitor, [_area_for_ttc(1.0, index * 0.2) for index in range(2)])

    monitor.reset("obj-1")

    resumed = monitor.update(
        [("obj-1", _detection(0.3))],
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        timestamp_s=0.4,
    )
    assert not resumed


def test_duplicate_timestamp_does_not_divide_by_zero():
    monitor = ApproachMonitor()
    for index in range(3):
        monitor.update(
            [("obj-1", _detection(_area_for_ttc(1.0, index * 0.2)))],
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            timestamp_s=index * 0.2,
        )

    repeated = monitor.update(
        [("obj-1", _detection(_area_for_ttc(1.0, 0.4)))],
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        timestamp_s=0.4,
    )

    assert isinstance(repeated, list)


def test_degenerate_frame_size_is_rejected():
    monitor = ApproachMonitor()

    assert monitor.update(
        [("obj-1", _detection(0.05))],
        frame_width=0,
        frame_height=480,
        timestamp_s=0.0,
    ) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"history_samples": 1},
        {"minimum_samples": 1},
        {"minimum_samples": 99},
        {"minimum_span_s": 0.0},
        {"warning_ttc_s": 0.0},
        {"imminent_ttc_s": 5.0, "warning_ttc_s": 1.0},
        {"minimum_area_ratio": 1.0},
        {"corridor_ratio": 0.0},
        {"minimum_corridor_overlap": 1.5},
        {"minimum_growth_consistency": -0.1},
        {"repeat_interval_s": -1.0},
        {"minimum_confidence": 1.5},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ApproachMonitor(**kwargs)


# --- 파이프라인 통합 -------------------------------------------------------


class _FakeTensor:
    def __init__(self, values: object) -> None:
        self._values = values

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def tolist(self) -> object:
        return self._values


class _Boxes:
    def __init__(self, xyxy: tuple[float, float, float, float]) -> None:
        self.xyxy = _FakeTensor([list(xyxy)])
        self.conf = _FakeTensor([0.9])
        self.cls = _FakeTensor([0.0])
        self.id = _FakeTensor([1.0])

    def __len__(self) -> int:
        return 1


class _Result:
    names = {0: "bicycle"}

    def __init__(self, xyxy: tuple[float, float, float, float]) -> None:
        self.boxes = _Boxes(xyxy)


class _GrowingBicycleModel:
    """Fake detector whose single bicycle box closes in on the camera."""

    def __init__(self, areas: list[float]) -> None:
        self._areas = areas
        self.calls = 0

    def track(self, frame, **kwargs) -> list[_Result]:
        area = self._areas[min(self.calls, len(self._areas) - 1)]
        self.calls += 1
        return [_Result(_box(area))]


def test_pipeline_emits_a_hazard_narration_for_a_closing_bicycle():
    import numpy as np

    from vision_agent.hazard import HAZARD_DETECTED
    from vision_agent.pipeline import FrameContext, PipelineConfig, create_vision_session

    areas = [_area_for_ttc(1.0, index * 0.2) for index in range(4)]
    session = create_vision_session(
        PipelineConfig(
            source="unused",
            min_seen_frames=1,
            classify_signal_states=False,
            allow_ocr_download=False,
        ),
        model=_GrowingBicycleModel(areas),
    )
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    narrations: list[str] = []
    event_types: set[str] = set()
    for index in range(len(areas)):
        analysis = session.process_frame(
            frame,
            FrameContext(
                source_sequence_id=index,
                processed_index=index,
                captured_at_s=index * 0.2,
                received_at_s=index * 0.2,
                processing_started_at_s=index * 0.2,
            ),
        )
        narrations.extend(analysis.narrations)
        event_types.update(item.event_type for item in analysis.analysis_events)

    assert HAZARD_DETECTED in event_types
    assert any(message.startswith("위험, 멈추세요.") for message in narrations), narrations


def test_pipeline_stays_quiet_when_hazard_detection_is_disabled():
    import numpy as np

    from vision_agent.hazard import HAZARD_DETECTED
    from vision_agent.pipeline import FrameContext, PipelineConfig, create_vision_session

    areas = [_area_for_ttc(1.0, index * 0.2) for index in range(4)]
    session = create_vision_session(
        PipelineConfig(
            source="unused",
            min_seen_frames=1,
            classify_signal_states=False,
            allow_ocr_download=False,
            hazard_detection_enabled=False,
        ),
        model=_GrowingBicycleModel(areas),
    )
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    event_types: set[str] = set()
    for index in range(len(areas)):
        analysis = session.process_frame(
            frame,
            FrameContext(
                source_sequence_id=index,
                processed_index=index,
                captured_at_s=index * 0.2,
                received_at_s=index * 0.2,
                processing_started_at_s=index * 0.2,
            ),
        )
        event_types.update(item.event_type for item in analysis.analysis_events)

    assert HAZARD_DETECTED not in event_types
    assert session.approach_monitor is None
