from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from vision_agent.analyzers.bus import BusAnalyzer
from vision_agent.event_manager import OBJECT_APPROACHING, SceneEventManager
from vision_agent.ocr import OcrLine, OcrResult
from vision_agent.types import AnalysisResult, Detection


def bus_detection(
    frame_index: int,
    *,
    center_x: float = 50.0,
    center_y: float = 50.0,
    width: float = 20.0,
    height: float = 40.0,
    confidence: float = 0.95,
) -> Detection:
    return Detection(
        frame_index=frame_index,
        timestamp_s=frame_index / 30.0,
        class_id=5,
        class_name="bus",
        confidence=confidence,
        xyxy=(
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ),
        track_id=7,
    )


def ocr_result(
    text: str,
    confidence: float = 0.9,
    *,
    available: bool = True,
) -> OcrResult:
    lines = (OcrLine(text=text, confidence=confidence),) if text else ()
    return OcrResult(
        lines=lines,
        engine_name="fake-ocr",
        is_available=available,
    )


class FakeOcrEngine:
    def __init__(self, results: Sequence[OcrResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[int, ...]] = []

    def recognize(self, image: np.ndarray) -> OcrResult:
        self.calls.append(image.shape)
        if not self.results:
            raise AssertionError("unexpected OCR call")
        return self.results.pop(0)


def analyze_sizes(
    analyzer: BusAnalyzer,
    sizes: Sequence[tuple[float, float]],
    *,
    stable_id: str = "stable-7",
    start_frame: int = 0,
) -> list[AnalysisResult]:
    return [
        analyzer.analyze(
            bus_detection(frame_index, width=width, height=height),
            stable_id=stable_id,
        )
        for frame_index, (width, height) in enumerate(sizes, start=start_frame)
    ]


def test_approaching_is_confirmed_from_consecutive_area_growth() -> None:
    results = analyze_sizes(
        BusAnalyzer(),
        [(20, 40), (22, 44), (24, 48), (26, 52)],
    )

    assert [result.state for result in results] == [
        "UNKNOWN",
        "UNKNOWN",
        "UNKNOWN",
        "APPROACHING",
    ]
    assert results[-1].confidence > 0.0
    assert results[-1].attributes["motion_observed_state"] == "APPROACHING"
    assert results[-1].attributes["motion_confirmed_frames"] == 2
    assert results[-1].is_uncertain is False


def test_receding_is_confirmed_from_consecutive_area_shrinkage() -> None:
    results = analyze_sizes(
        BusAnalyzer(),
        [(28, 56), (25, 50), (22, 44), (19, 38)],
    )

    assert results[-1].state == "RECEDING"
    assert results[-1].attributes["area_change_ratio"] < 0.0


def test_stopped_requires_both_stable_area_and_stable_center() -> None:
    stopped_analyzer = BusAnalyzer()
    stopped = [
        stopped_analyzer.analyze(
            bus_detection(frame_index),
            stable_id="stopped",
        )
        for frame_index in range(4)
    ]
    moving_analyzer = BusAnalyzer()
    moving_sideways = [
        moving_analyzer.analyze(
            bus_detection(frame_index, center_x=50.0 + 5.0 * frame_index),
            stable_id="moving",
        )
        for frame_index in range(4)
    ]

    assert stopped[-1].state == "STOPPED"
    assert moving_sideways[-1].state == "UNKNOWN"
    assert moving_sideways[-1].is_uncertain is True


def test_center_direction_conflict_prevents_approaching_guess() -> None:
    analyzer = BusAnalyzer()
    results = [
        analyzer.analyze(
            bus_detection(
                frame_index,
                center_y=80.0 - frame_index * 8.0,
                width=20.0 + frame_index * 2.0,
                height=40.0 + frame_index * 4.0,
            ),
            stable_id="stable-conflict",
        )
        for frame_index in range(5)
    ]

    assert all(result.state == "UNKNOWN" for result in results)


def test_frame_gap_breaks_motion_confirmation_streak() -> None:
    analyzer = BusAnalyzer()
    analyzer.analyze(bus_detection(0, width=20, height=40), stable_id="stable-7")
    analyzer.analyze(bus_detection(1, width=22, height=44), stable_id="stable-7")
    candidate = analyzer.analyze(
        bus_detection(2, width=24, height=48),
        stable_id="stable-7",
    )
    after_gap = analyzer.analyze(
        bus_detection(4, width=26, height=52),
        stable_id="stable-7",
    )

    assert candidate.attributes["motion_candidate_frames"] == 1
    assert after_gap.state == "UNKNOWN"
    assert after_gap.attributes["motion_candidate_frames"] == 0
    assert after_gap.attributes["observations_are_consecutive"] is False


def test_motion_histories_are_isolated_and_resettable() -> None:
    analyzer = BusAnalyzer()
    a_results: list[AnalysisResult] = []
    b_results: list[AnalysisResult] = []
    for frame_index, size in enumerate((20.0, 22.0, 24.0, 26.0)):
        a_results.append(
            analyzer.analyze(
                bus_detection(frame_index, width=size, height=size * 2),
                stable_id="bus-a",
            )
        )
        b_results.append(
            analyzer.analyze(
                bus_detection(frame_index, width=30, height=60),
                stable_id="bus-b",
            )
        )

    assert a_results[-1].state == "APPROACHING"
    assert b_results[-1].state == "STOPPED"

    analyzer.reset("bus-a")
    reset_result = analyzer.analyze(
        bus_detection(4, width=28, height=56),
        stable_id="bus-a",
    )
    assert reset_result.state == "UNKNOWN"


def test_route_number_requires_three_identical_high_confidence_frames() -> None:
    engine = FakeOcrEngine(
        [
            ocr_result("310?"),
            ocr_result("3102", 0.91),
            ocr_result("3102", 0.92),
            ocr_result("3102", 0.93),
        ]
    )
    analyzer = BusAnalyzer(ocr_engine=engine)
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    results = [
        analyzer.analyze(
            bus_detection(frame_index),
            stable_id="stable-7",
            crop=crop,
        )
        for frame_index in range(4)
    ]

    assert [result.attributes["route_number"] for result in results] == [
        None,
        None,
        None,
        "3102",
    ]
    assert results[-1].attributes["ocr_confirmed_frames"] == 3
    assert results[-1].attributes["route_confidence"] == pytest.approx(0.92)
    assert results[-1].attributes["route_roi_xyxy"] == (8, 0, 92, 55)
    assert engine.calls == [(55, 84, 3)] * 4


def test_conflicting_or_low_confidence_ocr_never_confirms_a_route() -> None:
    engine = FakeOcrEngine(
        [
            ocr_result("3102", 0.95),
            ocr_result("3103", 0.95),
            ocr_result("3102", 0.70),
            OcrResult(
                lines=(
                    OcrLine("3102", 0.95),
                    OcrLine("3103", 0.95),
                ),
                engine_name="fake-ocr",
            ),
        ]
    )
    analyzer = BusAnalyzer(ocr_engine=engine)
    crop = np.zeros((80, 120, 3), dtype=np.uint8)

    results = [
        analyzer.analyze(
            bus_detection(frame_index),
            stable_id="stable-7",
            crop=crop,
        )
        for frame_index in range(4)
    ]

    assert all(result.attributes["route_number"] is None for result in results)
    assert results[-1].attributes["ocr_candidate_frames"] == 0


def test_missing_crop_and_low_detection_confidence_do_not_call_ocr() -> None:
    engine = FakeOcrEngine([ocr_result("3102")])
    analyzer = BusAnalyzer(ocr_engine=engine)

    missing_crop = analyzer.analyze(bus_detection(0), stable_id="stable-7")
    low_detection = analyzer.analyze(
        bus_detection(1, confidence=0.2),
        stable_id="stable-7",
        crop=np.zeros((100, 100, 3), dtype=np.uint8),
    )

    assert engine.calls == []
    assert missing_crop.attributes["route_number"] is None
    assert "bus_crop_unavailable" in missing_crop.attributes["uncertainty_reasons"]
    assert low_detection.state == "UNKNOWN"
    assert "low_detection_confidence" in low_detection.attributes["uncertainty_reasons"]


def test_ocr_frame_gap_breaks_candidate_but_preserves_confirmed_route() -> None:
    engine = FakeOcrEngine([ocr_result("M5107") for _ in range(5)])
    analyzer = BusAnalyzer(ocr_engine=engine)
    crop = np.zeros((60, 80, 3), dtype=np.uint8)

    confirmed = [
        analyzer.analyze(
            bus_detection(frame_index),
            stable_id="stable-7",
            crop=crop,
        )
        for frame_index in range(3)
    ][-1]
    after_gap = analyzer.analyze(
        bus_detection(4),
        stable_id="stable-7",
        crop=crop,
    )

    assert confirmed.attributes["route_number"] == "M5107"
    assert after_gap.attributes["route_number"] == "M5107"
    assert after_gap.attributes["ocr_confirmed_frames"] == 1
    assert after_gap.attributes["observations_are_consecutive"] is False


def test_route_change_is_emitted_only_after_reconfirmation() -> None:
    engine = FakeOcrEngine(
        [ocr_result("3102") for _ in range(3)] + [ocr_result("3103") for _ in range(3)]
    )
    analyzer = BusAnalyzer(ocr_engine=engine)
    crop = np.zeros((60, 80, 3), dtype=np.uint8)
    results = [
        analyzer.analyze(
            bus_detection(frame_index),
            stable_id="stable-7",
            crop=crop,
        )
        for frame_index in range(6)
    ]

    assert results[2].attributes["route_number"] == "3102"
    assert results[3].attributes["route_number"] is None
    assert results[4].attributes["route_number"] is None
    assert results[4].attributes["last_confirmed_route"] == "3102"
    assert results[5].attributes["route_number"] == "3103"
    assert results[5].attributes["route_changed"] is True


def test_stale_confirmed_route_is_not_attached_to_later_approach() -> None:
    engine = FakeOcrEngine([ocr_result("3102") for _ in range(3)])
    analyzer = BusAnalyzer(ocr_engine=engine)
    crop = np.zeros((60, 80, 3), dtype=np.uint8)
    manager = SceneEventManager(auto_presence=False)

    for frame_index, size in enumerate((20.0, 22.0, 24.0)):
        result = analyzer.analyze(
            bus_detection(frame_index, width=size, height=size * 2),
            stable_id="stable-7",
            crop=crop,
        )
        manager.update([result], frame_index / 30.0)

    approaching = analyzer.analyze(
        bus_detection(3, width=26.0, height=52.0),
        stable_id="stable-7",
        crop=None,
    )
    events = manager.update([approaching], 0.1)

    assert approaching.state == "APPROACHING"
    assert approaching.attributes["route_number"] is None
    assert approaching.attributes["last_confirmed_route"] == "3102"
    event = next(event for event in events if event.event_type == OBJECT_APPROACHING)
    assert event.attributes["route_number"] is None


def test_confirmed_approach_creates_one_domain_event() -> None:
    analyzer = BusAnalyzer()
    manager = SceneEventManager(auto_presence=False)
    emitted_types: list[str] = []
    for frame_index, size in enumerate((20.0, 22.0, 24.0, 26.0, 28.0)):
        result = analyzer.analyze(
            bus_detection(frame_index, width=size, height=size * 2),
            stable_id="stable-7",
        )
        events = manager.update([result], result.attributes["bbox_area"] or 0.0)
        emitted_types.extend(event.event_type for event in events)

    assert emitted_types.count(OBJECT_APPROACHING) == 1


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("motion_window_frames", 1),
        ("minimum_motion_confirmed_frames", 0),
        ("minimum_ocr_confirmed_frames", True),
        ("minimum_detection_confidence", -0.1),
        ("minimum_motion_confidence", -0.1),
        ("minimum_ocr_confidence", 1.1),
        ("minimum_area_change_ratio", 0.0),
        ("minimum_route_confidence_margin", 1.1),
        ("minimum_direction_consistency", 0.0),
        ("area_jitter_tolerance_ratio", 1.1),
        ("maximum_motion_frame_gap", -1),
        ("route_ocr_interval_frames", 0),
        ("route_ocr_requires_relevant_motion", "yes"),
    ],
)
def test_invalid_configuration_is_rejected(keyword: str, value: object) -> None:
    with pytest.raises(ValueError):
        BusAnalyzer(**{keyword: value})  # type: ignore[arg-type]


def test_long_window_confirms_direction_despite_bbox_jitter() -> None:
    analyzer = BusAnalyzer(
        motion_window_frames=9,
        minimum_area_change_ratio=0.1,
        minimum_direction_consistency=0.65,
        area_jitter_tolerance_ratio=0.02,
    )
    sizes = (100.0, 102.0, 101.0, 103.0, 102.0, 105.0, 104.0, 107.0, 106.0, 109.0)
    offsets = (0.0, 2.0, -1.0, 1.0, -2.0, 0.0, 2.0, -1.0, 1.0, 0.0)
    results = [
        analyzer.analyze(
            bus_detection(
                frame_index,
                center_x=200.0 + offset,
                center_y=200.0 - offset,
                width=size,
                height=size,
            ),
            stable_id="jittering-bus",
        )
        for frame_index, (size, offset) in enumerate(zip(sizes, offsets, strict=True))
    ]

    assert [result.state for result in results] == ["UNKNOWN"] * 9 + ["APPROACHING"]
    assert results[-1].attributes["increasing_consistency"] >= 0.65
    assert results[-1].attributes["motion_window_observations"] == 9

    no_trend_analyzer = BusAnalyzer(
        motion_window_frames=9,
        minimum_area_change_ratio=0.1,
    )
    no_trend_sizes = (100.0, 102.0, 99.0, 101.0, 100.0, 102.0, 99.0, 101.0, 100.0, 101.0)
    no_trend_results = [
        no_trend_analyzer.analyze(
            bus_detection(frame_index, width=size, height=size),
            stable_id="no-trend-bus",
        )
        for frame_index, size in enumerate(no_trend_sizes)
    ]

    assert all(result.state not in {"APPROACHING", "RECEDING"} for result in no_trend_results)


def test_short_gaps_and_one_low_confidence_observation_preserve_motion_history() -> None:
    analyzer = BusAnalyzer(
        motion_window_frames=9,
        minimum_area_change_ratio=0.1,
        minimum_detection_confidence=0.3,
        maximum_motion_frame_gap=1,
    )
    frames = (0, 1, 2, 3, 5, 6, 7, 8, 9, 10)
    sizes = (100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 107.0, 106.0, 109.0, 111.0)
    results = [
        analyzer.analyze(
            bus_detection(frame_index, width=size, height=size),
            stable_id="short-gap-bus",
        )
        for frame_index, size in zip(frames, sizes, strict=True)
    ]

    assert results[4].attributes["observations_are_consecutive"] is False
    assert results[-1].state == "APPROACHING"

    low_confidence_analyzer = BusAnalyzer(
        motion_window_frames=9,
        minimum_area_change_ratio=0.1,
        minimum_detection_confidence=0.3,
        maximum_motion_frame_gap=1,
    )
    low_confidence_results = []
    for frame_index, size in enumerate((100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120)):
        confidence = 0.2 if frame_index == 4 else 0.95
        low_confidence_results.append(
            low_confidence_analyzer.analyze(
                bus_detection(frame_index, width=size, height=size, confidence=confidence),
                stable_id="low-confidence-gap-bus",
            )
        )

    assert low_confidence_results[4].state == "UNKNOWN"
    assert low_confidence_results[4].attributes["unreliable_motion_frames"] == 1
    assert low_confidence_results[-1].state == "APPROACHING"


def test_tolerated_frame_gap_preserves_ocr_vote_but_long_gap_resets_motion() -> None:
    engine = FakeOcrEngine([ocr_result("3102") for _ in range(3)])
    analyzer = BusAnalyzer(ocr_engine=engine, maximum_motion_frame_gap=1)
    crop = np.zeros((60, 80, 3), dtype=np.uint8)
    route_results = [
        analyzer.analyze(
            bus_detection(frame_index),
            stable_id="short-ocr-gap",
            crop=crop,
        )
        for frame_index in (0, 1, 3)
    ]

    assert route_results[-1].attributes["route_number"] == "3102"
    assert route_results[-1].attributes["ocr_confirmed_frames"] == 3

    long_gap_analyzer = BusAnalyzer(
        motion_window_frames=9,
        minimum_area_change_ratio=0.1,
        maximum_motion_frame_gap=1,
    )
    long_gap_results = [
        long_gap_analyzer.analyze(
            bus_detection(frame_index, width=size, height=size),
            stable_id="long-gap-bus",
        )
        for frame_index, size in zip(
            (0, 1, 2, 3, 4, 25, 26, 27, 28, 29),
            (100, 102, 104, 106, 108, 110, 112, 114, 116, 118),
            strict=True,
        )
    ]

    assert all(result.state == "UNKNOWN" for result in long_gap_results)
    assert long_gap_results[5].attributes["observations_are_consecutive"] is False
    assert long_gap_results[-1].attributes["motion_window_observations"] == 5


def test_dominant_route_candidate_survives_lower_confidence_advertisement_digits() -> None:
    engine = FakeOcrEngine(
        [
            OcrResult(
                lines=(
                    OcrLine("532", 0.99),
                    OcrLine("5", 0.70),
                    OcrLine("41", 0.65),
                ),
                engine_name="fake-ocr",
            )
            for _ in range(3)
        ]
    )
    analyzer = BusAnalyzer(ocr_engine=engine)
    crop = np.zeros((100, 160, 3), dtype=np.uint8)
    results = [
        analyzer.analyze(
            bus_detection(frame_index),
            stable_id="route-532",
            crop=crop,
        )
        for frame_index in range(3)
    ]

    assert results[-1].attributes["route_number"] == "532"
    assert results[-1].attributes["route_confidence"] == pytest.approx(0.95)


def test_route_ocr_interval_counts_only_new_ocr_observations() -> None:
    engine = FakeOcrEngine([ocr_result("532") for _ in range(3)])
    analyzer = BusAnalyzer(
        ocr_engine=engine,
        route_ocr_interval_frames=2,
    )
    crop = np.zeros((80, 120, 3), dtype=np.uint8)
    results = [
        analyzer.analyze(
            bus_detection(frame_index),
            stable_id="interval-route",
            crop=crop,
        )
        for frame_index in range(6)
    ]

    assert [result.attributes["ocr_was_run"] for result in results] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert len(engine.calls) == 3
    assert results[3].attributes["ocr_candidate_frames"] == 2
    assert results[4].attributes["route_number"] == "532"
    assert results[5].attributes["route_number"] == "532"


def test_pipeline_style_route_ocr_waits_for_relevant_bus_motion() -> None:
    engine = FakeOcrEngine([ocr_result("532")])
    analyzer = BusAnalyzer(
        ocr_engine=engine,
        route_ocr_requires_relevant_motion=True,
    )
    crop = np.zeros((80, 120, 3), dtype=np.uint8)
    results = [
        analyzer.analyze(
            bus_detection(frame_index),
            stable_id="stopped-route",
            crop=crop,
        )
        for frame_index in range(4)
    ]

    assert [result.state for result in results] == [
        "UNKNOWN",
        "UNKNOWN",
        "UNKNOWN",
        "STOPPED",
    ]
    assert len(engine.calls) == 1
    assert all(
        "route_ocr_waiting_for_relevant_motion" in result.attributes["uncertainty_reasons"]
        for result in results[:3]
    )
    assert results[-1].attributes["ocr_was_run"] is True
