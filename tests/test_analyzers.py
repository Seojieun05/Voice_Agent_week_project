from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from vision_agent.analyzers import (
    BusAnalyzer,
    GenericVisionAnalyzer,
    KioskAnalyzer,
    TextObjectAnalyzer,
    TrafficLightAnalyzer,
)
from vision_agent.signals import SignalStateResult
from vision_agent.types import AnalysisEvent, AnalysisResult, Detection, SignalState


def detection(
    class_name: str = "traffic light",
    *,
    track_id: int | None = 1,
    frame_index: int = 0,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        frame_index=frame_index,
        timestamp_s=frame_index / 30.0,
        class_id=9,
        class_name=class_name,
        confidence=confidence,
        xyxy=(0.0, 0.0, 10.0, 20.0),
        track_id=track_id,
    )


def observation(state: SignalState, *, confidence: float = 0.9) -> SignalStateResult:
    return SignalStateResult(
        state=state,
        confidence=confidence if state is not SignalState.UNKNOWN else 0.0,
        red_ratio=0.2 if state is SignalState.RED else 0.0,
        green_ratio=0.2 if state is SignalState.GREEN else 0.0,
        yellow_ratio=0.2 if state is SignalState.YELLOW else 0.0,
    )


def analyze_states(
    analyzer: TrafficLightAnalyzer,
    states: Iterable[SignalState],
    *,
    stable_id: str = "stable-1",
    start_frame: int = 0,
) -> list[AnalysisResult]:
    return [
        analyzer.analyze(
            detection(frame_index=frame_index),
            stable_id=stable_id,
            precomputed_signal_result=observation(state),
        )
        for frame_index, state in enumerate(states, start=start_frame)
    ]


def test_analysis_result_and_event_are_json_ready() -> None:
    result = AnalysisResult(
        object_type="traffic_light",
        stable_id="stable-1",
        state=SignalState.GREEN,
        confidence=0.9,
        attributes={"changed": False},
        is_uncertain=False,
    )
    event = AnalysisEvent(
        event_type="OBJECT_STATE_CHANGED",
        object_type="traffic_light",
        stable_id="stable-1",
        timestamp_s=1.25,
        previous_state=SignalState.GREEN,
        current_state=SignalState.RED,
        confidence=0.8,
    )

    assert result.to_dict()["state"] == "GREEN"
    assert event.to_dict()["previous_state"] == "GREEN"
    assert event.to_dict()["current_state"] == "RED"


def test_green_is_confirmed_after_three_consecutive_frames() -> None:
    results = analyze_states(
        TrafficLightAnalyzer(minimum_confirmed_frames=3),
        [SignalState.GREEN] * 3,
    )

    assert [result.state for result in results] == ["UNKNOWN", "UNKNOWN", "GREEN"]
    assert results[-1].attributes["confirmed_frames"] == 3
    assert results[-1].attributes["changed"] is False
    assert results[-1].is_uncertain is False


def test_green_to_red_is_reported_once_after_three_red_frames() -> None:
    analyzer = TrafficLightAnalyzer(minimum_confirmed_frames=3)
    baseline = analyze_states(analyzer, [SignalState.GREEN] * 3)
    transition = analyze_states(analyzer, [SignalState.RED] * 5, start_frame=3)

    assert baseline[-1].state == "GREEN"
    assert [result.state for result in transition[:3]] == ["GREEN", "GREEN", "RED"]
    changed = [result for result in transition if result.attributes["changed"]]
    assert len(changed) == 1
    assert changed[0].attributes["previous_state"] == "GREEN"
    assert changed[0].attributes["observed_state"] == "RED"
    assert transition[0].attributes["confirmed_frames"] == 0
    assert transition[0].attributes["candidate_state"] == "RED"
    assert transition[0].attributes["candidate_frames"] == 1
    assert transition[-1].state == "RED"
    assert transition[-1].attributes["changed"] is False


def test_unknown_breaks_candidate_streak_without_changing_confirmed_state() -> None:
    analyzer = TrafficLightAnalyzer(minimum_confirmed_frames=3)
    analyze_states(analyzer, [SignalState.GREEN] * 3)

    before_unknown = analyze_states(analyzer, [SignalState.RED] * 2, start_frame=3)
    unknown = analyze_states(analyzer, [SignalState.UNKNOWN], start_frame=5)[0]
    after_unknown = analyze_states(analyzer, [SignalState.RED] * 2, start_frame=6)

    assert all(result.state == "GREEN" for result in before_unknown)
    assert unknown.state == "UNKNOWN"
    assert unknown.is_uncertain is True
    assert all(result.state == "GREEN" for result in after_unknown)
    assert not any(result.attributes["changed"] for result in after_unknown)

    confirmed = analyze_states(analyzer, [SignalState.RED], start_frame=8)[0]
    assert confirmed.state == "RED"
    assert confirmed.attributes["changed"] is True


def test_signal_histories_are_isolated_by_stable_id() -> None:
    analyzer = TrafficLightAnalyzer(minimum_confirmed_frames=3)

    green = analyze_states(analyzer, [SignalState.GREEN] * 3, stable_id="stable-green")
    red = analyze_states(analyzer, [SignalState.RED] * 3, stable_id="stable-red")

    assert green[-1].state == "GREEN"
    assert red[-1].state == "RED"
    assert red[-1].attributes["changed"] is False


def test_missing_video_frame_breaks_transition_candidate_streak() -> None:
    analyzer = TrafficLightAnalyzer(minimum_confirmed_frames=3)
    analyze_states(analyzer, [SignalState.GREEN] * 3)
    analyze_states(analyzer, [SignalState.RED] * 2, start_frame=3)

    after_gap = analyze_states(analyzer, [SignalState.RED], start_frame=6)[0]

    assert after_gap.state == "GREEN"
    assert after_gap.attributes["changed"] is False
    assert after_gap.attributes["candidate_frames"] == 1
    assert after_gap.attributes["observations_are_consecutive"] is False

    transition = analyze_states(analyzer, [SignalState.RED] * 2, start_frame=7)
    assert transition[-1].state == "RED"
    assert transition[-1].attributes["changed"] is True


def test_reset_forgets_one_signals_confirmation_history() -> None:
    analyzer = TrafficLightAnalyzer(minimum_confirmed_frames=3)
    analyze_states(analyzer, [SignalState.GREEN] * 3)

    analyzer.reset("stable-1")
    new_baseline = analyze_states(analyzer, [SignalState.RED] * 3, start_frame=3)

    assert new_baseline[-1].state == "RED"
    assert new_baseline[-1].attributes["changed"] is False


def test_yellow_is_treated_as_a_known_state() -> None:
    results = analyze_states(
        TrafficLightAnalyzer(minimum_confirmed_frames=3),
        [SignalState.YELLOW] * 3,
    )

    assert results[-1].state == "YELLOW"
    assert results[-1].attributes["yellow_ratio"] == pytest.approx(0.2)


def test_green_to_yellow_transition_is_confirmed_once() -> None:
    analyzer = TrafficLightAnalyzer(minimum_confirmed_frames=3)
    analyze_states(analyzer, [SignalState.GREEN] * 3)

    transition = analyze_states(analyzer, [SignalState.YELLOW] * 4, start_frame=3)

    changed = [result for result in transition if result.attributes["changed"]]
    assert len(changed) == 1
    assert changed[0].state == "YELLOW"
    assert changed[0].attributes["previous_state"] == "GREEN"


class _FailingClassifier:
    def classify(self, crop: np.ndarray) -> SignalStateResult:
        raise AssertionError("classifier must not be called")


def test_precomputed_observation_avoids_duplicate_classifier_call() -> None:
    analyzer = TrafficLightAnalyzer(
        classifier=_FailingClassifier(),
        minimum_confirmed_frames=1,
    )

    result = analyzer.analyze(
        detection(),
        stable_id="stable-1",
        crop=np.zeros((20, 10, 3), dtype=np.uint8),
        precomputed_signal_result=observation(SignalState.GREEN),
    )

    assert result.state == "GREEN"
    assert result.attributes["green_ratio"] == pytest.approx(0.2)


def test_traffic_analysis_preserves_detector_confidence_for_presence_policy() -> None:
    result = TrafficLightAnalyzer(minimum_confirmed_frames=1).analyze(
        detection(confidence=0.1),
        stable_id="stable-1",
        precomputed_signal_result=observation(SignalState.GREEN),
    )

    assert result.attributes["detection_confidence"] == pytest.approx(0.1)
    assert result.state == "UNKNOWN"
    assert result.attributes["reason"] == "low_detection_confidence"
    assert result.is_uncertain is True


def test_low_detector_signal_does_not_contribute_to_state_confirmation() -> None:
    analyzer = TrafficLightAnalyzer(
        minimum_confirmed_frames=2,
        minimum_detection_confidence=0.2,
    )

    low = analyzer.analyze(
        detection(frame_index=0, confidence=0.1),
        stable_id="stable-1",
        precomputed_signal_result=observation(SignalState.GREEN, confidence=0.99),
    )
    first_reliable = analyzer.analyze(
        detection(frame_index=1, confidence=0.3),
        stable_id="stable-1",
        precomputed_signal_result=observation(SignalState.GREEN, confidence=0.99),
    )
    confirmed = analyzer.analyze(
        detection(frame_index=2, confidence=0.3),
        stable_id="stable-1",
        precomputed_signal_result=observation(SignalState.GREEN, confidence=0.99),
    )

    assert low.attributes["candidate_frames"] == 0
    assert first_reliable.state == "UNKNOWN"
    assert first_reliable.attributes["candidate_frames"] == 1
    assert confirmed.state == "GREEN"


def test_disabled_signal_analysis_does_not_create_or_call_classifier() -> None:
    analyzer = TrafficLightAnalyzer(classifier=_FailingClassifier(), enabled=False)

    result = analyzer.analyze(
        detection(),
        stable_id="stable-1",
        crop=np.zeros((20, 10, 3), dtype=np.uint8),
        precomputed_signal_result=observation(SignalState.GREEN),
    )

    assert analyzer.classifier is None
    assert result.state == "UNKNOWN"
    assert result.attributes["reason"] == "signal_state_analysis_disabled"
    assert result.is_uncertain is True


@pytest.mark.parametrize(
    ("analyzer", "class_name", "expected_state", "reason"),
    [
        (BusAnalyzer(), "bus", "UNKNOWN", "insufficient_or_ambiguous_motion"),
        (KioskAnalyzer(), "kiosk", "UNKNOWN", "ocr_engine_unavailable"),
        (TextObjectAnalyzer(), "sign", None, "text_crop_unavailable"),
        (GenericVisionAnalyzer(), "vending machine", "UNKNOWN", "generic_vision_disabled"),
    ],
)
def test_non_signal_analyzers_are_uncertain_without_required_evidence(
    analyzer: object,
    class_name: str,
    expected_state: str | None,
    reason: str,
) -> None:
    result = analyzer.analyze(detection(class_name), stable_id="stable-7")  # type: ignore[attr-defined]

    assert result.stable_id == "stable-7"
    assert result.state == expected_state
    assert result.confidence == 0.0
    assert result.is_uncertain is True
    assert result.attributes["reason"] == reason


@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_confirmation_frame_count_is_rejected(value: int) -> None:
    with pytest.raises(ValueError):
        TrafficLightAnalyzer(minimum_confirmed_frames=value)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_signal_detection_confidence_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="minimum_detection_confidence"):
        TrafficLightAnalyzer(minimum_detection_confidence=value)
