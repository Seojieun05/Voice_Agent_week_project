from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from vision_agent.analyzers.kiosk import (
    CONFIRMATION,
    ORDER_TYPE_SELECTION,
    PAYMENT,
    UNKNOWN,
    KioskAnalyzer,
)
from vision_agent.types import Detection


@dataclass(frozen=True)
class _Line:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class _Result:
    lines: tuple[_Line, ...]
    is_available: bool = True
    error: str | None = None


class _QueueOcrEngine:
    def __init__(self, *results: _Result | Exception) -> None:
        self.results = list(results)
        self.calls = 0

    def recognize(self, _image: np.ndarray) -> _Result:
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _result(*lines: str | tuple[str, float]) -> _Result:
    normalized_lines = tuple(
        _Line(line, 0.9) if isinstance(line, str) else _Line(line[0], line[1]) for line in lines
    )
    return _Result(normalized_lines)


def _detection(frame_index: int, *, confidence: float = 0.9) -> Detection:
    return Detection(
        frame_index=frame_index,
        timestamp_s=frame_index / 30,
        class_id=100,
        class_name="kiosk",
        confidence=confidence,
        xyxy=(0.0, 0.0, 20.0, 30.0),
        track_id=1,
    )


def _crop() -> np.ndarray:
    return np.zeros((30, 20, 3), dtype=np.uint8)


def _analyze(
    analyzer: KioskAnalyzer,
    frame_index: int,
    *,
    stable_id: str = "stable-1",
    detection_confidence: float = 0.9,
):
    return analyzer.analyze(
        _detection(frame_index, confidence=detection_confidence),
        stable_id=stable_id,
        crop=_crop(),
    )


def test_normalized_order_options_require_three_consecutive_frames() -> None:
    screen = _result("  매장   식사  ", "포장", "메뉴를 선택하세요")
    engine = _QueueOcrEngine(screen, screen, screen)
    analyzer = KioskAnalyzer(engine, minimum_confirmed_frames=3)

    results = [_analyze(analyzer, frame) for frame in range(3)]

    assert [result.state for result in results] == [UNKNOWN, UNKNOWN, ORDER_TYPE_SELECTION]
    assert results[-1].attributes["visible_text"] == [
        "매장 식사",
        "포장",
        "메뉴를 선택하세요",
    ]
    assert results[-1].attributes["visible_options"] == ["매장 식사", "포장"]
    assert results[-1].attributes["confirmed_frames"] == 3
    assert results[-1].confidence == pytest.approx(0.9)
    assert results[-1].is_uncertain is False


@pytest.mark.parametrize(
    ("screen", "expected_stage"),
    [
        (_result("결제 방법", "카드", "현금"), PAYMENT),
        (_result("주문이 완료되었습니다", "감사합니다"), CONFIRMATION),
        (_result("EAT IN", "TAKE-OUT"), ORDER_TYPE_SELECTION),
    ],
)
def test_supported_stages_are_confirmed_conservatively(
    screen: _Result,
    expected_stage: str,
) -> None:
    engine = _QueueOcrEngine(screen, screen)
    analyzer = KioskAnalyzer(engine, minimum_confirmed_frames=2)

    first = _analyze(analyzer, 0)
    confirmed = _analyze(analyzer, 1)

    assert first.state == UNKNOWN
    assert first.is_uncertain is True
    assert confirmed.state == expected_stage
    assert confirmed.is_uncertain is False


def test_explicit_confirmation_wins_over_buttons_visible_beneath_dialog() -> None:
    confirmation_dialog = _result(
        "주문하시겠습니까?",
        "매장 식사",
        "포장",
        "확인",
        "취소",
    )
    analyzer = KioskAnalyzer(
        _QueueOcrEngine(confirmation_dialog),
        minimum_confirmed_frames=1,
    )

    result = _analyze(analyzer, 0)

    assert result.state == CONFIRMATION
    assert result.attributes["visible_options"] == [
        "매장 식사",
        "포장",
        "확인",
        "취소",
    ]


def test_general_bbox_text_is_exposed_as_button_candidate_without_guessing_stage() -> None:
    menu = _Result(
        (
            _Line("불고기 버거", 0.95, (10, 20, 120, 50)),
            _Line("새우 버거", 0.94, (10, 60, 120, 90)),
            _Line("메뉴를 선택하세요", 0.93, None),
        )
    )
    result = _analyze(KioskAnalyzer(_QueueOcrEngine(menu), minimum_confirmed_frames=1), 0)

    assert result.state == UNKNOWN
    assert result.attributes["visible_options"] == []
    assert result.attributes["button_candidates"] == [
        {"text": "불고기 버거", "confidence": 0.95, "bbox": [10, 20, 120, 50]},
        {"text": "새우 버거", "confidence": 0.94, "bbox": [10, 60, 120, 90]},
    ]


def test_low_confidence_and_ambiguous_text_remain_unknown() -> None:
    screen = _result(("매장 식사", 0.95), ("포장", 0.4), ("환영합니다", 0.99))
    engine = _QueueOcrEngine(screen, screen, screen)
    analyzer = KioskAnalyzer(engine, minimum_ocr_confidence=0.6)

    result = [_analyze(analyzer, frame) for frame in range(3)][-1]

    assert result.state == UNKNOWN
    assert result.attributes["visible_text"] == ["매장 식사", "환영합니다"]
    assert result.attributes["visible_options"] == ["매장 식사"]
    assert result.attributes["reason"] == "screen_stage_unknown"
    assert result.is_uncertain is True


def test_unknown_observation_breaks_stage_candidate_streak() -> None:
    order = _result("매장 식사", "포장")
    unknown = _result("환영합니다")
    engine = _QueueOcrEngine(order, order, unknown, order, order, order)
    analyzer = KioskAnalyzer(engine, minimum_confirmed_frames=3)

    states = [_analyze(analyzer, frame).state for frame in range(6)]

    assert states == [UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, ORDER_TYPE_SELECTION]


def test_missing_frame_breaks_stage_and_fingerprint_candidates() -> None:
    order = _result("매장 식사", "포장")
    engine = _QueueOcrEngine(order, order, order, order, order)
    analyzer = KioskAnalyzer(
        engine,
        minimum_confirmed_frames=3,
        minimum_screen_change_frames=3,
    )

    outputs = [_analyze(analyzer, frame) for frame in (0, 1, 3, 4, 5)]

    assert [result.state for result in outputs] == [
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
        ORDER_TYPE_SELECTION,
    ]
    assert outputs[-1].attributes["screen_changed"] is True
    assert outputs[-1].attributes["screen_initial_confirmation"] is True


def test_ocr_interval_skips_calls_without_adding_confirmation_votes() -> None:
    order = _result("매장 식사", "포장")
    engine = _QueueOcrEngine(order, order, order)
    analyzer = KioskAnalyzer(
        engine,
        minimum_confirmed_frames=3,
        minimum_screen_change_frames=3,
        ocr_interval_frames=2,
    )

    outputs = [_analyze(analyzer, frame) for frame in range(5)]

    assert engine.calls == 3
    assert [result.attributes["ocr_was_run"] for result in outputs] == [
        True,
        False,
        True,
        False,
        True,
    ]
    assert [result.attributes["screen_candidate_frames"] for result in outputs] == [
        1,
        1,
        2,
        2,
        3,
    ]
    assert [result.state for result in outputs[:-1]] == [UNKNOWN] * 4
    assert outputs[-1].state == ORDER_TYPE_SELECTION


def test_processed_frame_gap_resets_throttled_kiosk_votes_and_runs_ocr_immediately() -> None:
    order = _result("매장 식사", "포장")
    engine = _QueueOcrEngine(order, order, order)
    analyzer = KioskAnalyzer(
        engine,
        minimum_confirmed_frames=2,
        minimum_screen_change_frames=2,
        ocr_interval_frames=3,
    )

    outputs = [_analyze(analyzer, frame) for frame in (0, 1, 3, 4, 5, 6)]

    assert engine.calls == 3
    assert outputs[2].attributes["ocr_was_run"] is True
    assert outputs[2].attributes["screen_candidate_frames"] == 1
    assert outputs[2].state == UNKNOWN
    assert outputs[-1].state == ORDER_TYPE_SELECTION


def test_confirmed_screen_change_is_reported_once() -> None:
    first_screen = _result("매장 식사", "포장", "메뉴 선택")
    next_screen = _result("결제 방법", "카드", "현금")
    engine = _QueueOcrEngine(
        first_screen,
        first_screen,
        next_screen,
        next_screen,
        next_screen,
    )
    analyzer = KioskAnalyzer(
        engine,
        minimum_confirmed_frames=2,
        minimum_screen_change_frames=2,
    )

    outputs = [_analyze(analyzer, frame) for frame in range(5)]

    assert [result.attributes["screen_changed"] for result in outputs] == [
        False,
        True,
        False,
        True,
        False,
    ]
    assert outputs[1].attributes["screen_initial_confirmation"] is True
    assert outputs[3].attributes["screen_initial_confirmation"] is False
    assert outputs[3].state == PAYMENT
    assert outputs[3].attributes["previous_state"] == ORDER_TYPE_SELECTION
    assert outputs[3].attributes["stage_changed"] is True
    assert outputs[4].attributes["stage_changed"] is False


def test_transient_or_reordered_ocr_does_not_report_screen_change() -> None:
    baseline = _result("매장 식사", "포장")
    reordered = _result(" 포장 ", "매장  식사")
    transient = _result("결제 방법", "카드")
    engine = _QueueOcrEngine(baseline, reordered, transient, baseline)
    analyzer = KioskAnalyzer(
        engine,
        minimum_confirmed_frames=1,
        minimum_screen_change_frames=2,
    )

    outputs = [_analyze(analyzer, frame) for frame in range(4)]

    assert [result.attributes["screen_changed"] for result in outputs] == [
        False,
        True,
        False,
        False,
    ]
    assert outputs[1].attributes["screen_initial_confirmation"] is True
    assert (
        outputs[1].attributes["screen_fingerprint"]
        == outputs[0].attributes["observed_screen_fingerprint"]
    )


def test_histories_are_isolated_by_stable_id() -> None:
    order = _result("매장 식사", "포장")
    payment = _result("결제 방법", "카드")
    engine = _QueueOcrEngine(order, payment, order, payment)
    analyzer = KioskAnalyzer(engine, minimum_confirmed_frames=2)

    first_order = _analyze(analyzer, 0, stable_id="stable-order")
    first_payment = _analyze(analyzer, 0, stable_id="stable-payment")
    confirmed_order = _analyze(analyzer, 1, stable_id="stable-order")
    confirmed_payment = _analyze(analyzer, 1, stable_id="stable-payment")

    assert first_order.state == UNKNOWN
    assert first_payment.state == UNKNOWN
    assert confirmed_order.state == ORDER_TYPE_SELECTION
    assert confirmed_payment.state == PAYMENT


def test_reset_forgets_only_requested_stable_id() -> None:
    order = _result("매장 식사", "포장")
    engine = _QueueOcrEngine(order, order, order, order, order)
    analyzer = KioskAnalyzer(engine, minimum_confirmed_frames=2)

    _analyze(analyzer, 0, stable_id="stable-1")
    assert _analyze(analyzer, 1, stable_id="stable-1").state == ORDER_TYPE_SELECTION
    _analyze(analyzer, 0, stable_id="stable-2")
    analyzer.reset("stable-1")

    assert _analyze(analyzer, 2, stable_id="stable-1").state == UNKNOWN
    assert _analyze(analyzer, 1, stable_id="stable-2").state == ORDER_TYPE_SELECTION


def test_missing_engine_crop_and_failed_ocr_are_explicitly_uncertain() -> None:
    without_engine = KioskAnalyzer().analyze(
        _detection(0),
        stable_id="stable-1",
        crop=_crop(),
    )
    no_crop_engine = _QueueOcrEngine(_result("매장 식사", "포장"))
    without_crop = KioskAnalyzer(no_crop_engine).analyze(
        _detection(0),
        stable_id="stable-1",
        crop=None,
    )
    failing_engine = _QueueOcrEngine(RuntimeError("OCR failed"))
    failed = _analyze(KioskAnalyzer(failing_engine), 0)

    assert without_engine.state == UNKNOWN
    assert without_engine.attributes["reason"] == "ocr_engine_unavailable"
    assert without_crop.attributes["reason"] == "screen_crop_unavailable"
    assert no_crop_engine.calls == 0
    assert failed.attributes["reason"] == "ocr_failed:RuntimeError"
    assert all(result.is_uncertain for result in (without_engine, without_crop, failed))


def test_low_detection_confidence_skips_ocr_and_breaks_confirmation() -> None:
    screen = _result("매장 식사", "포장")
    engine = _QueueOcrEngine(screen)
    analyzer = KioskAnalyzer(engine, minimum_confirmed_frames=1)

    result = _analyze(analyzer, 0, detection_confidence=0.1)

    assert engine.calls == 0
    assert result.state == UNKNOWN
    assert result.confidence == 0.0
    assert result.attributes["reason"] == "low_detection_confidence"
    assert result.attributes["screen_changed"] is False
    assert result.is_uncertain is True


def test_kiosk_confidence_combines_detection_and_ocr_evidence() -> None:
    screen = _result("매장 식사", "포장")
    analyzer = KioskAnalyzer(
        _QueueOcrEngine(screen),
        minimum_confirmed_frames=1,
        minimum_screen_change_frames=1,
    )

    result = _analyze(analyzer, 0, detection_confidence=0.55)

    assert result.state == ORDER_TYPE_SELECTION
    assert result.confidence == pytest.approx(0.55)
    assert result.attributes["screen_confidence"] == pytest.approx(0.55)


def test_unavailable_ocr_result_is_not_treated_as_text() -> None:
    engine = _QueueOcrEngine(_Result((), is_available=False, error="language data unavailable"))

    result = _analyze(KioskAnalyzer(engine), 0)

    assert result.state == UNKNOWN
    assert result.attributes["visible_text"] == []
    assert result.attributes["reason"] == "language data unavailable"

    inference_error = _analyze(
        KioskAnalyzer(_QueueOcrEngine(_Result((), error="inference failed"))),
        0,
    )
    assert inference_error.attributes["reason"] == "inference failed"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"minimum_ocr_confidence": -0.1}, "minimum_ocr_confidence"),
        ({"minimum_detection_confidence": -0.1}, "minimum_detection_confidence"),
        ({"minimum_ocr_confidence": float("nan")}, "minimum_ocr_confidence"),
        ({"minimum_confirmed_frames": 0}, "minimum_confirmed_frames"),
        ({"minimum_confirmed_frames": True}, "minimum_confirmed_frames"),
        ({"minimum_screen_change_frames": 0}, "minimum_screen_change_frames"),
        ({"ocr_interval_frames": 0}, "ocr_interval_frames"),
        ({"ocr_interval_frames": True}, "ocr_interval_frames"),
    ],
)
def test_invalid_configuration_is_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        KioskAnalyzer(**kwargs)  # type: ignore[arg-type]
