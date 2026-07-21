from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from vision_agent.analyzers.text_object import TextObjectAnalyzer
from vision_agent.ocr import OcrLine, OcrResult
from vision_agent.types import AnalysisResult, Detection


def detection(
    frame_index: int,
    *,
    class_name: str = "sign",
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        frame_index=frame_index,
        timestamp_s=frame_index / 30,
        class_id=99,
        class_name=class_name,
        confidence=confidence,
        xyxy=(0.0, 0.0, 160.0, 80.0),
        track_id=1,
    )


def result(text: str, confidence: float = 0.9) -> OcrResult:
    return OcrResult(
        lines=(OcrLine(text, confidence, (0, 0, 100, 20)),),
        engine_name="fake",
    )


class SequenceOcrEngine:
    def __init__(self, results: Sequence[OcrResult | Exception]) -> None:
        self.results = list(results)
        self.calls = 0

    def recognize(self, image: np.ndarray) -> OcrResult:
        item = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def analyze_frames(
    analyzer: TextObjectAnalyzer,
    frame_indices: Sequence[int],
    *,
    stable_id: str = "stable-1",
) -> list[AnalysisResult]:
    crop = np.zeros((80, 160, 3), dtype=np.uint8)
    return [
        analyzer.analyze(detection(frame_index), stable_id=stable_id, crop=crop)
        for frame_index in frame_indices
    ]


def test_text_is_confirmed_only_after_three_consecutive_matching_frames() -> None:
    analyzer = TextObjectAnalyzer(
        SequenceOcrEngine([result("출구")] * 3),
        minimum_confirmed_frames=3,
    )

    results = analyze_frames(analyzer, [0, 1, 2])

    assert [item.is_uncertain for item in results] == [True, True, False]
    assert results[0].attributes["text"] is None
    assert results[1].attributes["ocr_candidate_frames"] == 2
    assert results[2].attributes["text"] == "출구"
    assert results[2].attributes["ocr_confirmed_frames"] == 3
    assert results[2].confidence == pytest.approx(0.9)


def test_multiple_ocr_lines_are_combined_in_reading_order() -> None:
    ocr_result = OcrResult(
        lines=(
            OcrLine("1층", 0.9, (0, 0, 40, 20)),
            OcrLine("안내", 0.8, (45, 0, 85, 20)),
        ),
        engine_name="fake",
    )
    analyzer = TextObjectAnalyzer(
        SequenceOcrEngine([ocr_result]),
        minimum_confirmed_frames=1,
    )

    analysis = analyze_frames(analyzer, [0])[0]

    assert analysis.attributes["text"] == "1층 안내"
    assert analysis.confidence == pytest.approx((0.9 * 2 + 0.8 * 2) / 4)


def test_low_confidence_text_is_not_a_confirmation_candidate() -> None:
    analyzer = TextObjectAnalyzer(
        SequenceOcrEngine([result("위험", 0.59)]),
        minimum_confirmed_frames=1,
        minimum_ocr_confidence=0.6,
    )

    analysis = analyze_frames(analyzer, [0])[0]

    assert analysis.is_uncertain is True
    assert analysis.attributes["text"] is None
    assert analysis.attributes["observed_text"] == "위험"
    assert analysis.attributes["ocr_candidate_frames"] == 0
    assert analysis.attributes["reason"] == "ocr_confidence_below_threshold"


@pytest.mark.parametrize("bbox", [(0, 0, 1, 1), None])
def test_small_or_missing_text_region_is_never_confirmed(
    bbox: tuple[int, int, int, int] | None,
) -> None:
    tiny = OcrResult(
        lines=(OcrLine("출구", 0.99, bbox),),
        engine_name="fake",
    )
    analyzer = TextObjectAnalyzer(
        SequenceOcrEngine([tiny] * 3),
        minimum_confirmed_frames=3,
    )

    analyses = analyze_frames(analyzer, [0, 1, 2])

    assert all(item.is_uncertain for item in analyses)
    assert analyses[-1].attributes["text"] is None
    assert analyses[-1].attributes["reason"] == (
        "ocr_text_region_too_small_or_unavailable"
    )


def test_different_digits_are_not_merged_as_similar_text() -> None:
    analyzer = TextObjectAnalyzer(
        SequenceOcrEngine(
            [
                result("서울역 1번 출구입니다"),
                result("서울역 2번 출구입니다"),
                result("서울역 2번 출구입니다"),
                result("서울역 2번 출구입니다"),
            ]
        ),
        minimum_confirmed_frames=3,
        minimum_text_similarity=0.8,
    )

    analyses = analyze_frames(analyzer, [0, 1, 2, 3])

    assert [item.attributes["ocr_candidate_frames"] for item in analyses] == [1, 1, 2, 3]
    assert analyses[2].is_uncertain is True
    assert analyses[3].attributes["text"] == "서울역 2번 출구입니다"


def test_text_change_requires_a_fresh_three_frame_streak() -> None:
    engine = SequenceOcrEngine(
        [result("대기")] * 3 + [result("입장")] * 3,
    )
    analyzer = TextObjectAnalyzer(engine, minimum_confirmed_frames=3)

    initial = analyze_frames(analyzer, [0, 1, 2])
    changed = analyze_frames(analyzer, [3, 4, 5])

    assert initial[-1].attributes["text"] == "대기"
    assert [item.is_uncertain for item in changed] == [True, True, False]
    assert changed[0].attributes["text"] == "대기"
    assert changed[-1].attributes["text"] == "입장"
    assert changed[-1].attributes["previous_text"] == "대기"
    assert changed[-1].attributes["text_changed"] is True


def test_inconsistent_ocr_resets_the_candidate_streak() -> None:
    analyzer = TextObjectAnalyzer(
        SequenceOcrEngine(
            [result("출구"), result("입구"), result("출구"), result("출구"), result("출구")]
        ),
        minimum_confirmed_frames=3,
    )

    analyses = analyze_frames(analyzer, [0, 1, 2, 3, 4])

    assert [item.attributes["ocr_candidate_frames"] for item in analyses] == [1, 1, 1, 2, 3]
    assert analyses[-1].attributes["text"] == "출구"
    assert analyses[-1].is_uncertain is False


def test_missing_video_frame_breaks_candidate_streak() -> None:
    analyzer = TextObjectAnalyzer(
        SequenceOcrEngine([result("출구")] * 4),
        minimum_confirmed_frames=3,
    )

    analyses = analyze_frames(analyzer, [0, 1, 3, 4])

    assert [item.attributes["ocr_candidate_frames"] for item in analyses] == [1, 2, 1, 2]
    assert all(item.is_uncertain for item in analyses)
    assert analyses[2].attributes["observations_are_consecutive"] is False


def test_missing_or_small_crop_does_not_call_ocr_engine() -> None:
    engine = SequenceOcrEngine([result("unused")])
    analyzer = TextObjectAnalyzer(engine)

    missing = analyzer.analyze(detection(0), stable_id="stable-1", crop=None)
    small = analyzer.analyze(
        detection(1),
        stable_id="stable-1",
        crop=np.zeros((10, 20, 3), dtype=np.uint8),
    )

    assert engine.calls == 0
    assert missing.attributes["reason"] == "text_crop_unavailable"
    assert small.attributes["reason"] == "text_crop_too_small"


def test_low_detection_confidence_does_not_call_ocr() -> None:
    engine = SequenceOcrEngine([result("출구")])
    analyzer = TextObjectAnalyzer(engine, minimum_confirmed_frames=1)

    analysis = analyzer.analyze(
        detection(0, confidence=0.1),
        stable_id="stable-1",
        crop=np.zeros((80, 160, 3), dtype=np.uint8),
    )

    assert engine.calls == 0
    assert analysis.attributes["reason"] == "low_detection_confidence"
    assert analysis.is_uncertain is True


def test_text_confidence_combines_detection_and_ocr_evidence() -> None:
    analyzer = TextObjectAnalyzer(
        SequenceOcrEngine([result("출구", 0.95)]),
        minimum_confirmed_frames=1,
    )

    analysis = analyzer.analyze(
        detection(0, confidence=0.55),
        stable_id="stable-1",
        crop=np.zeros((80, 160, 3), dtype=np.uint8),
    )

    assert analysis.attributes["ocr_confidence"] == pytest.approx(0.95)
    assert analysis.confidence == pytest.approx(0.55)


def test_empty_ocr_result_is_uncertain() -> None:
    analyzer = TextObjectAnalyzer(
        SequenceOcrEngine([OcrResult(engine_name="fake")]),
        minimum_confirmed_frames=1,
    )

    analysis = analyze_frames(analyzer, [0])[0]

    assert analysis.is_uncertain is True
    assert analysis.attributes["reason"] == "ocr_text_not_detected"


def test_unavailable_or_failing_engine_degrades_safely() -> None:
    unavailable = TextObjectAnalyzer(
        SequenceOcrEngine(
            [OcrResult(engine_name="fake", is_available=False, error="not installed")]
        ),
        minimum_confirmed_frames=1,
    )
    failing = TextObjectAnalyzer(
        SequenceOcrEngine([RuntimeError("broken")]),
        minimum_confirmed_frames=1,
    )

    unavailable_result = analyze_frames(unavailable, [0])[0]
    failing_result = analyze_frames(failing, [0])[0]

    assert unavailable_result.attributes["reason"] == "ocr_engine_unavailable"
    assert failing_result.attributes["reason"] == "ocr_engine_error"
    assert "broken" in str(failing_result.attributes["ocr_error"])


def test_histories_are_isolated_and_resettable_by_stable_id() -> None:
    analyzer = TextObjectAnalyzer(
        SequenceOcrEngine([result("출구")] * 5),
        minimum_confirmed_frames=2,
    )
    crop = np.zeros((80, 160, 3), dtype=np.uint8)

    first_a = analyzer.analyze(detection(0), stable_id="a", crop=crop)
    first_b = analyzer.analyze(detection(0), stable_id="b", crop=crop)
    second_a = analyzer.analyze(detection(1), stable_id="a", crop=crop)
    analyzer.reset("a")
    after_reset = analyzer.analyze(detection(2), stable_id="a", crop=crop)

    assert first_a.is_uncertain is True
    assert first_b.is_uncertain is True
    assert second_a.is_uncertain is False
    assert after_reset.is_uncertain is True
    assert after_reset.attributes["ocr_candidate_frames"] == 1


def test_unicode_case_and_whitespace_normalization_stabilizes_text() -> None:
    analyzer = TextObjectAnalyzer(
        SequenceOcrEngine([result("ＥＸＩＴ"), result(" exit ")]),
        minimum_confirmed_frames=2,
        minimum_text_similarity=1.0,
    )

    analyses = analyze_frames(analyzer, [0, 1])

    assert analyses[-1].is_uncertain is False
    assert analyses[-1].attributes["text"] in {"ＥＸＩＴ", "exit"}


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("minimum_confirmed_frames", 0),
        ("minimum_confirmed_frames", True),
        ("minimum_ocr_confidence", -0.1),
        ("minimum_detection_confidence", -0.1),
        ("minimum_ocr_confidence", float("nan")),
        ("minimum_text_similarity", 1.1),
        ("minimum_crop_width", 0),
        ("minimum_crop_height", True),
        ("minimum_crop_pixels", -1),
        ("minimum_text_width", 0),
        ("minimum_text_height", True),
        ("minimum_text_pixels", -1),
    ],
)
def test_invalid_configuration_is_rejected(keyword: str, value: object) -> None:
    with pytest.raises(ValueError):
        TextObjectAnalyzer(**{keyword: value})  # type: ignore[arg-type]
