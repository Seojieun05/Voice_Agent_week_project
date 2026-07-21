from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from vision_agent.analyzers.generic import GenericVisionAnalyzer
from vision_agent.types import Detection
from vision_agent.vlm import VisionLanguageResult


class _FakeModel:
    def __init__(self, *results: VisionLanguageResult) -> None:
        self.results = deque(results)
        self.calls = 0

    def describe(self, image: np.ndarray, prompt: str) -> VisionLanguageResult:
        self.calls += 1
        return self.results.popleft()


def detection(frame_index: int, *, confidence: float = 0.8) -> Detection:
    return Detection(
        frame_index=frame_index,
        timestamp_s=frame_index / 30,
        class_id=24,
        class_name="vending machine",
        confidence=confidence,
        xyxy=(0.0, 0.0, 20.0, 20.0),
        track_id=1,
    )


def test_generic_description_requires_repeated_similar_results() -> None:
    model = _FakeModel(
        VisionLanguageResult("빨간 자판기가 보입니다.", 0.8),
        VisionLanguageResult("빨간색 자판기가 보입니다.", 0.8),
    )
    analyzer = GenericVisionAnalyzer(
        model,
        inference_interval_frames=1,
        similarity_threshold=0.7,
    )
    crop = np.zeros((20, 20, 3), dtype=np.uint8)

    first = analyzer.analyze(detection(0), stable_id="stable-1", crop=crop)
    confirmed = analyzer.analyze(detection(1), stable_id="stable-1", crop=crop)

    assert first.state == "UNKNOWN"
    assert first.is_uncertain is True
    assert confirmed.state == "DESCRIBED"
    assert confirmed.attributes["description"] == "빨간색 자판기가 보입니다."
    assert confirmed.is_uncertain is False


def test_generic_vlm_is_throttled_between_inference_frames() -> None:
    model = _FakeModel(VisionLanguageResult("기계가 보입니다.", 0.8))
    analyzer = GenericVisionAnalyzer(model, inference_interval_frames=10)
    crop = np.zeros((20, 20, 3), dtype=np.uint8)

    analyzer.analyze(detection(0), stable_id="stable-1", crop=crop)
    skipped = analyzer.analyze(detection(1), stable_id="stable-1", crop=crop)

    assert model.calls == 1
    assert skipped.attributes["inference_performed"] is False


def test_generic_vlm_rejects_low_confidence_and_resets_by_id() -> None:
    model = _FakeModel(
        VisionLanguageResult("추측 설명", 0.2),
        VisionLanguageResult("새 설명", 0.9),
    )
    analyzer = GenericVisionAnalyzer(model, inference_interval_frames=1)
    crop = np.zeros((20, 20, 3), dtype=np.uint8)

    rejected = analyzer.analyze(detection(0), stable_id="stable-1", crop=crop)
    analyzer.reset("stable-1")
    after_reset = analyzer.analyze(detection(1), stable_id="stable-1", crop=crop)

    assert rejected.is_uncertain is True
    assert rejected.attributes["description"] is None
    assert after_reset.attributes["candidate_results"] == 1


def test_disabled_generic_vlm_is_explicitly_uncertain() -> None:
    result = GenericVisionAnalyzer().analyze(detection(0), stable_id="stable-1")

    assert result.state == "UNKNOWN"
    assert result.is_uncertain is True
    assert result.attributes["reason"] == "generic_vision_disabled"


def test_frame_gap_breaks_generic_confirmation_candidate() -> None:
    model = _FakeModel(
        VisionLanguageResult("기계가 보입니다.", 0.8),
        VisionLanguageResult("기계가 보입니다.", 0.8),
    )
    analyzer = GenericVisionAnalyzer(model, inference_interval_frames=1)
    crop = np.zeros((20, 20, 3), dtype=np.uint8)

    first = analyzer.analyze(detection(0), stable_id="stable-1", crop=crop)
    after_gap = analyzer.analyze(detection(100), stable_id="stable-1", crop=crop)

    assert first.attributes["candidate_results"] == 1
    assert after_gap.attributes["candidate_results"] == 1
    assert after_gap.state == "UNKNOWN"


def test_default_similarity_does_not_merge_opposite_descriptions() -> None:
    model = _FakeModel(
        VisionLanguageResult("버튼이 켜져 있습니다.", 0.8),
        VisionLanguageResult("버튼이 꺼져 있습니다.", 0.8),
    )
    analyzer = GenericVisionAnalyzer(model, inference_interval_frames=1)
    crop = np.zeros((20, 20, 3), dtype=np.uint8)

    analyzer.analyze(detection(0), stable_id="stable-1", crop=crop)
    second = analyzer.analyze(detection(1), stable_id="stable-1", crop=crop)

    assert second.attributes["candidate_results"] == 1
    assert second.state == "UNKNOWN"


def test_generic_backend_exception_degrades_to_uncertain_result() -> None:
    class BrokenModel:
        def describe(self, image: np.ndarray, prompt: str) -> VisionLanguageResult:
            raise RuntimeError("broken")

    result = GenericVisionAnalyzer(BrokenModel(), inference_interval_frames=1).analyze(
        detection(0),
        stable_id="stable-1",
        crop=np.zeros((20, 20, 3), dtype=np.uint8),
    )

    assert result.is_uncertain is True
    assert result.attributes["reason"] == "vlm_backend_error:RuntimeError"


def test_generic_model_is_not_called_outside_explicit_allowlist() -> None:
    model = _FakeModel(VisionLanguageResult("사람이 보입니다.", 0.9))
    analyzer = GenericVisionAnalyzer(
        model,
        inference_interval_frames=1,
        allowed_object_types={"unknown panel"},
    )

    result = analyzer.analyze(
        detection(0),
        stable_id="stable-1",
        crop=np.zeros((20, 20, 3), dtype=np.uint8),
    )

    assert model.calls == 0
    assert result.attributes["reason"] == "generic_vision_class_not_allowed"


def test_generic_waits_for_stable_object_before_first_inference() -> None:
    model = _FakeModel(VisionLanguageResult("기계가 보입니다.", 0.9))
    analyzer = GenericVisionAnalyzer(
        model,
        inference_interval_frames=1,
        minimum_seen_frames_before_inference=3,
    )
    crop = np.zeros((20, 20, 3), dtype=np.uint8)

    first = analyzer.analyze(detection(0), stable_id="stable-1", crop=crop)
    second = analyzer.analyze(detection(1), stable_id="stable-1", crop=crop)
    third = analyzer.analyze(detection(2), stable_id="stable-1", crop=crop)

    assert model.calls == 1
    assert first.attributes["reason"] == "generic_object_not_stable"
    assert second.attributes["reason"] == "generic_object_not_stable"
    assert third.attributes["inference_performed"] is True


def test_low_detection_confidence_never_calls_generic_model() -> None:
    model = _FakeModel(VisionLanguageResult("기계가 보입니다.", 0.9))
    analyzer = GenericVisionAnalyzer(model, inference_interval_frames=1)

    result = analyzer.analyze(
        detection(0, confidence=0.1),
        stable_id="stable-1",
        crop=np.zeros((20, 20, 3), dtype=np.uint8),
    )

    assert model.calls == 0
    assert result.attributes["reason"] == "low_detection_confidence"
    assert result.is_uncertain is True


def test_generic_confidence_combines_detection_and_vlm_evidence() -> None:
    model = _FakeModel(
        VisionLanguageResult("기계가 보입니다.", 0.9),
        VisionLanguageResult("기계가 보입니다.", 0.9),
    )
    analyzer = GenericVisionAnalyzer(model, inference_interval_frames=1)
    crop = np.zeros((20, 20, 3), dtype=np.uint8)

    analyzer.analyze(detection(0, confidence=0.55), stable_id="stable-1", crop=crop)
    confirmed = analyzer.analyze(
        detection(1, confidence=0.55),
        stable_id="stable-1",
        crop=crop,
    )

    assert confirmed.attributes["vlm_confidence"] == pytest.approx(0.9)
    assert confirmed.confidence == pytest.approx(0.55)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_detection_confidence_threshold_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="minimum_detection_confidence"):
        GenericVisionAnalyzer(minimum_detection_confidence=value)
