from __future__ import annotations

from collections import deque

import numpy as np

from vision_agent.analyzers import (
    GenericVisionAnalyzer,
    KioskAnalyzer,
    TextObjectAnalyzer,
    TrafficLightAnalyzer,
)
from vision_agent.event_manager import SceneEventManager
from vision_agent.narration import NarrationPolicy
from vision_agent.ocr import OcrLine, OcrResult
from vision_agent.signals import SignalStateResult
from vision_agent.types import Detection, SignalState
from vision_agent.vlm import VisionLanguageResult


def detection(
    frame_index: int,
    class_name: str,
    class_id: int,
    *,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        frame_index=frame_index,
        timestamp_s=frame_index / 30.0,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        xyxy=(0.0, 0.0, 200.0, 100.0),
        track_id=1,
    )


class RepeatingOcr:
    def __init__(self, result: OcrResult) -> None:
        self.result = result

    def recognize(self, image: np.ndarray) -> OcrResult:
        return self.result


class SequenceVlm:
    def __init__(self, *results: VisionLanguageResult) -> None:
        self.results = deque(results)

    def describe(self, image: np.ndarray, prompt: str) -> VisionLanguageResult:
        return self.results.popleft()


def test_text_analysis_flows_to_confirmed_narration() -> None:
    analyzer = TextObjectAnalyzer(
        RepeatingOcr(
            OcrResult(
                lines=(OcrLine("출구", 0.92, (10, 10, 80, 35)),),
                engine_name="fake",
            )
        )
    )
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)
    policy = NarrationPolicy()
    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    messages: list[str] = []

    for frame_index in range(3):
        result = analyzer.analyze(
            detection(frame_index, "sign", 11),
            stable_id="stable-1",
            crop=crop,
        )
        messages.extend(policy.narrate(manager.update([result], frame_index / 30.0)))

    assert messages == ["표지판에 출구라고 표시되어 있습니다."]


def test_initial_kiosk_screen_flows_to_button_narration() -> None:
    analyzer = KioskAnalyzer(
        RepeatingOcr(
            OcrResult(
                lines=(
                    OcrLine("매장 식사", 0.91, (10, 20, 90, 45)),
                    OcrLine("포장", 0.93, (100, 20, 160, 45)),
                ),
                engine_name="fake",
            )
        )
    )
    manager = SceneEventManager(auto_presence=False)
    policy = NarrationPolicy()
    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    messages: list[str] = []

    for frame_index in range(3):
        result = analyzer.analyze(
            detection(frame_index, "kiosk", 90),
            stable_id="stable-2",
            crop=crop,
        )
        messages.extend(policy.narrate(manager.update([result], frame_index / 30.0)))

    assert messages == ["매장 식사와 포장 중 하나를 선택하는 화면입니다."]


def test_generic_description_flows_to_narration_after_two_results() -> None:
    analyzer = GenericVisionAnalyzer(
        SequenceVlm(
            VisionLanguageResult("파란 자판기가 보입니다.", 0.8),
            VisionLanguageResult("파란 자판기가 보입니다.", 0.8),
        ),
        inference_interval_frames=1,
    )
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)
    policy = NarrationPolicy()
    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    messages: list[str] = []

    for frame_index in range(2):
        result = analyzer.analyze(
            detection(frame_index, "vending machine", 24),
            stable_id="stable-3",
            crop=crop,
        )
        messages.extend(policy.narrate(manager.update([result], frame_index / 30.0)))

    assert messages == ["파란 자판기가 보입니다."]


def test_low_detector_signal_cannot_narrate_until_a_new_reliable_transition() -> None:
    analyzer = TrafficLightAnalyzer(
        minimum_confirmed_frames=1,
        minimum_detection_confidence=0.2,
    )
    manager = SceneEventManager(auto_presence=False)
    policy = NarrationPolicy()

    def analyze(frame_index: int, state: SignalState, detector_confidence: float):
        return analyzer.analyze(
            detection(
                frame_index,
                "traffic light",
                9,
                confidence=detector_confidence,
            ),
            stable_id="stable-4",
            precomputed_signal_result=SignalStateResult(
                state=state,
                confidence=0.99,
                red_ratio=0.2 if state is SignalState.RED else 0.0,
                green_ratio=0.2 if state is SignalState.GREEN else 0.0,
            ),
        )

    messages: list[str] = []
    for frame_index, state, detector_confidence in (
        (0, SignalState.GREEN, 0.1),
        (1, SignalState.RED, 0.1),
        (2, SignalState.RED, 0.9),
        (3, SignalState.GREEN, 0.9),
    ):
        result = analyze(frame_index, state, detector_confidence)
        messages.extend(policy.narrate(manager.update([result], frame_index / 30.0)))

    assert messages == ["신호등 표시가 초록색으로 바뀌었습니다."]
