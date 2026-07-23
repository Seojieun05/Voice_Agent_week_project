from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np

from vision_agent.event_manager import OBJECT_STATE_CHANGED
from vision_agent.pipeline import (
    FrameContext,
    PipelineConfig,
    create_vision_session,
)
from vision_agent.signals import SignalStateResult
from vision_agent.types import SignalState


class _Tensor:
    def __init__(self, values: object) -> None:
        self.values = values

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def tolist(self) -> object:
        return self.values


class _Boxes:
    def __init__(
        self,
        *,
        class_id: int,
        confidence: float,
        track_id: int,
    ) -> None:
        self.xyxy = _Tensor([[1.0, 1.0, 15.0, 15.0]])
        self.conf = _Tensor([confidence])
        self.cls = _Tensor([float(class_id)])
        self.id = _Tensor([float(track_id)])

    def __len__(self) -> int:
        return 1


class _Result:
    def __init__(
        self,
        *,
        class_id: int,
        class_name: str,
        confidence: float,
        track_id: int,
    ) -> None:
        self.boxes = _Boxes(
            class_id=class_id,
            confidence=confidence,
            track_id=track_id,
        )
        self.names = {class_id: class_name}


class _Tracker:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _SequenceModel:
    def __init__(
        self,
        frames: list[tuple[int, str, float, int]],
    ) -> None:
        self.frames = deque(frames)
        self.track_calls: list[dict[str, object]] = []
        self.tracker = _Tracker()
        self.predictor = SimpleNamespace(trackers=[self.tracker])

    def track(
        self,
        frame: np.ndarray,
        **kwargs: object,
    ) -> list[_Result]:
        self.track_calls.append(dict(kwargs))
        class_id, class_name, confidence, track_id = self.frames.popleft()
        return [
            _Result(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                track_id=track_id,
            )
        ]


class _SignalSequence:
    def __init__(self, states: list[SignalState]) -> None:
        self.states = deque(states)

    def classify(self, crop: np.ndarray) -> SignalStateResult:
        state = self.states.popleft()
        return SignalStateResult(
            state=state,
            confidence=0.9,
            red_ratio=0.2 if state is SignalState.RED else 0.0,
            green_ratio=0.2 if state is SignalState.GREEN else 0.0,
        )


def _context(
    source_sequence_id: int,
    processed_index: int,
    *,
    processing_started_at_s: float,
) -> FrameContext:
    return FrameContext(
        source_sequence_id=source_sequence_id,
        processed_index=processed_index,
        captured_at_s=processed_index / 10.0,
        received_at_s=processing_started_at_s,
        processing_started_at_s=processing_started_at_s,
    )


def test_source_sequence_gaps_do_not_break_processed_signal_streak() -> None:
    states = [
        SignalState.GREEN,
        SignalState.GREEN,
        SignalState.GREEN,
        SignalState.RED,
        SignalState.RED,
        SignalState.RED,
    ]
    model = _SequenceModel([(9, "traffic light", 0.9, 1)] * len(states))
    session = create_vision_session(
        PipelineConfig(
            source="<live>",
            device="cpu",
            ocr_backend="none",
            min_seen_frames=1,
            min_signal_state_frames=3,
        ),
        live_mode=True,
        model=model,
        signal_classifier=_SignalSequence(states),
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    changes = []
    messages: list[str] = []

    for processed_index, source_sequence_id in enumerate((10, 30, 80, 120, 180, 250)):
        analysis = session.process_frame(
            frame,
            _context(
                source_sequence_id,
                processed_index,
                processing_started_at_s=processed_index * 0.1,
            ),
        )
        assert analysis.detections[0].frame_index == processed_index
        assert analysis.stable_keys_by_index[0] == "traffic light:stable-1"
        changes.extend(
            event for event in analysis.analysis_events if event.event_type == OBJECT_STATE_CHANGED
        )
        messages.extend(analysis.narrations)

    assert len(changes) == 1
    assert changes[0].previous_state == "GREEN"
    assert changes[0].current_state == "RED"
    assert messages == ["신호등 표시가 빨간색으로 바뀌었습니다."]


def test_long_processing_gap_resets_tracker_and_stable_ids() -> None:
    model = _SequenceModel(
        [
            (5, "bus", 0.9, 1),
            (2, "car", 0.9, 2),
        ]
    )
    session = create_vision_session(
        PipelineConfig(
            source="<live>",
            device="cpu",
            ocr_backend="none",
            classify_signal_states=False,
            min_seen_frames=1,
        ),
        live_mode=True,
        maximum_state_gap_s=0.5,
        model=model,
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    first = session.process_frame(
        frame,
        _context(1, 0, processing_started_at_s=10.0),
    )
    second = session.process_frame(
        frame,
        _context(2, 1, processing_started_at_s=11.0),
    )

    assert first.stable_keys_by_index[0] == "bus:stable-1"
    assert second.stable_keys_by_index[0] == "car:stable-1"
    assert model.tracker.reset_calls == 1


def test_live_tracker_defaults_to_botsort_and_allows_bytetrack_override() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    config = PipelineConfig(
        source="<live>",
        device="cpu",
        ocr_backend="none",
        classify_signal_states=False,
    )
    default_model = _SequenceModel([(5, "bus", 0.9, 1)])
    default_session = create_vision_session(
        config,
        live_mode=True,
        model=default_model,
    )
    default_session.process_frame(
        frame,
        _context(1, 0, processing_started_at_s=1.0),
    )

    override_model = _SequenceModel([(5, "bus", 0.9, 1)])
    override_session = create_vision_session(
        config,
        live_mode=True,
        tracker_override="bytetrack.yaml",
        model=override_model,
    )
    override_session.process_frame(
        frame,
        _context(1, 0, processing_started_at_s=1.0),
    )

    assert default_model.track_calls[0]["tracker"] == "botsort.yaml"
    assert override_model.track_calls[0]["tracker"] == "bytetrack.yaml"


def test_mp4_session_keeps_bytetrack_default() -> None:
    model = _SequenceModel([(5, "bus", 0.9, 1)])
    session = create_vision_session(
        PipelineConfig(
            source="video.mp4",
            device="cpu",
            ocr_backend="none",
            classify_signal_states=False,
        ),
        model=model,
    )

    session.process_frame(
        np.zeros((20, 20, 3), dtype=np.uint8),
        _context(0, 0, processing_started_at_s=1.0),
    )

    assert model.track_calls[0]["tracker"] == "bytetrack.yaml"
