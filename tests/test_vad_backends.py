from __future__ import annotations

import logging

import pytest
import torch

import vision_agent.vad_backends as vad_backends
from vision_agent.vad_backends import (
    SILERO_WINDOW_SAMPLES,
    SileroVadAdapter,
    WebRtcVadAdapter,
    create_vad,
)
from vision_agent.voice import FRAME_BYTES, SAMPLE_RATE, TurnDetector


class _FakeSileroModel:
    """Scripted stand-in for the Silero TorchScript module."""

    def __init__(self, probabilities: list[float] | None = None) -> None:
        self.probabilities = list(probabilities or [])
        self.windows: list[torch.Tensor] = []
        self.sample_rates: list[int] = []
        self.resets = 0

    def __call__(self, window: torch.Tensor, sample_rate: int) -> float:
        self.windows.append(window)
        self.sample_rates.append(sample_rate)
        return self.probabilities.pop(0) if self.probabilities else 0.0

    def reset_states(self) -> None:
        self.resets += 1


class _FakeWebRtcVad:
    """Deterministic stand-in for webrtcvad.Vad with a fixed verdict."""

    def __init__(self, result: bool) -> None:
        self.result = result
        self.calls: list[tuple[bytes, int]] = []

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        self.calls.append((frame, sample_rate))
        return self.result


def _frame(value: int = 0) -> bytes:
    # 20 ms / 320 samples of the repeated int16 value 0x<vv><vv>.
    return bytes([value % 256]) * FRAME_BYTES


def _sample(value: int) -> float:
    # The float the adapter must produce for one byte-repeated int16 sample.
    return (value % 256) * 0x0101 / 32768.0


def test_silero_adapter_buffers_320_sample_frames_into_512_sample_windows() -> None:
    model = _FakeSileroModel([0.9, 0.9])
    adapter = SileroVadAdapter(model=model)

    # One 320-sample frame cannot fill a 512-sample window: no inference,
    # and the initial probability (0.0) stays below the threshold.
    assert adapter.is_speech(_frame(1), SAMPLE_RATE) is False
    assert model.windows == []

    # The second frame completes a window; the verdict updates.
    assert adapter.is_speech(_frame(2), SAMPLE_RATE) is True
    assert len(model.windows) == 1
    window = model.windows[0]
    assert window.shape == (SILERO_WINDOW_SAMPLES,)
    assert window.dtype == torch.float32
    assert model.sample_rates == [SAMPLE_RATE]
    # int16 -> float32 normalization: 320 samples of frame 1, 192 of frame 2.
    assert float(window[0]) == pytest.approx(_sample(1))
    assert float(window[-1]) == pytest.approx(_sample(2))

    # 128 leftover samples + one frame is still short; the next frame after
    # that completes the second window.
    assert adapter.is_speech(_frame(3), SAMPLE_RATE) is True  # stale but recent verdict
    assert len(model.windows) == 1
    adapter.is_speech(_frame(4), SAMPLE_RATE)
    assert len(model.windows) == 2


def test_silero_adapter_switches_to_the_guard_threshold_when_echo_guard_is_set() -> None:
    model = _FakeSileroModel([0.6, 0.8, 0.8])
    adapter = SileroVadAdapter(model=model, threshold=0.5, guard_threshold=0.75)

    adapter.is_speech(_frame(), SAMPLE_RATE)
    assert adapter.is_speech(_frame(), SAMPLE_RATE) is True  # 0.6 >= 0.5

    # Same latest probability, raised bar: no longer speech.
    adapter.echo_guard = True
    assert adapter.is_speech(_frame(), SAMPLE_RATE) is False  # 0.6 < 0.75

    # A confident probability still clears the guard bar.
    assert adapter.is_speech(_frame(), SAMPLE_RATE) is True  # 0.8 >= 0.75

    adapter.echo_guard = False
    assert adapter.is_speech(_frame(), SAMPLE_RATE) is True  # 0.8 >= 0.5


def test_silero_adapter_reset_clears_buffer_probability_and_model_state() -> None:
    model = _FakeSileroModel([0.9, 0.9])
    adapter = SileroVadAdapter(model=model)
    adapter.is_speech(_frame(1), SAMPLE_RATE)
    assert adapter.is_speech(_frame(2), SAMPLE_RATE) is True

    adapter.reset()
    assert model.resets == 1

    # The stale probability is gone: one post-reset frame cannot fill a
    # window, so the verdict is silence again.
    assert adapter.is_speech(_frame(3), SAMPLE_RATE) is False
    assert len(model.windows) == 1

    # The 128 leftover samples are gone too: the second window is built from
    # post-reset frames only, so it starts with frame 3, not frame 2.
    adapter.is_speech(_frame(4), SAMPLE_RATE)
    assert len(model.windows) == 2
    assert float(model.windows[1][0]) == pytest.approx(_sample(3))


@pytest.mark.parametrize(
    ("threshold", "guard_threshold"),
    [(0.0, 0.75), (1.5, 0.75), (0.5, 0.0), (0.5, 0.4)],
)
def test_silero_adapter_rejects_invalid_thresholds(
    threshold: float,
    guard_threshold: float,
) -> None:
    with pytest.raises(ValueError):
        SileroVadAdapter(
            threshold=threshold,
            guard_threshold=guard_threshold,
            model=_FakeSileroModel(),
        )


def test_webrtc_adapter_switches_to_the_guard_instance() -> None:
    normal = _FakeWebRtcVad(result=True)
    guard = _FakeWebRtcVad(result=False)
    adapter = WebRtcVadAdapter(vad=normal, guard_vad=guard)

    assert adapter.is_speech(_frame(), SAMPLE_RATE) is True
    assert len(normal.calls) == 1 and guard.calls == []

    adapter.echo_guard = True
    assert adapter.is_speech(_frame(), SAMPLE_RATE) is False
    assert len(normal.calls) == 1 and len(guard.calls) == 1
    assert guard.calls[0] == (_frame(), SAMPLE_RATE)

    adapter.reset()  # stateless: must simply not raise
    adapter.echo_guard = False
    assert adapter.is_speech(_frame(), SAMPLE_RATE) is True


def test_create_vad_falls_back_to_webrtc_when_silero_fails_to_load(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken_adapter(**_kwargs: object) -> object:
        raise ImportError("silero_vad is not installed")

    monkeypatch.setattr(vad_backends, "SileroVadAdapter", broken_adapter)
    with caplog.at_level(logging.WARNING, logger="vision_agent.vad_backends"):
        vad = create_vad("silero", aggressiveness=2)

    assert isinstance(vad, WebRtcVadAdapter)
    assert any("falling back to webrtcvad" in record.message for record in caplog.records)


def test_create_vad_builds_the_webrtc_backend_directly() -> None:
    assert isinstance(create_vad("webrtc", aggressiveness=1), WebRtcVadAdapter)


def test_create_vad_keeps_configuration_errors_loud() -> None:
    # Unknown backends and bad thresholds are operator mistakes, not
    # environment failures: they must not silently fall back.
    with pytest.raises(ValueError):
        create_vad("bogus")
    with pytest.raises(ValueError):
        create_vad("silero", threshold=0.0)


class _RecordingVad:
    """Backend fake exposing the full adapter contract for TurnDetector."""

    def __init__(self) -> None:
        self.echo_guard = False
        self.resets = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return False

    def reset(self) -> None:
        self.resets += 1


def test_turn_detector_reset_resets_a_stateful_vad() -> None:
    vad = _RecordingVad()
    detector = TurnDetector(vad=vad)
    baseline = vad.resets  # the constructor already runs one reset()

    detector.reset()
    assert vad.resets == baseline + 1


def test_turn_detector_echo_guard_delegates_to_the_vad() -> None:
    vad = _RecordingVad()
    detector = TurnDetector(vad=vad)
    assert detector.echo_guard is False

    detector.echo_guard = True
    assert vad.echo_guard is True
    assert detector.echo_guard is True

    detector.echo_guard = False
    assert vad.echo_guard is False


def test_turn_detector_echo_guard_is_ignored_without_backend_support() -> None:
    class _PlainVad:
        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            return False

    detector = TurnDetector(vad=_PlainVad())
    detector.echo_guard = True  # plain webrtcvad path: silently ignored
    assert detector.echo_guard is False
