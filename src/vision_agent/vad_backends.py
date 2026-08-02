"""Pluggable VAD backends behind one ``is_speech``/``reset``/``echo_guard`` contract.

:class:`~vision_agent.voice.TurnDetector` only ever calls
``is_speech(frame, sample_rate)`` on its injected ``vad``, so any backend that
speaks that interface plugs in without touching the endpointing state machine.
The adapters here add two capabilities on top of that minimal contract:

* ``reset()`` clears per-utterance state between turns. Silero keeps an RNN
  hidden state across calls; letting it leak from one turn into the next
  would bias the onset of the following utterance.
* ``echo_guard`` raises the detection bar while the agent's own TTS may be
  audible on the client's speaker, so playback bleed is less likely to read
  as the user starting a turn (barge-in false positives).
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import numpy as np

LOGGER = logging.getLogger(__name__)

# Silero's streaming model accepts exactly one window size at 16 kHz (512
# samples, 32 ms); anything else is rejected by the TorchScript graph.
SILERO_WINDOW_SAMPLES = 512
_INT16_SCALE = 32768.0

# 프로세스 전역 모델 캐시. 로드는 최초 1회 수백 ms + torch import 1초대라
# 연결마다 반복하면 이벤트 루프가 그만큼 얼어붙는다(비전 릴레이까지 정지).
# /ws/audio는 동시 1세션(audio_gate)이므로 상태 있는 모델을 공유해도 안전하고,
# 어댑터 reset()이 턴 경계마다 reset_states()를 부른다.
_shared_model: object | None = None
_shared_model_lock = threading.Lock()


def _load_shared_silero_model() -> object:
    global _shared_model
    with _shared_model_lock:
        if _shared_model is None:
            import torch
            from silero_vad import load_silero_vad

            model = load_silero_vad()
            # JIT 워밍업: 첫 추론이 100ms+를 그래프 컴파일에 쓴다. 여기(로더,
            # to_thread 안)서 소진해 두면 첫 실프레임 판정이 늦지 않는다.
            with torch.no_grad():
                model(torch.zeros(SILERO_WINDOW_SAMPLES), 16000)
            model.reset_states()
            _shared_model = model
    return _shared_model


class VadBackendProtocol(Protocol):
    """Runtime contract shared by every VAD backend (and its test fakes)."""

    echo_guard: bool

    def is_speech(self, frame: bytes, sample_rate: int) -> bool: ...

    def reset(self) -> None: ...


class SileroVadAdapter:
    """Neural VAD (Silero) behind the frame-wise ``is_speech`` contract.

    The turn detector feeds 20 ms / 320-sample frames but the model only
    accepts 512-sample windows, so incoming frames accumulate in a byte
    buffer and inference runs once per full window. Between windows
    ``is_speech`` reuses the latest probability: the staleness is at most
    one 20 ms frame, well below the endpointing timescales (hundreds of ms).

    The ``model`` argument exists for testing: inject a callable returning
    scripted probabilities and no model is loaded.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        guard_threshold: float = 0.75,
        model: object | None = None,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be within (0, 1]")
        if not 0.0 < guard_threshold <= 1.0:
            raise ValueError("guard_threshold must be within (0, 1]")
        if guard_threshold < threshold:
            raise ValueError("guard_threshold must not be below threshold")
        # torch arrives alongside silero-vad; both stay optional for
        # installations that never use audio (mirrors webrtcvad in voice.py).
        import torch

        if model is None:
            model = _load_shared_silero_model()
        self._torch = torch
        self._model = model
        self._threshold = threshold
        self._guard_threshold = guard_threshold
        self.echo_guard = False
        self._pending = bytearray()
        self._probability = 0.0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        self._pending.extend(frame)
        window_bytes = SILERO_WINDOW_SAMPLES * 2  # mono int16
        while len(self._pending) >= window_bytes:
            window = bytes(self._pending[:window_bytes])
            del self._pending[:window_bytes]
            samples = np.frombuffer(window, dtype=np.int16).astype(np.float32) / _INT16_SCALE
            with self._torch.no_grad():
                output = self._model(self._torch.from_numpy(samples), sample_rate)
            # The model returns a [1, 1] tensor; float() collapses it.
            self._probability = float(output)
        bar = self._guard_threshold if self.echo_guard else self._threshold
        return self._probability >= bar

    def reset(self) -> None:
        self._pending.clear()
        self._probability = 0.0
        # Clear the RNN state too: a hidden state carried over from the
        # previous turn would bias the first windows of the next one.
        self._model.reset_states()


class WebRtcVadAdapter:
    """The existing webrtcvad energy-based VAD behind the same contract.

    webrtcvad exposes no probability to raise a threshold on, so the echo
    guard switches to a maximum-aggressiveness (3) instance instead. Both
    instances can be injected for testing.
    """

    def __init__(
        self,
        *,
        aggressiveness: int = 2,
        vad: object | None = None,
        guard_vad: object | None = None,
    ) -> None:
        if vad is None or guard_vad is None:
            # webrtcvad stays optional for installations that never use audio.
            import webrtcvad

            if vad is None:
                vad = webrtcvad.Vad(aggressiveness)
            if guard_vad is None:
                guard_vad = webrtcvad.Vad(3)
        self._vad = vad
        self._guard_vad = guard_vad
        self.echo_guard = False

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        vad = self._guard_vad if self.echo_guard else self._vad
        return bool(vad.is_speech(frame, sample_rate))

    def reset(self) -> None:
        return None  # webrtcvad keeps no cross-frame state


def create_vad(
    backend: str,
    *,
    aggressiveness: int = 2,
    threshold: float = 0.5,
    guard_threshold: float = 0.75,
) -> VadBackendProtocol:
    """Build the configured VAD backend, degrading from silero to webrtc.

    Silero needs torch plus the bundled model file; when either is missing
    the audio path must still come up, so environment failures downgrade to
    the dependency that ships with the server extras instead of breaking the
    connection. Configuration mistakes (bad thresholds, unknown backend)
    stay loud: those are the operator's to fix.
    """
    normalized = backend.strip().lower()
    if normalized == "silero":
        try:
            return SileroVadAdapter(threshold=threshold, guard_threshold=guard_threshold)
        except ValueError:
            raise
        except Exception:
            LOGGER.warning(
                "silero VAD backend unavailable; falling back to webrtcvad",
                exc_info=True,
            )
            return WebRtcVadAdapter(aggressiveness=aggressiveness)
    if normalized == "webrtc":
        return WebRtcVadAdapter(aggressiveness=aggressiveness)
    raise ValueError(f"vad backend must be 'silero' or 'webrtc', got {backend!r}")
