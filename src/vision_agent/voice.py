"""Hands-free voice support: turn detection (VAD + endpointing) and speech I/O.

The client streams raw microphone audio as fixed-size PCM frames over
``/ws/audio``; the :class:`TurnDetector` decides when the user finished a
turn (endpointing) so no push-to-talk button is needed. Committed utterances
are transcribed and synthesized against the xAI speech endpoints, which
accept the same ``GROK_API_KEY``/``XAI_API_KEY`` credentials as the chat
client.

Everything here is synchronous and intended to run off the event loop (via
``asyncio.to_thread``), mirroring ``chat.GrokChatClient``.
"""

from __future__ import annotations

import io
import logging
import os
import wave
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import httpx

LOGGER = logging.getLogger(__name__)

DEFAULT_SPEECH_BASE_URL = "https://api.x.ai/v1"
SPEECH_API_KEY_ENVS = ("GROK_API_KEY", "XAI_API_KEY")

# The frame contract. webrtcvad accepts only 10/20/30 ms frames at
# 8/16/32/48 kHz, so every client must send exactly this shape: mono int16
# little-endian, 16 kHz, 20 ms per frame (320 samples, 640 bytes).
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2

# Onset debounce: declare a turn only after 3 speech frames in the last 5,
# so one hot frame (a door slam) is not a turn.
ONSET_FRAMES = 3
ONSET_WINDOW = 5


class SpeechServiceError(Exception):
    """Raised when speech I/O fails; carries a stable error code."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class SpeechClientProtocol(Protocol):
    """Runtime contract used by the audio WebSocket route and its fakes."""

    def transcribe(self, pcm: bytes) -> str: ...

    def synthesize(self, text: str) -> Iterator[bytes]: ...


class TurnDetectorProtocol(Protocol):
    """Runtime contract used by the audio WebSocket route and its fakes."""

    speaking: bool

    def feed(self, frame: bytes) -> bytes | None: ...

    def reset(self) -> None: ...


class TurnDetector:
    """Per-frame VAD plus an endpointing state machine (IDLE/SPEAKING).

    ``feed()`` is called with every 20 ms frame while the agent is idle. It
    returns ``None``, or -- the moment the turn ends -- the complete utterance
    as PCM bytes. ``speaking`` reflects whether a turn is in progress.

    The ``vad`` argument exists for testing: inject a stub with a scripted
    ``is_speech()`` and the state machine becomes fully deterministic.
    """

    def __init__(
        self,
        *,
        silence_ms: int = 700,
        prefix_ms: int = 300,
        min_speech_ms: int = 250,
        vad_aggressiveness: int = 2,
        max_utterance_ms: int = 30_000,
        vad: object | None = None,
    ) -> None:
        if vad is None:
            # webrtcvad stays optional for installations that never use audio.
            import webrtcvad

            vad = webrtcvad.Vad(vad_aggressiveness)
        self._vad = vad
        self._silence_ms = silence_ms
        self._prefix_ms = prefix_ms
        self._min_speech_ms = min_speech_ms
        self._max_utterance_ms = max_utterance_ms
        self.speaking = False
        self._frames: list[bytes] = []
        self._prefix: deque[bytes] = deque(maxlen=max(1, prefix_ms // FRAME_MS))
        self._onset: deque[bool] = deque(maxlen=ONSET_WINDOW)
        self._quiet_ms = 0
        self._speech_ms = 0
        self.reset()

    def reset(self) -> None:
        self.speaking = False
        self._frames = []
        self._prefix = deque(maxlen=max(1, self._prefix_ms // FRAME_MS))
        self._onset = deque(maxlen=ONSET_WINDOW)
        self._quiet_ms = 0
        self._speech_ms = 0
        # Stateful backends (Silero's RNN in vad_backends) must not carry
        # one turn's state into the next; plain webrtcvad has no reset().
        vad_reset = getattr(self._vad, "reset", None)
        if callable(vad_reset):
            vad_reset()

    @property
    def echo_guard(self) -> bool:
        """Whether the VAD currently applies its raised (anti-echo) bar."""
        return bool(getattr(self._vad, "echo_guard", False))

    @echo_guard.setter
    def echo_guard(self, value: bool) -> None:
        # Delegate to backends that support it (vad_backends adapters); a
        # plain webrtcvad instance (the default constructor path) has no
        # such knob, so the request is deliberately a no-op there.
        if hasattr(self._vad, "echo_guard"):
            self._vad.echo_guard = bool(value)

    def feed(self, frame: bytes) -> bytes | None:
        is_speech = bool(self._vad.is_speech(frame, SAMPLE_RATE))

        if not self.speaking:
            # IDLE: remember recent audio (prefix padding, so onset lag does
            # not eat the first syllable) and debounce the onset.
            self._prefix.append(frame)
            self._onset.append(is_speech)
            if sum(self._onset) >= ONSET_FRAMES:
                self.speaking = True
                self._frames = list(self._prefix)
                self._speech_ms = sum(self._onset) * FRAME_MS
                self._quiet_ms = 0
                LOGGER.info("[SPEECH START]")
            return None

        # SPEAKING: keep every frame (pauses are part of the audio), count
        # consecutive silence, and commit the turn at silence_ms.
        self._frames.append(frame)
        if is_speech:
            self._speech_ms += FRAME_MS
            self._quiet_ms = 0
        else:
            self._quiet_ms += FRAME_MS

        elapsed_ms = len(self._frames) * FRAME_MS
        forced = elapsed_ms >= self._max_utterance_ms
        if self._quiet_ms < self._silence_ms and not forced:
            return None

        utterance = b"".join(self._frames)
        speech_ms = self._speech_ms
        duration_s = elapsed_ms / 1000
        self.reset()

        if speech_ms < self._min_speech_ms:
            # A cough, not a turn: dropping it here saves an STT + chat call.
            LOGGER.info("[DISCARDED] %d ms of speech below min_speech_ms", speech_ms)
            return None

        if forced:
            LOGGER.info("[SPEECH END forced at %.1fs -> pipeline]", duration_s)
        else:
            LOGGER.info("[SPEECH END after %.1fs -> pipeline]", duration_s)
        return utterance


def wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw mono int16 PCM in a WAV header so the STT API can decode it."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return buffer.getvalue()


@dataclass(frozen=True, slots=True)
class SpeechConfig:
    api_key: str
    base_url: str = DEFAULT_SPEECH_BASE_URL
    tts_voice: str = "ara"
    tts_language: str = "auto"
    stt_timeout_s: float = 20.0
    tts_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not self.tts_voice.strip():
            raise ValueError("tts_voice must not be empty")
        if self.stt_timeout_s <= 0.0:
            raise ValueError("stt_timeout_s must be positive")
        if self.tts_timeout_s <= 0.0:
            raise ValueError("tts_timeout_s must be positive")

    @classmethod
    def from_environment(cls) -> SpeechConfig:
        api_key = ""
        for name in SPEECH_API_KEY_ENVS:
            raw_value = os.getenv(name)
            if raw_value is not None and raw_value.strip():
                api_key = raw_value.strip()
                break
        if not api_key:
            raise SpeechServiceError(
                "MISSING_API_KEY",
                "speech API key is not configured; set GROK_API_KEY or XAI_API_KEY",
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv("GROK_BASE_URL", DEFAULT_SPEECH_BASE_URL).strip()
            or DEFAULT_SPEECH_BASE_URL,
            tts_voice=os.getenv("TTS_VOICE", "ara").strip() or "ara",
            tts_language=os.getenv("TTS_LANGUAGE", "auto").strip() or "auto",
            stt_timeout_s=float(os.getenv("GROK_STT_TIMEOUT_S", "20")),
            tts_timeout_s=float(os.getenv("GROK_TTS_TIMEOUT_S", "30")),
        )


class GrokSpeechClient:
    """Synchronous xAI speech client (STT upload, streamed TTS download).

    The blocking HTTP calls are intended to run off the event loop (via
    ``asyncio.to_thread``). A custom ``transport`` can be injected so tests
    never reach the network.
    """

    def __init__(
        self,
        config: SpeechConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            transport=transport,
            headers={"Authorization": f"Bearer {config.api_key}"},
        )

    @classmethod
    def from_environment(cls) -> GrokSpeechClient:
        return cls(SpeechConfig.from_environment())

    def close(self) -> None:
        self._client.close()

    def transcribe(self, pcm: bytes) -> str:
        try:
            response = self._client.post(
                "/stt",
                files={"file": ("turn.wav", wav_bytes(pcm), "audio/wav")},
                timeout=self._config.stt_timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise SpeechServiceError(
                "STT_TIMEOUT",
                "STT API request timed out",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise SpeechServiceError(
                "STT_UNAVAILABLE",
                "STT API request failed",
                retryable=True,
            ) from exc

        if response.status_code != 200:
            # Do not log or forward the upstream body: clients only need a
            # stable code (same policy as the chat client).
            LOGGER.warning("STT API returned status %s", response.status_code)
            raise SpeechServiceError(
                "STT_ERROR",
                f"STT API returned status {response.status_code}",
                retryable=response.status_code >= 500,
            )

        try:
            text = response.json()["text"]
        except (ValueError, KeyError, TypeError) as exc:
            raise SpeechServiceError(
                "INVALID_STT_RESPONSE",
                "STT API returned an unexpected response shape",
            ) from exc
        return str(text)

    def synthesize(self, text: str) -> Iterator[bytes]:
        """Yield encoded audio chunks as the TTS API produces them.

        Chunks are relayed downstream as they arrive so playback can start at
        chunk #1 instead of after the full download (time-to-first-audio).
        """
        body = {
            "text": text,
            "voice_id": self._config.tts_voice,
            "language": self._config.tts_language,
        }
        try:
            with self._client.stream(
                "POST",
                "/tts",
                json=body,
                timeout=self._config.tts_timeout_s,
            ) as response:
                if response.status_code != 200:
                    LOGGER.warning("TTS API returned status %s", response.status_code)
                    raise SpeechServiceError(
                        "TTS_ERROR",
                        f"TTS API returned status {response.status_code}",
                        retryable=response.status_code >= 500,
                    )
                for chunk in response.iter_bytes(chunk_size=4096):
                    if chunk:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise SpeechServiceError(
                "TTS_TIMEOUT",
                "TTS API request timed out",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise SpeechServiceError(
                "TTS_UNAVAILABLE",
                "TTS API request failed",
                retryable=True,
            ) from exc
