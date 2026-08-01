from __future__ import annotations

import io
import json
import wave
from collections.abc import Callable

import httpx
import pytest

from vision_agent.voice import (
    FRAME_BYTES,
    FRAME_MS,
    SAMPLE_RATE,
    GrokSpeechClient,
    SpeechConfig,
    SpeechServiceError,
    TurnDetector,
    wav_bytes,
)


class _ScriptedVad:
    """Deterministic stand-in for webrtcvad: pops one scripted result per frame."""

    def __init__(self, results: list[bool]) -> None:
        self.results = list(results)
        self.frames: list[bytes] = []

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        assert sample_rate == SAMPLE_RATE
        self.frames.append(frame)
        return self.results.pop(0) if self.results else False


def _frame(value: int) -> bytes:
    return bytes([value % 256]) * FRAME_BYTES


def _detector(vad: _ScriptedVad, **overrides: int) -> TurnDetector:
    knobs: dict[str, int] = {
        "silence_ms": 100,
        "prefix_ms": 100,
        "min_speech_ms": 60,
        "max_utterance_ms": 30_000,
    }
    knobs.update(overrides)
    return TurnDetector(vad=vad, **knobs)


def test_frame_contract_matches_webrtcvad_requirements() -> None:
    # 20 ms of mono int16 at 16 kHz: 320 samples, 640 bytes.
    assert FRAME_MS == 20
    assert FRAME_BYTES == 640


def test_single_hot_frame_does_not_start_a_turn() -> None:
    vad = _ScriptedVad([True, True] + [False] * 10)
    detector = _detector(vad)

    for index in range(12):
        assert detector.feed(_frame(index)) is None
        assert detector.speaking is False


def test_turn_includes_prefix_padding_and_trailing_silence() -> None:
    # Onset needs 3 speech frames in the last 5; the committed utterance must
    # start with the pre-onset ring buffer so first syllables survive.
    vad = _ScriptedVad([True, False, True, False, True] + [False] * 5)
    detector = _detector(vad)

    frames = [_frame(index) for index in range(10)]
    results = [detector.feed(frame) for frame in frames]

    assert results[:4] == [None] * 4
    assert detector.speaking is False  # committed and reset on the last frame
    assert results[9] == b"".join(frames)  # prefix + speech + trailing silence
    assert results[4] is None and all(result is None for result in results[5:9])


def test_mid_sentence_pause_shorter_than_silence_ms_does_not_split_the_turn() -> None:
    # 40 ms of silence mid-turn (below silence_ms=100) must not end the turn;
    # the silence counter resets on the next speech frame.
    vad = _ScriptedVad([True, True, True, False, False, True] + [False] * 5)
    detector = _detector(vad)

    frames = [_frame(index) for index in range(11)]
    results = [detector.feed(frame) for frame in frames]

    assert all(result is None for result in results[:10])
    assert results[10] == b"".join(frames)


def test_short_blip_is_discarded_without_returning_an_utterance() -> None:
    vad = _ScriptedVad([True, True, True] + [False] * 5)
    detector = _detector(vad, min_speech_ms=250)

    results = [detector.feed(_frame(index)) for index in range(8)]

    assert all(result is None for result in results)
    assert detector.speaking is False


def test_max_utterance_ms_forces_the_turn_to_commit() -> None:
    # Continuous speech never yields silence_ms of quiet; the cap must end
    # the turn anyway.
    vad = _ScriptedVad([True] * 12)
    detector = _detector(vad, silence_ms=1_000, prefix_ms=20, max_utterance_ms=200)

    results = [detector.feed(_frame(index)) for index in range(12)]

    committed = [result for result in results if result is not None]
    assert len(committed) == 1
    assert len(committed[0]) == 10 * FRAME_BYTES  # 200 ms of frames
    assert detector.speaking is False


def test_reset_clears_a_turn_in_progress() -> None:
    vad = _ScriptedVad([True, True, True, True])
    detector = _detector(vad)

    for index in range(4):
        detector.feed(_frame(index))
    assert detector.speaking is True

    detector.reset()
    assert detector.speaking is False
    # After reset the next silence does not commit anything.
    assert detector.feed(_frame(99)) is None


def test_wav_bytes_wraps_pcm_in_a_mono_16k_header() -> None:
    pcm = b"\x01\x02" * 320
    with wave.open(io.BytesIO(wav_bytes(pcm)), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == SAMPLE_RATE
        assert reader.readframes(reader.getnframes()) == pcm


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    **config_overrides: object,
) -> GrokSpeechClient:
    config = SpeechConfig(api_key="test-key", **config_overrides)  # type: ignore[arg-type]
    return GrokSpeechClient(config, transport=httpx.MockTransport(handler))


def test_transcribe_posts_wav_and_returns_text() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["authorization"] = request.headers["Authorization"]
        captured["content_type"] = request.headers["Content-Type"]
        captured["body"] = request.read()
        return httpx.Response(200, json={"text": "지금 앞에 뭐가 보여?"})

    client = _client(handler)
    assert client.transcribe(b"\x00\x01" * 320) == "지금 앞에 뭐가 보여?"
    assert captured["path"] == "/v1/stt"  # base_url path + endpoint
    assert captured["authorization"] == "Bearer test-key"
    assert str(captured["content_type"]).startswith("multipart/form-data")
    body = captured["body"]
    assert isinstance(body, bytes) and b"turn.wav" in body and b"RIFF" in body


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(500, True), (503, True), (400, False)],
)
def test_transcribe_maps_upstream_status_to_stt_error(status_code: int, retryable: bool) -> None:
    client = _client(lambda request: httpx.Response(status_code))
    with pytest.raises(SpeechServiceError) as excinfo:
        client.transcribe(b"\x00" * FRAME_BYTES)
    assert excinfo.value.code == "STT_ERROR"
    assert excinfo.value.retryable is retryable


def test_transcribe_maps_timeouts_and_transport_failures() -> None:
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(SpeechServiceError) as excinfo:
        _client(timeout_handler).transcribe(b"\x00" * FRAME_BYTES)
    assert excinfo.value.code == "STT_TIMEOUT"
    assert excinfo.value.retryable is True

    def broken_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(SpeechServiceError) as excinfo:
        _client(broken_handler).transcribe(b"\x00" * FRAME_BYTES)
    assert excinfo.value.code == "STT_UNAVAILABLE"


def test_transcribe_rejects_unexpected_response_shape() -> None:
    client = _client(lambda request: httpx.Response(200, json={"no_text": True}))
    with pytest.raises(SpeechServiceError) as excinfo:
        client.transcribe(b"\x00" * FRAME_BYTES)
    assert excinfo.value.code == "INVALID_STT_RESPONSE"


def test_synthesize_streams_chunks_and_sends_voice_settings() -> None:
    captured: dict[str, object] = {}
    audio = bytes(range(256)) * 24  # larger than one 4096-byte chunk

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.read())
        return httpx.Response(200, content=audio)

    client = _client(handler, tts_voice="eve", tts_language="auto")
    chunks = list(client.synthesize("안녕하세요"))

    assert b"".join(chunks) == audio
    assert len(chunks) > 1  # streamed, not one lump
    assert captured["path"] == "/v1/tts"  # base_url path + endpoint
    assert captured["body"] == {"text": "안녕하세요", "voice_id": "eve", "language": "auto"}


def test_synthesize_maps_upstream_status_to_tts_error() -> None:
    client = _client(lambda request: httpx.Response(502))
    with pytest.raises(SpeechServiceError) as excinfo:
        next(client.synthesize("hello"))
    assert excinfo.value.code == "TTS_ERROR"
    assert excinfo.value.retryable is True


def test_synthesize_maps_timeouts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(SpeechServiceError) as excinfo:
        next(_client(handler).synthesize("hello"))
    assert excinfo.value.code == "TTS_TIMEOUT"


def test_speech_config_from_environment_requires_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(SpeechServiceError) as excinfo:
        SpeechConfig.from_environment()
    assert excinfo.value.code == "MISSING_API_KEY"


def test_speech_config_from_environment_reads_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "fallback-key")
    monkeypatch.setenv("TTS_VOICE", "rex")
    monkeypatch.setenv("GROK_STT_TIMEOUT_S", "5")
    config = SpeechConfig.from_environment()
    assert config.api_key == "fallback-key"
    assert config.tts_voice == "rex"
    assert config.stt_timeout_s == pytest.approx(5.0)
    assert config.base_url == "https://api.x.ai/v1"


@pytest.mark.parametrize(
    ("field_name", "overrides"),
    [
        ("api_key", {"api_key": " "}),
        ("base_url", {"base_url": ""}),
        ("tts_voice", {"tts_voice": " "}),
        ("stt_timeout_s", {"stt_timeout_s": 0.0}),
        ("tts_timeout_s", {"tts_timeout_s": -1.0}),
    ],
)
def test_speech_config_rejects_invalid_values(
    field_name: str,
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {"api_key": "test-key"}
    values.update(overrides)
    with pytest.raises(ValueError, match=field_name):
        SpeechConfig(**values)  # type: ignore[arg-type]
