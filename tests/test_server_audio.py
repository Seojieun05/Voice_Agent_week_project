from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vision_agent.chat import ChatServiceError
from vision_agent.server import ServerConfig, _transcript_echo_overlap, create_app
from vision_agent.voice import FRAME_BYTES, SpeechServiceError


class _FakeSession:
    model_load_ms = 1.0

    def process_frame(self, frame: object, context: object) -> object:  # pragma: no cover
        raise AssertionError("audio tests must not process vision frames")

    def reset(self) -> None:
        return None


class _FakeTurnDetector:
    """Scripted endpointing: commits a turn after ``commit_after`` frames.

    ``speculate_after`` scripts speculative STT: after that many frames of a
    turn, ``take_speculative()`` offers the audio so far. ``speculation_valid``
    controls whether the commit then reports that candidate as still matching
    the turn (``committed_speculation``), as the real detector does when the
    triggering pause runs to commit uninterrupted.
    """

    def __init__(
        self,
        commit_after: int = 3,
        speculate_after: int | None = None,
        speculation_valid: bool = True,
    ) -> None:
        self.commit_after = commit_after
        self.speculate_after = speculate_after
        self.speculation_valid = speculation_valid
        self.speaking = False
        self.echo_guard = False
        self.committed_speculation: int | None = None
        self.fed: list[bytes] = []
        # The guard state observed at each feed(): lets tests verify that
        # response-window frames were heard through the raised anti-echo bar.
        self.guard_log: list[bool] = []
        self.resets = 0
        self._buffer: list[bytes] = []
        self._generation = 0
        self._pending: tuple[int, bytes] | None = None
        self._live: int | None = None

    def feed(self, frame: bytes) -> bytes | None:
        self.fed.append(frame)
        self.guard_log.append(self.echo_guard)
        self._buffer.append(frame)
        self.speaking = True
        if self.speculate_after is not None and len(self._buffer) == self.speculate_after:
            self._generation += 1
            self._pending = (self._generation, b"".join(self._buffer))
            self._live = self._generation
        if len(self._buffer) >= self.commit_after:
            utterance = b"".join(self._buffer)
            self._buffer = []
            self.speaking = False
            self.committed_speculation = self._live if self.speculation_valid else None
            self._pending = None
            self._live = None
            return utterance
        return None

    def take_speculative(self) -> tuple[int, bytes] | None:
        pending, self._pending = self._pending, None
        return pending

    def reset(self) -> None:
        self.speaking = False
        self._buffer = []
        self._pending = None
        self._live = None
        self.committed_speculation = None
        self.resets += 1


class _FakeSpeechClient:
    def __init__(
        self,
        transcript: str = "지금 앞에 뭐가 보여?",
        chunks: tuple[bytes, ...] = (b"mp3-one", b"mp3-two"),
    ) -> None:
        self.transcript = transcript
        self.chunks = list(chunks)
        self.transcribe_calls: list[bytes] = []
        self.synthesize_calls: list[str] = []
        self.stt_error: Exception | None = None
        # Raised by the next transcribe() call only, then cleared: lets a
        # speculative call fail while the fallback call still succeeds.
        self.stt_error_once: Exception | None = None
        self.tts_error: Exception | None = None
        self.closed = False
        self.stt_started = threading.Event()
        self.stt_release: threading.Event | None = None

    def transcribe(self, pcm: bytes) -> str:
        self.transcribe_calls.append(pcm)
        self.stt_started.set()
        if self.stt_release is not None and not self.stt_release.wait(timeout=2.0):
            raise RuntimeError("stt_release was never set")
        if self.stt_error_once is not None:
            error, self.stt_error_once = self.stt_error_once, None
            raise error
        if self.stt_error is not None:
            raise self.stt_error
        return self.transcript

    def synthesize(self, text: str) -> Iterator[bytes]:
        self.synthesize_calls.append(text)
        if self.tts_error is not None:
            raise self.tts_error
        yield from self.chunks

    def close(self) -> None:
        self.closed = True


class _FakeChatClient:
    def __init__(self, answer: str = "테스트 답변입니다.") -> None:
        self.answer = answer
        self.calls: list[tuple[dict[str, object], str]] = []
        self.error: Exception | None = None
        self.closed = False

    def create_answer(self, scene_state: Mapping[str, object], user_question: str) -> str:
        self.calls.append((dict(scene_state), user_question))
        if self.error is not None:
            raise self.error
        return self.answer

    def close(self) -> None:
        self.closed = True


def _test_config(**overrides: object) -> ServerConfig:
    return ServerConfig(
        max_receive_fps=0.0,
        max_frame_bytes=1024 * 1024,
        max_frame_width=128,
        max_frame_height=128,
        **overrides,  # type: ignore[arg-type]
    )


def _audio_app(
    speech_client: _FakeSpeechClient,
    chat_client: _FakeChatClient,
    detector: _FakeTurnDetector,
    **config_overrides: object,
):
    return create_app(
        _test_config(**config_overrides),
        lambda: _FakeSession(),
        chat_client_factory=lambda: chat_client,
        speech_client_factory=lambda: speech_client,
        turn_detector_factory=lambda: detector,
    )


def _frame(value: int = 0) -> bytes:
    return bytes([value % 256]) * FRAME_BYTES


def _start_audio(websocket, session_id: str = "audio-session") -> None:
    websocket.send_json({"type": "start", "session_id": session_id})
    assert websocket.receive_json() == {"type": "ready", "session_id": session_id}


def _drive_turn(websocket, frames: int = 3) -> None:
    for index in range(frames):
        websocket.send_bytes(_frame(index))
    assert websocket.receive_json() == {"type": "vad", "speaking": True}
    assert websocket.receive_json() == {"type": "vad", "speaking": False}


def test_audio_turn_streams_transcript_reply_and_tts_chunks() -> None:
    speech = _FakeSpeechClient()
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3)
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            _drive_turn(websocket)

            # 3 frames x 640 bytes = 1920 bytes of PCM = 60 ms.
            assert websocket.receive_json() == {"type": "turn", "duration_ms": 60}
            assert websocket.receive_json() == {
                "type": "transcript",
                "text": "지금 앞에 뭐가 보여?",
            }
            assert websocket.receive_json() == {"type": "audio_start"}
            assert websocket.receive_bytes() == b"mp3-one"
            assert websocket.receive_bytes() == b"mp3-two"

            end = websocket.receive_json()
            assert end["type"] == "audio_end"
            assert end["reply"] == "테스트 답변입니다."
            assert end["tool_calls"] == []
            assert set(end["timings"]) == {"stt", "llm", "tts_first", "tts_total", "total"}
            assert all(value >= 0 for value in end["timings"].values())

            websocket.send_json({"type": "playback_done"})
            assert websocket.receive_json() == {"type": "listening"}

    assert speech.transcribe_calls == [_frame(0) + _frame(1) + _frame(2)]
    assert speech.synthesize_calls == ["테스트 답변입니다."]
    assert chat.calls[0][1] == "지금 앞에 뭐가 보여?"
    assert detector.resets >= 1


def test_valid_speculation_is_reused_instead_of_a_second_stt_call() -> None:
    # The candidate fires two frames in (audio = frames 0-1); the commit on
    # frame 2 reports it still valid, so the turn must reuse that result and
    # never transcribe the full utterance.
    speech = _FakeSpeechClient()
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3, speculate_after=2)
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            _drive_turn(websocket)

            assert websocket.receive_json() == {"type": "turn", "duration_ms": 60}
            assert websocket.receive_json() == {
                "type": "transcript",
                "text": "지금 앞에 뭐가 보여?",
            }
            assert websocket.receive_json() == {"type": "audio_start"}
            websocket.receive_bytes()
            websocket.receive_bytes()
            end = websocket.receive_json()
            assert end["type"] == "audio_end"
            assert end["timings"]["stt_speculative"] == 1

    assert speech.transcribe_calls == [_frame(0) + _frame(1)]


def test_stale_speculation_falls_back_to_transcribing_the_full_utterance() -> None:
    # speculation_valid=False models the user resuming speech after the
    # candidate fired: the early call ran (and is wasted), but the committed
    # turn must be transcribed from its full audio.
    speech = _FakeSpeechClient()
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3, speculate_after=2, speculation_valid=False)
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            _drive_turn(websocket)

            assert websocket.receive_json()["type"] == "turn"
            assert websocket.receive_json()["type"] == "transcript"
            assert websocket.receive_json() == {"type": "audio_start"}
            websocket.receive_bytes()
            websocket.receive_bytes()
            end = websocket.receive_json()
            assert end["type"] == "audio_end"
            assert "stt_speculative" not in end["timings"]

    full_utterance = _frame(0) + _frame(1) + _frame(2)
    speculative_pcm = _frame(0) + _frame(1)
    assert sorted(speech.transcribe_calls) == sorted([speculative_pcm, full_utterance])


def test_failed_speculation_falls_back_and_the_turn_still_completes() -> None:
    speech = _FakeSpeechClient()
    speech.stt_error_once = SpeechServiceError("STT_TIMEOUT", "boom", retryable=True)
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3, speculate_after=2)
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            _drive_turn(websocket)

            assert websocket.receive_json()["type"] == "turn"
            # The speculative call died, the fallback call answered: the turn
            # flow proceeds as if speculation never happened.
            assert websocket.receive_json() == {
                "type": "transcript",
                "text": "지금 앞에 뭐가 보여?",
            }
            assert websocket.receive_json() == {"type": "audio_start"}
            websocket.receive_bytes()
            websocket.receive_bytes()
            end = websocket.receive_json()
            assert end["type"] == "audio_end"
            assert "stt_speculative" not in end["timings"]

    assert speech.transcribe_calls == [
        _frame(0) + _frame(1),  # speculative call (raised)
        _frame(0) + _frame(1) + _frame(2),  # fallback on the full utterance
    ]


def test_speculative_stt_ms_must_stay_below_silence_ms() -> None:
    with pytest.raises(ValueError, match="speculative_stt_ms"):
        _test_config(speculative_stt_ms=700)  # equal to silence_ms: never usable
    with pytest.raises(ValueError, match="speculative_stt_ms"):
        _test_config(speculative_stt_ms=-1)
    assert _test_config(speculative_stt_ms=0).speculative_stt_ms == 0  # disabled


def test_audio_start_must_be_a_json_text_message() -> None:
    speech = _FakeSpeechClient()
    chat = _FakeChatClient()
    with TestClient(_audio_app(speech, chat, _FakeTurnDetector())) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            websocket.send_bytes(_frame())
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert response["code"] == "INVALID_START"
            with pytest.raises(WebSocketDisconnect) as disconnected:
                websocket.receive_json()
            assert disconnected.value.code == 1008


@pytest.mark.parametrize(
    "start_message",
    [
        {"type": "start"},
        {"type": "start", "session_id": ""},
        {"type": "start", "session_id": "s" * 129},
        {"type": "frame", "session_id": "audio-session"},
    ],
)
def test_audio_start_rejects_invalid_start_messages(start_message: dict[str, object]) -> None:
    speech = _FakeSpeechClient()
    chat = _FakeChatClient()
    with TestClient(_audio_app(speech, chat, _FakeTurnDetector())) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            websocket.send_json(start_message)
            response = websocket.receive_json()
            assert response["code"] == "INVALID_START"
            with pytest.raises(WebSocketDisconnect) as disconnected:
                websocket.receive_json()
            assert disconnected.value.code == 1008


def test_second_concurrent_audio_session_is_rejected_and_first_remains_usable() -> None:
    speech = _FakeSpeechClient()
    chat = _FakeChatClient()
    detector = _FakeTurnDetector()
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as first:
            _start_audio(first)
            with client.websocket_connect("/ws/audio") as second:
                response = second.receive_json()
                assert response["code"] == "SESSION_BUSY"
                with pytest.raises(WebSocketDisconnect) as disconnected:
                    second.receive_json()
                assert disconnected.value.code == 1013

            _drive_turn(first)
            assert first.receive_json()["type"] == "turn"


def test_audio_session_coexists_with_a_vision_session() -> None:
    speech = _FakeSpeechClient()
    chat = _FakeChatClient()
    with TestClient(_audio_app(speech, chat, _FakeTurnDetector())) as client:
        with client.websocket_connect("/ws/vision") as vision:
            vision.send_json(
                {
                    "type": "start",
                    "session_id": "shared-session",
                    "source_width": 64,
                    "source_height": 48,
                    "source_fps": 15.0,
                }
            )
            # The audio socket must not be blocked by the vision gate.
            with client.websocket_connect("/ws/audio") as audio:
                _start_audio(audio, session_id="shared-session")
                assert client.get("/health").json() == {
                    "status": "ok",
                    "active_session": True,
                }


def test_wrong_size_binary_frames_are_ignored() -> None:
    speech = _FakeSpeechClient()
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3)
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            websocket.send_bytes(b"partial")
            websocket.send_bytes(b"\x00" * (FRAME_BYTES + 2))
            _drive_turn(websocket)
            assert websocket.receive_json()["type"] == "turn"

    assert all(len(frame) == FRAME_BYTES for frame in detector.fed)
    assert len(detector.fed) == 3


def test_frames_during_a_response_are_dropped_when_barge_in_is_disabled() -> None:
    speech = _FakeSpeechClient()
    speech.stt_release = threading.Event()
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3)
    with TestClient(_audio_app(speech, chat, detector, barge_in_enabled=False)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            _drive_turn(websocket)
            assert speech.stt_started.wait(timeout=2.0)

            # The response is in flight and barge-in is off: these frames
            # must never reach the detector (the legacy echo defense).
            websocket.send_bytes(_frame(97))
            websocket.send_bytes(_frame(98))
            speech.stt_release.set()

            messages = [websocket.receive_json() for _ in range(3)]
            assert [message["type"] for message in messages] == [
                "turn",
                "transcript",
                "audio_start",
            ]
            websocket.receive_bytes()
            websocket.receive_bytes()
            assert websocket.receive_json()["type"] == "audio_end"
            websocket.send_json({"type": "playback_done"})
            assert websocket.receive_json() == {"type": "listening"}

    assert len(detector.fed) == 3


def test_barge_in_cancels_the_response_and_the_interrupting_turn_proceeds() -> None:
    speech = _FakeSpeechClient()
    speech.stt_release = threading.Event()
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3)
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            _drive_turn(websocket)
            assert websocket.receive_json()["type"] == "turn"
            assert speech.stt_started.wait(timeout=2.0)

            # Speech onset while the response is in flight: the server must
            # confirm the barge-in and cancel the respond task.
            websocket.send_bytes(_frame(50))
            assert websocket.receive_json() == {"type": "vad", "speaking": True}
            assert websocket.receive_json() == {"type": "interrupted"}
            speech.stt_release.set()  # release the cancelled turn's STT thread

            # The interrupting utterance ends: the cancelled response emits
            # nothing further (no transcript/audio_end), and a brand-new turn
            # runs to completion in order.
            websocket.send_bytes(_frame(51))
            websocket.send_bytes(_frame(52))
            assert websocket.receive_json() == {"type": "vad", "speaking": False}
            assert websocket.receive_json()["type"] == "turn"
            assert websocket.receive_json()["type"] == "transcript"
            assert websocket.receive_json()["type"] == "audio_start"
            websocket.receive_bytes()
            websocket.receive_bytes()
            assert websocket.receive_json()["type"] == "audio_end"
            websocket.send_json({"type": "playback_done"})
            assert websocket.receive_json() == {"type": "listening"}

    # The first turn was cancelled during STT, so only the interrupting turn
    # reached the chat client.
    assert speech.transcribe_calls == [
        _frame(0) + _frame(1) + _frame(2),
        _frame(50) + _frame(51) + _frame(52),
    ]
    assert len(chat.calls) == 1
    # Response-window audio was heard through the raised anti-echo bar, and
    # the guard dropped the moment the barge-in was confirmed.
    assert detector.guard_log == [False, False, False, True, False, False]


def test_barged_in_echo_transcript_skips_the_chat_call() -> None:
    # STT of the interrupting "speech" returns the very words the agent was
    # saying: that is the speaker bleeding into the mic, not the user.
    # 3+ tokens on purpose — shorter transcripts are exempt from the echo
    # verdict (_ECHO_MIN_TOKENS) so real one-word follow-ups survive.
    speech = _FakeSpeechClient(transcript="지금 신호가 초록불입니다")
    chat = _FakeChatClient(answer="지금 신호가 초록불입니다. 길을 건너세요.")
    detector = _FakeTurnDetector(commit_after=3)
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            _drive_turn(websocket)
            assert websocket.receive_json()["type"] == "turn"
            assert websocket.receive_json()["type"] == "transcript"
            assert websocket.receive_json()["type"] == "audio_start"
            websocket.receive_bytes()
            websocket.receive_bytes()
            assert websocket.receive_json()["type"] == "audio_end"

            # Playback is still running (no playback_done): an "utterance"
            # made of the reply's own words barges in.
            websocket.send_bytes(_frame(60))
            assert websocket.receive_json() == {"type": "vad", "speaking": True}
            assert websocket.receive_json() == {"type": "interrupted"}
            websocket.send_bytes(_frame(61))
            websocket.send_bytes(_frame(62))
            assert websocket.receive_json() == {"type": "vad", "speaking": False}
            assert websocket.receive_json()["type"] == "turn"
            # Echo verdict: no transcript, no chat call — straight back to
            # listening.
            assert websocket.receive_json() == {"type": "listening"}

    assert len(speech.transcribe_calls) == 2
    assert len(chat.calls) == 1
    assert speech.synthesize_calls == ["지금 신호가 초록불입니다. 길을 건너세요."]


def test_playback_done_is_ignored_while_the_user_is_speaking() -> None:
    speech = _FakeSpeechClient()
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3)
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            websocket.send_bytes(_frame(0))  # onset: detector.speaking = True
            assert websocket.receive_json() == {"type": "vad", "speaking": True}

            # A stray mark mid-utterance must not reset the turn in progress
            # (after a barge-in the client's player can still emit one).
            websocket.send_json({"type": "playback_done"})

            websocket.send_bytes(_frame(1))
            websocket.send_bytes(_frame(2))
            # Had the mark been honored, a "listening" (and a detector reset,
            # splitting the turn) would appear here instead.
            assert websocket.receive_json() == {"type": "vad", "speaking": False}
            assert websocket.receive_json()["type"] == "turn"
            assert websocket.receive_json()["type"] == "transcript"

    assert detector.resets == 0
    assert speech.transcribe_calls == [_frame(0) + _frame(1) + _frame(2)]


def test_transcript_echo_overlap_measures_the_transcripts_word_set() -> None:
    assert _transcript_echo_overlap("앞에 버스가 있어요", "앞에 버스가 있어요") == 1.0
    # Ratio is over the transcript's words: an echo fragment (3+ tokens) of a
    # longer reply still scores 1.0.
    assert (
        _transcript_echo_overlap("지금 길을 건너세요", "지금 신호가 초록불이니 길을 건너세요")
        == 1.0
    )
    assert _transcript_echo_overlap("앞에 버스가 오나요", "앞에 버스가 있어요") == pytest.approx(
        2 / 3
    )
    # Punctuation must not break matching: the reply carries "있습니다." while
    # STT yields "있습니다".
    assert _transcript_echo_overlap("앞에 버스가 있습니다", "앞에 버스가 있습니다.") == 1.0
    # Short interjections (< 3 tokens) are never judged as echo, even on a
    # perfect word match — a real "오른쪽?" follow-up must survive.
    assert _transcript_echo_overlap("오른쪽", "정류장은 오른쪽에 있습니다") == 0.0
    assert _transcript_echo_overlap("길을 건너세요", "지금 신호가 초록불이니 길을 건너세요") == 0.0
    # Case-insensitive for Latin fragments in Korean STT output (padded to
    # three tokens to clear the minimum).
    assert _transcript_echo_overlap("Bus 정류장 어디에", "bus 정류장") == pytest.approx(2 / 3)
    # Empty sides can never be judged as echo.
    assert _transcript_echo_overlap("", "테스트 답변입니다.") == 0.0
    assert _transcript_echo_overlap("테스트", "") == 0.0
    assert _transcript_echo_overlap("   ", "   ") == 0.0


def test_barge_in_env_var_disables_only_on_zero_or_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ServerConfig().barge_in_enabled is True
    for raw_value, expected in (
        ("0", False),
        ("false", False),
        ("FALSE", False),
        ("1", True),
        ("typo", True),  # a mistyped value must not silently disable barge-in
    ):
        monkeypatch.setenv("VISION_SERVER_BARGE_IN", raw_value)
        assert ServerConfig.from_environment().barge_in_enabled is expected
    monkeypatch.delenv("VISION_SERVER_BARGE_IN")
    assert ServerConfig.from_environment().barge_in_enabled is True


def test_stt_failure_sends_error_and_the_connection_recovers() -> None:
    speech = _FakeSpeechClient()
    speech.stt_error = SpeechServiceError(
        "STT_ERROR",
        "STT API returned status 500",
        retryable=True,
    )
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3)
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            _drive_turn(websocket)
            assert websocket.receive_json()["type"] == "turn"
            error = websocket.receive_json()
            assert error == {
                "type": "error",
                "code": "STT_ERROR",
                "message": "STT API returned status 500",
            }
            assert websocket.receive_json() == {"type": "listening"}
            assert chat.calls == []

            speech.stt_error = None
            _drive_turn(websocket)
            assert websocket.receive_json()["type"] == "turn"
            assert websocket.receive_json()["type"] == "transcript"


def test_tts_failure_sends_error_and_the_connection_recovers() -> None:
    speech = _FakeSpeechClient()
    speech.tts_error = SpeechServiceError(
        "TTS_ERROR",
        "TTS API returned status 502",
        retryable=True,
    )
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3)
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            _drive_turn(websocket)
            assert websocket.receive_json()["type"] == "turn"
            assert websocket.receive_json()["type"] == "transcript"
            assert websocket.receive_json()["type"] == "audio_start"
            error = websocket.receive_json()
            assert error["code"] == "TTS_ERROR"
            assert websocket.receive_json() == {"type": "listening"}

            speech.tts_error = None
            _drive_turn(websocket)
            assert websocket.receive_json()["type"] == "turn"


def test_chat_failure_is_reported_with_its_stable_code() -> None:
    speech = _FakeSpeechClient()
    chat = _FakeChatClient()
    chat.error = ChatServiceError(
        "UPSTREAM_TIMEOUT",
        "Grok API request timed out",
        retryable=True,
    )
    detector = _FakeTurnDetector(commit_after=3)
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            _drive_turn(websocket)
            assert websocket.receive_json()["type"] == "turn"
            assert websocket.receive_json()["type"] == "transcript"
            error = websocket.receive_json()
            assert error == {
                "type": "error",
                "code": "UPSTREAM_TIMEOUT",
                "message": "Grok API request timed out",
            }
            assert websocket.receive_json() == {"type": "listening"}
            assert speech.synthesize_calls == []


def test_empty_transcript_skips_the_chat_call() -> None:
    speech = _FakeSpeechClient(transcript="   ")
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3)
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            _drive_turn(websocket)
            assert websocket.receive_json()["type"] == "turn"
            assert websocket.receive_json() == {"type": "listening"}

    assert chat.calls == []
    assert speech.synthesize_calls == []


def test_long_transcripts_are_truncated_to_the_question_limit() -> None:
    speech = _FakeSpeechClient(transcript="가나다라마바사아자차카타")
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3)
    with TestClient(_audio_app(speech, chat, detector, max_question_length=10)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            _drive_turn(websocket)
            assert websocket.receive_json()["type"] == "turn"
            assert websocket.receive_json() == {
                "type": "transcript",
                "text": "가나다라마바사아자차",
            }
            assert websocket.receive_json()["type"] == "audio_start"

    assert chat.calls[0][1] == "가나다라마바사아자차"


def test_audio_answers_use_the_latest_scene_state() -> None:
    speech = _FakeSpeechClient()
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3)
    app = _audio_app(speech, chat, detector)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            app.state.scene_store.update(
                "audio-session",
                analysis_events=[],
                narrations=["보행자 신호가 초록불로 바뀌었습니다."],
                visible_objects=[{"object_type": "pedestrian_signal", "state": "GREEN"}],
                scene_confidence=0.9,
                raw_detections=[],
                updated_at_ms=time.time_ns() // 1_000_000,
            )
            _drive_turn(websocket)
            assert websocket.receive_json()["type"] == "turn"
            assert websocket.receive_json()["type"] == "transcript"
            assert websocket.receive_json()["type"] == "audio_start"

    scene_state = chat.calls[0][0]
    assert scene_state["latest_narrations"] == ["보행자 신호가 초록불로 바뀌었습니다."]
    assert scene_state["visible_objects"] == [
        {"object_type": "pedestrian_signal", "state": "GREEN"}
    ]


def test_playback_done_without_a_turn_is_safe() -> None:
    speech = _FakeSpeechClient()
    chat = _FakeChatClient()
    with TestClient(_audio_app(speech, chat, _FakeTurnDetector())) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            websocket.send_json({"type": "playback_done"})
            assert websocket.receive_json() == {"type": "listening"}


def test_unknown_control_messages_return_an_error_without_closing() -> None:
    speech = _FakeSpeechClient()
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3)
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            websocket.send_json({"type": "bogus"})
            assert websocket.receive_json()["code"] == "INVALID_MESSAGE"
            websocket.send_text("not json")
            assert websocket.receive_json()["code"] == "INVALID_MESSAGE"
            _drive_turn(websocket)
            assert websocket.receive_json()["type"] == "turn"


def test_speech_client_is_created_lazily_and_closed_on_shutdown() -> None:
    speech = _FakeSpeechClient()
    chat = _FakeChatClient()
    factory_calls = []

    def speech_factory() -> _FakeSpeechClient:
        factory_calls.append(True)
        return speech

    detector = _FakeTurnDetector(commit_after=3)
    app = create_app(
        _test_config(),
        lambda: _FakeSession(),
        chat_client_factory=lambda: chat,
        speech_client_factory=speech_factory,
        turn_detector_factory=lambda: detector,
    )
    with TestClient(app) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            assert factory_calls == []  # nothing built before the first turn

            _drive_turn(websocket)
            assert websocket.receive_json()["type"] == "turn"
            assert websocket.receive_json()["type"] == "transcript"
            assert websocket.receive_json()["type"] == "audio_start"
            websocket.receive_bytes()
            websocket.receive_bytes()
            assert websocket.receive_json()["type"] == "audio_end"
            websocket.send_json({"type": "playback_done"})
            assert websocket.receive_json() == {"type": "listening"}

            _drive_turn(websocket)
            assert websocket.receive_json()["type"] == "turn"

    assert factory_calls == [True]  # built once, reused across turns
    assert speech.closed is True


@pytest.mark.parametrize(
    ("field_name", "overrides"),
    [
        ("silence_ms", {"silence_ms": 0}),
        ("prefix_ms", {"prefix_ms": -1}),
        ("min_speech_ms", {"min_speech_ms": -1}),
        ("vad_aggressiveness", {"vad_aggressiveness": 4}),
        ("max_utterance_ms", {"max_utterance_ms": 700}),
    ],
)
def test_server_config_rejects_invalid_endpointing_values(
    field_name: str,
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=field_name):
        ServerConfig(**overrides)  # type: ignore[arg-type]
