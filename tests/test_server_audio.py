from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vision_agent.chat import ChatServiceError
from vision_agent.server import ServerConfig, create_app
from vision_agent.voice import FRAME_BYTES, SpeechServiceError


class _FakeSession:
    model_load_ms = 1.0

    def process_frame(self, frame: object, context: object) -> object:  # pragma: no cover
        raise AssertionError("audio tests must not process vision frames")

    def reset(self) -> None:
        return None


class _FakeTurnDetector:
    """Scripted endpointing: commits a turn after ``commit_after`` frames."""

    def __init__(self, commit_after: int = 3) -> None:
        self.commit_after = commit_after
        self.speaking = False
        self.fed: list[bytes] = []
        self.resets = 0
        self._buffer: list[bytes] = []

    def feed(self, frame: bytes) -> bytes | None:
        self.fed.append(frame)
        self._buffer.append(frame)
        self.speaking = True
        if len(self._buffer) >= self.commit_after:
            utterance = b"".join(self._buffer)
            self._buffer = []
            self.speaking = False
            return utterance
        return None

    def reset(self) -> None:
        self.speaking = False
        self._buffer = []
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
        self.tts_error: Exception | None = None
        self.closed = False
        self.stt_started = threading.Event()
        self.stt_release: threading.Event | None = None

    def transcribe(self, pcm: bytes) -> str:
        self.transcribe_calls.append(pcm)
        self.stt_started.set()
        if self.stt_release is not None and not self.stt_release.wait(timeout=2.0):
            raise RuntimeError("stt_release was never set")
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


def test_frames_during_a_response_are_dropped() -> None:
    speech = _FakeSpeechClient()
    speech.stt_release = threading.Event()
    chat = _FakeChatClient()
    detector = _FakeTurnDetector(commit_after=3)
    with TestClient(_audio_app(speech, chat, detector)) as client:
        with client.websocket_connect("/ws/audio") as websocket:
            _start_audio(websocket)
            _drive_turn(websocket)
            assert speech.stt_started.wait(timeout=2.0)

            # The response is in flight: these frames must never reach the
            # detector (that branch becomes barge-in later).
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
