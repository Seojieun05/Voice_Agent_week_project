from __future__ import annotations

import pytest

from vision_agent.frame_buffer import FrameRingBuffer


def test_ring_buffer_keeps_only_newest_frames() -> None:
    buffer = FrameRingBuffer(max_frames=2)
    for sequence_id in (1, 2, 3):
        buffer.append(
            "session-a",
            f"jpeg-{sequence_id}".encode(),
            sequence_id=sequence_id,
            received_at_ms=sequence_id * 1000,
        )

    frames = buffer.recent("session-a")

    assert [frame.sequence_id for frame in frames] == [2, 3]
    assert frames[-1].jpeg_bytes == b"jpeg-3"


def test_recent_filters_stale_frames_and_unknown_sessions() -> None:
    buffer = FrameRingBuffer(max_frames=5)
    buffer.append("session-a", b"old", sequence_id=1, received_at_ms=0)
    buffer.append("session-a", b"fresh", sequence_id=2, received_at_ms=9_000)

    fresh = buffer.recent("session-a", max_age_ms=10_000, now_ms=12_000)

    assert [frame.jpeg_bytes for frame in fresh] == [b"fresh"]
    assert buffer.recent("unknown") == []


def test_sessions_evicted_least_recently_used() -> None:
    buffer = FrameRingBuffer(max_frames=2, max_sessions=2)
    buffer.append("first", b"1", sequence_id=1)
    buffer.append("second", b"2", sequence_id=1)
    buffer.append("first", b"1b", sequence_id=2)
    buffer.append("third", b"3", sequence_id=1)

    assert buffer.recent("second") == []
    assert len(buffer.recent("first")) == 2
    assert len(buffer.recent("third")) == 1


@pytest.mark.parametrize("kwargs", [{"max_frames": 0}, {"max_sessions": 0}])
def test_buffer_rejects_invalid_limits(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        FrameRingBuffer(**kwargs)
