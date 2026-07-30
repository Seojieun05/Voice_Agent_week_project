import pytest

from vision_agent.event_manager import (
    DESCRIPTION_CONFIRMED,
    OBJECT_APPEARED,
    OBJECT_APPROACHING,
    OBJECT_DISAPPEARED,
    OBJECT_STATE_CHANGED,
    SCREEN_CHANGED,
    TEXT_CONFIRMED,
)
from vision_agent.hazard import HAZARD_DETECTED
from vision_agent.narration import NarrationPolicy, NarrationScheduler
from vision_agent.types import AnalysisEvent


def event(
    event_type: str,
    *,
    object_type: str = "pedestrian_signal",
    stable_id: str = "stable-1",
    timestamp_s: float = 1.0,
    previous_state: str | None = None,
    current_state: str | None = None,
    confidence: float = 0.9,
    attributes: dict[str, object] | None = None,
    is_uncertain: bool = False,
) -> AnalysisEvent:
    return AnalysisEvent(
        event_type=event_type,
        object_type=object_type,
        stable_id=stable_id,
        timestamp_s=timestamp_s,
        previous_state=previous_state,
        current_state=current_state,
        confidence=confidence,
        attributes=attributes or {},
        is_uncertain=is_uncertain,
    )


def test_green_to_red_uses_fixed_pedestrian_signal_sentence() -> None:
    policy = NarrationPolicy()
    changed = event(
        OBJECT_STATE_CHANGED,
        previous_state="GREEN",
        current_state="RED",
    )

    assert policy.narrate(changed) == ["보행자 신호가 빨간색으로 바뀌었습니다."]


def test_repeated_identical_event_does_not_repeat_message() -> None:
    policy = NarrationPolicy(duplicate_cooldown_s=5.0)
    first = event(
        OBJECT_STATE_CHANGED,
        previous_state="GREEN",
        current_state="RED",
        timestamp_s=2.733,
    )
    repeated = event(
        OBJECT_STATE_CHANGED,
        previous_state="GREEN",
        current_state="RED",
        timestamp_s=2.8,
    )

    assert len(policy.narrate(first)) == 1
    assert policy.narrate(repeated) == []


@pytest.mark.parametrize(
    ("previous_state", "current_state", "is_uncertain"),
    [
        ("GREEN", "UNKNOWN", False),
        ("UNKNOWN", "RED", False),
        ("GREEN", "RED", True),
    ],
)
def test_unknown_or_uncertain_signal_does_not_generate_safety_sentence(
    previous_state: str,
    current_state: str,
    is_uncertain: bool,
) -> None:
    policy = NarrationPolicy()
    changed = event(
        OBJECT_STATE_CHANGED,
        previous_state=previous_state,
        current_state=current_state,
        is_uncertain=is_uncertain,
    )

    assert policy.narrate(changed) == []


def test_signal_change_wins_over_simultaneous_general_appearance() -> None:
    policy = NarrationPolicy()
    appeared = event(
        OBJECT_APPEARED,
        object_type="person",
        stable_id="stable-2",
    )
    changed = event(
        OBJECT_STATE_CHANGED,
        previous_state="GREEN",
        current_state="RED",
    )

    assert policy.narrate([appeared, changed]) == ["보행자 신호가 빨간색으로 바뀌었습니다."]


def test_general_appearance_and_disappearance_templates() -> None:
    policy = NarrationPolicy(
        max_messages_per_batch=2,
        presence_narration_object_types=("bus", "kiosk"),
    )

    messages = policy.narrate(
        [
            event(OBJECT_APPEARED, object_type="bus", stable_id="stable-7"),
            event(
                OBJECT_DISAPPEARED,
                object_type="kiosk",
                stable_id="stable-12",
            ),
        ]
    )

    assert messages == ["버스가 감지되었습니다.", "키오스크가 화면에서 사라졌습니다."]


@pytest.mark.parametrize("object_type", ["person", "car", "chair", "bottle"])
def test_general_coco_presence_is_muted_by_default(object_type: str) -> None:
    policy = NarrationPolicy(max_messages_per_batch=2)

    messages = policy.narrate(
        [
            event(OBJECT_APPEARED, object_type=object_type),
            event(OBJECT_DISAPPEARED, object_type=object_type),
        ]
    )

    assert messages == []


def test_bus_and_kiosk_templates_need_no_external_api() -> None:
    policy = NarrationPolicy(max_messages_per_batch=2)
    bus = event(
        OBJECT_APPROACHING,
        object_type="bus",
        stable_id="stable-7",
        attributes={"route_number": "3102"},
    )
    kiosk = event(
        SCREEN_CHANGED,
        object_type="kiosk",
        stable_id="stable-12",
        attributes={"visible_options": ["매장 식사", "포장"]},
    )

    assert policy.narrate([kiosk, bus]) == [
        "3102번 버스가 들어오고 있습니다.",
        "매장 식사와 포장 중 하나를 선택하는 화면입니다.",
    ]


def test_bus_approach_can_be_muted_without_muting_confirmed_route_number() -> None:
    policy = NarrationPolicy(
        max_messages_per_batch=2,
        allow_bus_approach=False,
    )
    approaching = event(
        OBJECT_APPROACHING,
        object_type="bus",
        stable_id="stable-7",
        attributes={"route_number": "3102"},
    )
    route_confirmed = event(
        TEXT_CONFIRMED,
        object_type="bus",
        stable_id="stable-7",
        attributes={"route_number": "3102"},
    )

    assert policy.narrate([approaching, route_confirmed]) == ["3102번 버스입니다."]
    assert approaching.event_type == OBJECT_APPROACHING


def test_distinct_kiosk_screen_fingerprints_are_not_deduplicated() -> None:
    policy = NarrationPolicy()
    first = event(
        SCREEN_CHANGED,
        object_type="kiosk",
        stable_id="stable-12",
        timestamp_s=1.0,
        attributes={
            "visible_options": ["확인"],
            "screen_fingerprint": "screen-a",
        },
    )
    changed = event(
        SCREEN_CHANGED,
        object_type="kiosk",
        stable_id="stable-12",
        timestamp_s=2.0,
        attributes={
            "visible_options": ["확인"],
            "screen_fingerprint": "screen-b",
        },
    )

    assert policy.narrate(first) == ["키오스크 화면에 확인 선택지가 있습니다."]
    assert policy.narrate(changed) == ["키오스크 화면에 확인 선택지가 있습니다."]


def test_confirmed_bus_number_and_sign_text_templates() -> None:
    policy = NarrationPolicy(max_messages_per_batch=2)

    messages = policy.narrate(
        [
            event(
                TEXT_CONFIRMED,
                object_type="bus",
                stable_id="stable-7",
                attributes={"route_number": "3102"},
            ),
            event(
                TEXT_CONFIRMED,
                object_type="sign",
                stable_id="stable-8",
                attributes={"text": "출구"},
            ),
        ]
    )

    assert messages == ["3102번 버스입니다.", "표지판에 출구라고 표시되어 있습니다."]


@pytest.mark.parametrize(
    ("object_type", "expected_label"),
    [
        ("stop_sign", "표지판"),
        ("monitor", "화면"),
        ("tv", "화면"),
    ],
)
def test_coco_text_aliases_use_korean_labels(
    object_type: str,
    expected_label: str,
) -> None:
    policy = NarrationPolicy()

    messages = policy.narrate(
        event(
            TEXT_CONFIRMED,
            object_type=object_type,
            attributes={"text": "안내"},
        )
    )

    assert messages == [f"{expected_label}에 안내라고 표시되어 있습니다."]


def test_confirmed_generic_description_uses_only_stabilized_backend_text() -> None:
    policy = NarrationPolicy()
    described = event(
        DESCRIPTION_CONFIRMED,
        object_type="vending_machine",
        stable_id="stable-20",
        confidence=0.7,
        attributes={"description": "빨간 자판기가 보입니다."},
    )

    assert policy.narrate(described) == ["빨간 자판기가 보입니다."]


def test_explicit_low_confidence_event_is_suppressed() -> None:
    policy = NarrationPolicy(
        minimum_confidence=0.8,
        presence_narration_object_types=("pedestrian_signal",),
    )

    assert policy.narrate(event(OBJECT_APPEARED, confidence=0.4)) == []
    assert policy.narrate(event(OBJECT_APPEARED, confidence=0.0)) == []


def test_same_semantic_event_can_be_narrated_after_cooldown() -> None:
    policy = NarrationPolicy(duplicate_cooldown_s=5.0)
    first = event(
        OBJECT_STATE_CHANGED,
        previous_state="GREEN",
        current_state="RED",
        timestamp_s=1.0,
    )
    later = event(
        OBJECT_STATE_CHANGED,
        previous_state="GREEN",
        current_state="RED",
        timestamp_s=7.0,
    )

    assert len(policy.narrate(first)) == 1
    assert len(policy.narrate(later)) == 1


def test_scheduler_retains_simultaneous_events_and_returns_priority_first() -> None:
    scheduler = NarrationScheduler()
    route = event(
        TEXT_CONFIRMED,
        object_type="bus",
        stable_id="stable-7",
        attributes={"route_number": "3102"},
    )
    signal = event(
        OBJECT_STATE_CHANGED,
        previous_state="GREEN",
        current_state="RED",
    )

    scheduler.enqueue([route, signal], now_s=1.0)
    first = scheduler.pop_next(now_s=1.0)
    second = scheduler.pop_next(now_s=1.0)

    assert first is not None
    assert first.message == "보행자 신호가 빨간색으로 바뀌었습니다."
    assert second is not None
    assert second.message == "3102번 버스입니다."
    assert scheduler.pop_next(now_s=1.0) is None


def test_scheduler_discards_expired_event_by_event_type_ttl() -> None:
    scheduler = NarrationScheduler(
        default_ttl_s=10.0,
        ttl_by_event_type={TEXT_CONFIRMED: 1.0},
    )
    route = event(
        TEXT_CONFIRMED,
        object_type="bus",
        stable_id="stable-7",
        attributes={"route_number": "3102"},
    )

    scheduler.enqueue(route, now_s=0.0)

    assert len(scheduler) == 1
    assert scheduler.pop_next(now_s=1.0) is None
    assert len(scheduler) == 0


def test_scheduler_deduplicates_queued_and_recently_emitted_semantic_event() -> None:
    scheduler = NarrationScheduler(duplicate_cooldown_s=5.0)
    first = event(
        OBJECT_STATE_CHANGED,
        previous_state="GREEN",
        current_state="RED",
        timestamp_s=1.0,
    )
    repeated = event(
        OBJECT_STATE_CHANGED,
        previous_state="GREEN",
        current_state="RED",
        timestamp_s=2.0,
    )

    scheduler.enqueue([first, repeated], now_s=0.0)

    assert len(scheduler) == 1
    assert scheduler.pop_next(now_s=0.0) is not None
    scheduler.enqueue(repeated, now_s=1.0)
    assert scheduler.pop_next(now_s=1.0) is None
    scheduler.enqueue(repeated, now_s=5.0)
    assert scheduler.pop_next(now_s=5.0) is not None


def test_scheduler_deduplicates_bus_approach_across_reconnected_stable_ids() -> None:
    scheduler = NarrationScheduler(duplicate_cooldown_s=5.0)
    first = event(
        OBJECT_APPROACHING,
        object_type="bus",
        stable_id="stable-1",
        timestamp_s=1.0,
    )
    reconnected = event(
        OBJECT_APPROACHING,
        object_type="bus",
        stable_id="stable-2",
        timestamp_s=3.0,
    )

    scheduler.enqueue(first, now_s=1.0)
    assert scheduler.pop_next(now_s=1.0) is not None

    scheduler.enqueue(reconnected, now_s=3.0)
    assert scheduler.pop_next(now_s=3.0) is None

    scheduler.enqueue(reconnected, now_s=6.0)
    assert scheduler.pop_next(now_s=6.0) is not None


def test_scheduler_capacity_preserves_higher_priority_candidate() -> None:
    scheduler = NarrationScheduler(max_queue_size=1)
    sign = event(
        TEXT_CONFIRMED,
        object_type="sign",
        stable_id="stable-8",
        attributes={"text": "출구"},
    )
    signal = event(
        OBJECT_STATE_CHANGED,
        previous_state="GREEN",
        current_state="RED",
    )

    scheduler.enqueue(sign, now_s=1.0)
    scheduler.enqueue(signal, now_s=1.0)
    selected = scheduler.pop_next(now_s=1.0)

    assert selected is not None
    assert selected.event is signal


def test_scheduler_reset_clears_queue_and_duplicate_history() -> None:
    scheduler = NarrationScheduler(duplicate_cooldown_s=30.0)
    signal = event(
        OBJECT_STATE_CHANGED,
        previous_state="GREEN",
        current_state="RED",
    )
    scheduler.enqueue(signal, now_s=1.0)
    assert scheduler.pop_next(now_s=1.0) is not None

    scheduler.enqueue(
        event(TEXT_CONFIRMED, object_type="sign", attributes={"text": "출구"}),
        now_s=2.0,
    )
    scheduler.reset()

    assert len(scheduler) == 0
    assert scheduler.pop_next(now_s=2.0) is None
    scheduler.enqueue(signal, now_s=2.0)
    assert scheduler.pop_next(now_s=2.0) is not None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"minimum_confidence": -0.1}, "minimum_confidence"),
        ({"duplicate_cooldown_s": -1.0}, "duplicate_cooldown_s"),
        ({"max_messages_per_batch": 0}, "max_messages_per_batch"),
    ],
)
def test_invalid_policy_configuration_is_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        NarrationPolicy(**kwargs)  # type: ignore[arg-type]


def hazard_event(
    *,
    level: str = "IMMINENT",
    object_type: str = "bicycle",
    zone: str = "CENTER",
    stable_id: str = "stable-haz",
    timestamp_s: float = 1.0,
    confidence: float = 0.6,
    emission_index: int = 1,
) -> AnalysisEvent:
    return event(
        HAZARD_DETECTED,
        object_type=object_type,
        stable_id=stable_id,
        timestamp_s=timestamp_s,
        current_state=level,
        confidence=confidence,
        attributes={
            "hazard_level": level,
            "zone": zone,
            "time_to_contact_s": 1.0,
            "in_path": True,
            "emission_index": emission_index,
        },
    )


def test_hazard_outranks_every_other_event():
    policy = NarrationPolicy()
    signal = event(
        OBJECT_STATE_CHANGED,
        previous_state="RED",
        current_state="GREEN",
    )

    selected = policy.select([signal, hazard_event()])

    assert len(selected) == 1
    assert selected[0].event.event_type == HAZARD_DETECTED
    assert NarrationPolicy.priority_for(hazard_event()) == 0


def test_imminent_hazard_leads_with_the_action():
    policy = NarrationPolicy()

    message = policy.message_for(hazard_event(level="IMMINENT", zone="RIGHT"))

    assert message is not None
    # 끼어들기로 뒤가 잘려도 "위험, 멈추세요."는 이미 전달된다
    assert message.startswith("위험, 멈추세요.")
    assert "오른쪽에서 자전거가 빠르게 다가옵니다." in message


def test_warning_hazard_leads_with_the_object():
    policy = NarrationPolicy()

    message = policy.message_for(hazard_event(level="WARNING", object_type="scooter", zone="LEFT"))

    assert message == "킥보드가 왼쪽에서 다가옵니다. 주의하세요."


def test_hazard_survives_the_generic_confidence_floor():
    policy = NarrationPolicy(minimum_confidence=0.9)

    # ApproachMonitor가 자체 게이트를 통과시킨 위험은 일반 문턱으로 지우지 않는다
    assert policy.message_for(hazard_event(confidence=0.4)) is not None


def test_unknown_hazard_level_produces_no_message():
    policy = NarrationPolicy()

    assert policy.message_for(hazard_event(level="MAYBE")) is None


def test_repeated_hazard_is_not_suppressed_by_the_scheduler_cooldown():
    scheduler = NarrationScheduler(NarrationPolicy(duplicate_cooldown_s=5.0), default_ttl_s=5.0)

    scheduler.enqueue(hazard_event(emission_index=1, timestamp_s=1.0), now_s=1.0)
    first = scheduler.pop_next(now_s=1.0)
    # ApproachMonitor가 반복 주기를 지켜 다시 내보낸 위험은 새 정보로 취급한다
    scheduler.enqueue(hazard_event(emission_index=2, timestamp_s=3.0), now_s=3.0)
    second = scheduler.pop_next(now_s=3.0)

    assert first is not None
    assert second is not None


def test_same_hazard_emission_is_still_deduplicated():
    scheduler = NarrationScheduler(NarrationPolicy(duplicate_cooldown_s=5.0), default_ttl_s=5.0)

    scheduler.enqueue(hazard_event(emission_index=1, timestamp_s=1.0), now_s=1.0)
    assert scheduler.pop_next(now_s=1.0) is not None
    scheduler.enqueue(hazard_event(emission_index=1, timestamp_s=1.2), now_s=1.2)
    assert scheduler.pop_next(now_s=1.2) is None
