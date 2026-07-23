from vision_agent.event_manager import (
    DESCRIPTION_CONFIRMED,
    OBJECT_APPEARED,
    OBJECT_APPROACHING,
    OBJECT_DISAPPEARED,
    OBJECT_STATE_CHANGED,
    SCREEN_CHANGED,
    TEXT_CONFIRMED,
    SceneEventManager,
)
from vision_agent.narration import NarrationPolicy
from vision_agent.types import AnalysisResult, SceneEvent, SignalState


def result(
    state: str | None,
    *,
    stable_id: str = "stable-1",
    object_type: str = "pedestrian_signal",
    confidence: float = 0.9,
    is_uncertain: bool = False,
) -> AnalysisResult:
    return AnalysisResult(
        object_type=object_type,
        stable_id=stable_id,
        state=state,
        confidence=confidence,
        attributes={"confirmed_frames": 3},
        is_uncertain=is_uncertain,
    )


def legacy_event(
    event_type: str,
    timestamp_s: float,
    *,
    previous_state: SignalState | None = None,
    current_state: SignalState | None = None,
) -> SceneEvent:
    return SceneEvent(
        event_type=event_type,
        object_key="traffic light:stable-1",
        class_name="traffic light",
        timestamp_s=timestamp_s,
        message="legacy message",
        previous_state=previous_state,
        current_state=current_state,
    )


def test_standalone_first_result_appears_and_missing_result_disappears() -> None:
    manager = SceneEventManager()

    appeared = manager.update([result("GREEN")], 0.0)
    unchanged = manager.update([result("GREEN")], 0.1)
    disappeared = manager.update([], 0.2)

    assert [event.event_type for event in appeared] == [OBJECT_APPEARED]
    assert appeared[0].stable_id == "stable-1"
    assert unchanged == []
    assert [event.event_type for event in disappeared] == [OBJECT_DISAPPEARED]


def test_general_presence_events_remain_available_when_default_policy_mutes_them() -> None:
    manager = SceneEventManager()
    policy = NarrationPolicy()
    person = AnalysisResult(
        object_type="person",
        stable_id="stable-2",
        state=None,
        confidence=0.9,
        attributes={"detection_confidence": 0.9},
        is_uncertain=False,
    )

    appeared = manager.update([person], 0.0)
    disappeared = manager.update([], 0.1)

    assert [event.event_type for event in appeared] == [OBJECT_APPEARED]
    assert [event.event_type for event in disappeared] == [OBJECT_DISAPPEARED]
    assert policy.narrate(appeared) == []
    assert policy.narrate(disappeared) == []


def test_known_state_change_emits_once_and_same_state_is_suppressed() -> None:
    manager = SceneEventManager(auto_presence=False)
    assert manager.update([result("GREEN")], 0.0) == []

    changed = manager.update([result("RED")], 0.1)

    assert len(changed) == 1
    assert changed[0].event_type == OBJECT_STATE_CHANGED
    assert changed[0].previous_state == "GREEN"
    assert changed[0].current_state == "RED"
    assert manager.update([result("RED")], 0.2) == []


def test_unknown_and_uncertain_results_neither_emit_nor_replace_known_state() -> None:
    manager = SceneEventManager(auto_presence=False)
    manager.update([result("GREEN")], 0.0)

    assert manager.update([result("UNKNOWN")], 0.1) == []
    assert manager.update([result("RED", is_uncertain=True)], 0.2) == []
    changed = manager.update([result("RED")], 0.3)

    assert len(changed) == 1
    assert changed[0].previous_state == "GREEN"
    assert changed[0].current_state == "RED"


def test_pipeline_mode_can_disable_all_automatic_events() -> None:
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)

    assert manager.update([result("GREEN")], 0.0) == []
    assert manager.update([result("RED")], 0.1) == []
    assert manager.update([], 0.2) == []


def test_legacy_events_are_translated_with_bare_stable_id_and_original_timestamp() -> None:
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)
    appeared = manager.update(
        [result("GREEN")],
        99.0,
        scene_events=[legacy_event("appeared", 1.25)],
    )
    changed = manager.update(
        [result("RED")],
        99.1,
        scene_events=[
            legacy_event(
                "signal_changed",
                2.733,
                previous_state=SignalState.GREEN,
                current_state=SignalState.RED,
            )
        ],
    )
    disappeared = manager.update(
        [],
        99.2,
        scene_events=[legacy_event("disappeared", 3.0)],
    )

    assert [event.event_type for event in appeared] == [OBJECT_APPEARED]
    assert appeared[0].stable_id == "stable-1"
    assert appeared[0].object_type == "pedestrian_signal"
    assert appeared[0].timestamp_s == 1.25
    assert [event.event_type for event in changed] == [OBJECT_STATE_CHANGED]
    assert changed[0].timestamp_s == 2.733
    assert changed[0].previous_state == "GREEN"
    assert changed[0].current_state == "RED"
    assert [event.event_type for event in disappeared] == [OBJECT_DISAPPEARED]


def test_legacy_and_derived_state_change_are_deduplicated_in_same_update() -> None:
    manager = SceneEventManager(auto_presence=False)
    manager.update([result("GREEN")], 0.0)

    events = manager.update(
        [result("RED")],
        2.733,
        scene_events=[
            legacy_event(
                "signal_changed",
                2.733,
                previous_state=SignalState.GREEN,
                current_state=SignalState.RED,
            )
        ],
    )

    assert [event.event_type for event in events] == [OBJECT_STATE_CHANGED]
    assert events[0].timestamp_s == 2.733


def test_repeated_legacy_event_is_suppressed() -> None:
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)
    event = legacy_event("appeared", 1.0)

    assert len(manager.update([], 1.0, scene_events=[event])) == 1
    assert manager.update([], 1.0, scene_events=[event]) == []


def test_stabilized_legacy_presence_does_not_inherit_analysis_uncertainty() -> None:
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)
    uncertain = result(
        None,
        object_type="bus",
        confidence=0.0,
        is_uncertain=True,
    )

    events = manager.update(
        [uncertain],
        1.0,
        scene_events=[legacy_event("appeared", 1.0)],
    )

    assert len(events) == 1
    assert events[0].confidence == 1.0
    assert events[0].is_uncertain is False


def test_low_detector_legacy_presence_waits_until_same_object_is_confident() -> None:
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)
    policy = NarrationPolicy(
        presence_narration_object_types=("kiosk",),
    )

    def kiosk(detection_confidence: float) -> AnalysisResult:
        return AnalysisResult(
            object_type="kiosk",
            stable_id="stable-1",
            state="UNKNOWN",
            confidence=0.0,
            attributes={"detection_confidence": detection_confidence},
            is_uncertain=True,
        )

    pending = manager.update(
        [kiosk(0.1)],
        1.0,
        scene_events=[legacy_event("appeared", 1.0)],
    )
    confirmed = manager.update([kiosk(0.9)], 1.1)

    assert pending == []
    assert [event.event_type for event in confirmed] == [OBJECT_APPEARED]
    assert confirmed[0].confidence == 0.9
    assert policy.narrate(confirmed) == ["키오스크가 감지되었습니다."]
    assert manager.update([kiosk(0.95)], 1.2) == []


def test_pending_low_confidence_presence_disappears_without_events() -> None:
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)
    low_confidence = AnalysisResult(
        object_type="kiosk",
        stable_id="stable-1",
        state="UNKNOWN",
        confidence=0.0,
        attributes={"detection_confidence": 0.1},
        is_uncertain=True,
    )

    assert (
        manager.update(
            [low_confidence],
            1.0,
            scene_events=[legacy_event("appeared", 1.0)],
        )
        == []
    )
    assert (
        manager.update(
            [],
            2.0,
            scene_events=[legacy_event("disappeared", 2.0)],
        )
        == []
    )


def test_unknown_legacy_transition_is_ignored() -> None:
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)
    event = legacy_event(
        "signal_changed",
        1.0,
        previous_state=SignalState.GREEN,
        current_state=SignalState.UNKNOWN,
    )

    assert manager.update([], 1.0, scene_events=[event]) == []


def test_low_detector_injected_signal_cannot_bypass_state_gate() -> None:
    manager = SceneEventManager(auto_presence=False)
    low_detector_signal = AnalysisResult(
        object_type="traffic_light",
        stable_id="stable-1",
        state="RED",
        confidence=0.95,
        attributes={
            "detection_confidence": 0.1,
            "minimum_detection_confidence": 0.2,
        },
        is_uncertain=False,
    )
    changed = legacy_event(
        "signal_changed",
        1.0,
        previous_state=SignalState.GREEN,
        current_state=SignalState.RED,
    )

    assert manager.update([low_detector_signal], 1.0, scene_events=[changed]) == []


def test_confirmed_bus_fields_create_approaching_and_text_events_once() -> None:
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)
    bus = AnalysisResult(
        object_type="bus",
        stable_id="stable-7",
        state="APPROACHING",
        confidence=0.91,
        attributes={"route_number": "3102"},
        is_uncertain=False,
    )

    events = manager.update([bus], 1.0)

    assert [event.event_type for event in events] == [
        OBJECT_APPROACHING,
        TEXT_CONFIRMED,
    ]
    assert manager.update([bus], 1.1) == []


def test_confirmed_kiosk_screen_change_is_deduplicated() -> None:
    manager = SceneEventManager(auto_presence=False)
    kiosk = AnalysisResult(
        object_type="kiosk",
        stable_id="stable-12",
        state="ORDER_TYPE_SELECTION",
        confidence=0.87,
        attributes={
            "visible_options": ["매장 식사", "포장"],
            "screen_changed": True,
        },
        is_uncertain=False,
    )

    assert [event.event_type for event in manager.update([kiosk], 1.0)] == [SCREEN_CHANGED]
    assert manager.update([kiosk], 1.1) == []


def test_kiosk_uses_real_fingerprint_and_emits_unknown_stage_screen() -> None:
    manager = SceneEventManager(auto_presence=False)

    def kiosk(fingerprint: str) -> AnalysisResult:
        return AnalysisResult(
            object_type="kiosk",
            stable_id="stable-12",
            state="UNKNOWN",
            confidence=0.0,
            attributes={
                "visible_text": ["가격이 변경되었습니다"],
                "visible_options": ["확인"],
                "screen_changed": True,
                "screen_is_confirmed": True,
                "screen_fingerprint": fingerprint,
                "screen_confidence": 0.87,
            },
            is_uncertain=True,
        )

    first = manager.update([kiosk("screen-a")], 1.0)
    second = manager.update([kiosk("screen-b")], 1.0)

    assert [event.event_type for event in first] == [SCREEN_CHANGED]
    assert [event.event_type for event in second] == [SCREEN_CHANGED]
    assert first[0].is_uncertain is False
    assert first[0].confidence == 0.87
    assert manager.update([kiosk("screen-b")], 1.1) == []


def test_distinct_text_events_at_same_pts_are_not_lost() -> None:
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)

    def sign(text: str) -> AnalysisResult:
        return AnalysisResult(
            object_type="sign",
            stable_id="stable-8",
            state=None,
            confidence=0.9,
            attributes={"text": text},
            is_uncertain=False,
        )

    first = manager.update([sign("출구")], 1.0)
    second = manager.update([sign("입구")], 1.0)

    assert [event.attributes["text"] for event in first] == ["출구"]
    assert [event.attributes["text"] for event in second] == ["입구"]


def test_low_confidence_approach_is_not_consumed_before_confidence_improves() -> None:
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)
    policy = NarrationPolicy()
    low_confidence_bus = AnalysisResult(
        object_type="bus",
        stable_id="stable-7",
        state="APPROACHING",
        confidence=0.95,
        attributes={
            "route_number": "3102",
            "route_confidence": 0.95,
            "motion_confidence": 0.4,
        },
        is_uncertain=False,
    )

    low_confidence_events = manager.update([low_confidence_bus], 1.0)

    assert [event.event_type for event in low_confidence_events] == [TEXT_CONFIRMED]
    assert policy.narrate(low_confidence_events) == ["3102번 버스입니다."]

    high_confidence_bus = AnalysisResult(
        object_type="bus",
        stable_id="stable-7",
        state="APPROACHING",
        confidence=0.95,
        attributes={
            "route_number": "3102",
            "route_confidence": 0.95,
            "motion_confidence": 0.9,
        },
        is_uncertain=False,
    )
    high_confidence_events = manager.update([high_confidence_bus], 1.1)

    assert [event.event_type for event in high_confidence_events] == [OBJECT_APPROACHING]
    assert high_confidence_events[0].confidence == 0.9
    assert policy.narrate(high_confidence_events) == ["3102번 버스가 들어오고 있습니다."]


def test_custom_narration_threshold_can_be_shared_with_approach_events() -> None:
    policy = NarrationPolicy(minimum_confidence=0.8)
    manager = SceneEventManager(
        auto_presence=False,
        derive_state_changes=False,
        minimum_approach_confidence=policy.minimum_confidence,
    )

    def bus(motion_confidence: float) -> AnalysisResult:
        return AnalysisResult(
            object_type="bus",
            stable_id="stable-7",
            state="APPROACHING",
            confidence=motion_confidence,
            attributes={"motion_confidence": motion_confidence},
            is_uncertain=False,
        )

    assert manager.update([bus(0.6)], 1.0) == []
    confirmed = manager.update([bus(0.9)], 1.1)

    assert [event.event_type for event in confirmed] == [OBJECT_APPROACHING]
    assert policy.narrate(confirmed) == ["버스가 접근하고 있습니다."]


def test_low_confidence_opposite_motion_does_not_restart_approach_episode() -> None:
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)

    def bus(state: str, confidence: float) -> AnalysisResult:
        return AnalysisResult(
            object_type="bus",
            stable_id="stable-7",
            state=state,
            confidence=confidence,
            attributes={"motion_confidence": confidence},
            is_uncertain=False,
        )

    assert [event.event_type for event in manager.update([bus("APPROACHING", 0.9)], 1.0)] == [
        OBJECT_APPROACHING
    ]
    assert manager.update([bus("RECEDING", 0.4)], 1.1) == []
    assert manager.update([bus("APPROACHING", 0.9)], 1.2) == []

    assert manager.update([bus("RECEDING", 0.9)], 1.3) == []
    restarted = manager.update([bus("APPROACHING", 0.9)], 1.4)
    assert [event.event_type for event in restarted] == [OBJECT_APPROACHING]


def test_custom_threshold_retries_same_text_after_confidence_improves() -> None:
    policy = NarrationPolicy(minimum_confidence=0.8)
    manager = SceneEventManager(
        auto_presence=False,
        derive_state_changes=False,
        minimum_domain_confidence=policy.minimum_confidence,
    )

    def sign(confidence: float) -> AnalysisResult:
        return AnalysisResult(
            object_type="sign",
            stable_id="stable-8",
            state=None,
            confidence=confidence,
            attributes={"text": "출구"},
            is_uncertain=False,
        )

    assert manager.update([sign(0.6)], 1.0) == []
    confirmed = manager.update([sign(0.9)], 1.1)

    assert [event.event_type for event in confirmed] == [TEXT_CONFIRMED]
    assert policy.narrate(confirmed) == ["표지판에 출구라고 표시되어 있습니다."]


def test_bus_route_event_combines_detector_and_ocr_confidence() -> None:
    policy = NarrationPolicy(minimum_confidence=0.8)
    manager = SceneEventManager(
        auto_presence=False,
        derive_state_changes=False,
        minimum_domain_confidence=policy.minimum_confidence,
    )

    def bus(detection_confidence: float) -> AnalysisResult:
        return AnalysisResult(
            object_type="bus",
            stable_id="stable-7",
            state="UNKNOWN",
            confidence=0.95,
            attributes={
                "detection_confidence": detection_confidence,
                "route_number": "3102",
                "route_confidence": 0.95,
            },
            is_uncertain=False,
        )

    assert manager.update([bus(0.5)], 1.0) == []
    confirmed = manager.update([bus(0.9)], 1.1)

    assert [event.event_type for event in confirmed] == [TEXT_CONFIRMED]
    assert confirmed[0].confidence == 0.9
    assert policy.narrate(confirmed) == ["3102번 버스입니다."]


def test_bus_state_uses_motion_not_route_confidence() -> None:
    manager = SceneEventManager(auto_presence=False, minimum_domain_confidence=0.8)

    def bus(state: str, motion_confidence: float) -> AnalysisResult:
        return AnalysisResult(
            object_type="bus",
            stable_id="stable-7",
            state=state,
            confidence=0.95,
            attributes={
                "detection_confidence": 0.95,
                "motion_confidence": motion_confidence,
                "route_confidence": 0.95,
            },
            is_uncertain=False,
        )

    assert manager.update([bus("STOPPED", 0.9)], 1.0) == []
    assert manager.update([bus("RECEDING", 0.4)], 1.1) == []
    confirmed = manager.update([bus("RECEDING", 0.9)], 1.2)

    assert [event.event_type for event in confirmed] == [OBJECT_STATE_CHANGED]
    assert confirmed[0].previous_state == "STOPPED"
    assert confirmed[0].current_state == "RECEDING"


def test_custom_threshold_retries_confirmed_kiosk_screen_without_change_pulse() -> None:
    policy = NarrationPolicy(minimum_confidence=0.8)
    manager = SceneEventManager(
        auto_presence=False,
        minimum_domain_confidence=policy.minimum_confidence,
    )

    def kiosk(screen_confidence: float, *, changed: bool) -> AnalysisResult:
        return AnalysisResult(
            object_type="kiosk",
            stable_id="stable-12",
            state="ORDER_TYPE_SELECTION",
            confidence=screen_confidence,
            attributes={
                "visible_options": ["매장 식사", "포장"],
                "screen_changed": changed,
                "screen_is_confirmed": True,
                "screen_fingerprint": "screen-a",
                "screen_confidence": screen_confidence,
            },
            is_uncertain=False,
        )

    assert manager.update([kiosk(0.6, changed=True)], 1.0) == []
    confirmed = manager.update([kiosk(0.9, changed=False)], 1.1)

    assert [event.event_type for event in confirmed] == [SCREEN_CHANGED]
    assert policy.narrate(confirmed) == ["매장 식사와 포장 중 하나를 선택하는 화면입니다."]


def test_custom_threshold_retries_same_description_after_confidence_improves() -> None:
    policy = NarrationPolicy(minimum_confidence=0.8)
    manager = SceneEventManager(
        auto_presence=False,
        derive_state_changes=False,
        minimum_domain_confidence=policy.minimum_confidence,
    )

    def described(confidence: float) -> AnalysisResult:
        return AnalysisResult(
            object_type="vending_machine",
            stable_id="stable-20",
            state="DESCRIBED",
            confidence=confidence,
            attributes={"description": "빨간 자판기가 보입니다."},
            is_uncertain=False,
        )

    assert manager.update([described(0.6)], 1.0) == []
    confirmed = manager.update([described(0.9)], 1.1)

    assert [event.event_type for event in confirmed] == [DESCRIPTION_CONFIRMED]
    assert policy.narrate(confirmed) == ["빨간 자판기가 보입니다."]


def test_custom_narration_threshold_records_low_confidence_signal_without_speaking() -> None:
    policy = NarrationPolicy(minimum_confidence=0.8)
    manager = SceneEventManager(
        auto_presence=False,
        minimum_domain_confidence=policy.minimum_confidence,
    )

    assert manager.update([result("GREEN", confidence=0.9)], 1.0) == []
    recorded = manager.update([result("RED", confidence=0.6)], 1.1)

    assert [event.event_type for event in recorded] == [OBJECT_STATE_CHANGED]
    assert recorded[0].previous_state == "GREEN"
    assert recorded[0].current_state == "RED"
    assert recorded[0].confidence == 0.6
    assert policy.narrate(recorded) == []
    assert manager.update([result("RED", confidence=0.9)], 1.2) == []


def test_explicit_reset_releases_domain_deduplication_for_retired_id() -> None:
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)
    sign = AnalysisResult(
        object_type="sign",
        stable_id="stable-8",
        state=None,
        confidence=0.9,
        attributes={"text": "출구"},
        is_uncertain=False,
    )

    assert [event.event_type for event in manager.update([sign], 1.0)] == [TEXT_CONFIRMED]
    manager.reset("stable-8")

    assert [event.event_type for event in manager.update([sign], 1.0)] == [TEXT_CONFIRMED]


def test_confirmed_generic_description_emits_only_when_text_changes() -> None:
    manager = SceneEventManager(auto_presence=False, derive_state_changes=False)
    first = AnalysisResult(
        object_type="vending_machine",
        stable_id="stable-20",
        state="DESCRIBED",
        confidence=0.7,
        attributes={"description": "빨간 자판기가 보입니다."},
        is_uncertain=False,
    )
    changed = AnalysisResult(
        object_type="vending_machine",
        stable_id="stable-20",
        state="DESCRIBED",
        confidence=0.7,
        attributes={"description": "자판기 앞에 결제 패널이 보입니다."},
        is_uncertain=False,
    )

    assert [event.event_type for event in manager.update([first], 1.0)] == [DESCRIPTION_CONFIRMED]
    assert manager.update([first], 1.1) == []
    assert [event.event_type for event in manager.update([changed], 2.0)] == [DESCRIPTION_CONFIRMED]
