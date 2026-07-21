from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vision_agent.ocr import (
    OcrEngine,
    OcrLine,
    OcrResult,
    RapidOcrEngine,
    UnavailableOcrEngine,
)


def test_ocr_result_joins_lines_and_weights_confidence_by_character_count() -> None:
    result = OcrResult(
        lines=(
            OcrLine("가", 0.6),
            OcrLine("나나다", 0.9),
            OcrLine("  ", 1.0),
        ),
        engine_name="test",
    )

    assert result.text == "가\n나나다"
    assert result.confidence == pytest.approx((0.6 + 0.9 * 3) / 4)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("inf"), float("nan")])
def test_ocr_line_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError):
        OcrLine("text", confidence)


@pytest.mark.parametrize(
    "bbox",
    [
        (0, 0, 0, 10),
        (0, 0, 10, 0),
        (0, 0, 1.5, 10),
    ],
)
def test_ocr_line_rejects_invalid_bbox(bbox: object) -> None:
    with pytest.raises(ValueError):
        OcrLine("text", 0.9, bbox=bbox)  # type: ignore[arg-type]


def test_ocr_engine_protocol_supports_structural_injection() -> None:
    assert isinstance(UnavailableOcrEngine(), OcrEngine)


def test_unavailable_engine_returns_explicit_result() -> None:
    result = UnavailableOcrEngine("disabled_for_test").recognize(
        np.zeros((32, 64, 3), dtype=np.uint8)
    )

    assert result.lines == ()
    assert result.is_available is False
    assert result.error == "disabled_for_test"


def test_rapidocr_is_loaded_lazily_and_parses_official_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    output = SimpleNamespace(
        boxes=[
            np.array([[1.2, 2.8], [10.1, 2.1], [10.6, 9.4], [1.0, 9.9]]),
            None,
        ],
        txts=["출구", "EXIT"],
        scores=[np.float32(0.95), 0.75],
    )

    class FakeRapidOCR:
        def __init__(self, **options: object) -> None:
            calls.append(("init", options))

        def __call__(self, image: np.ndarray) -> object:
            calls.append(("call", image.shape))
            return output

    def fake_import(name: str) -> object:
        calls.append(("import", name))
        return SimpleNamespace(RapidOCR=FakeRapidOCR)

    monkeypatch.setattr("vision_agent.ocr.importlib.import_module", fake_import)
    engine = RapidOcrEngine(language="default", use_det=True)
    assert calls == []

    first = engine.recognize(np.zeros((32, 64, 3), dtype=np.uint8))
    second = engine.recognize(np.zeros((32, 64, 3), dtype=np.uint8))

    assert first.text == "출구\nEXIT"
    assert first.lines[0].bbox == (1, 2, 11, 10)
    assert first.lines[1].bbox is None
    assert first.confidence == pytest.approx((0.95 * 2 + 0.75 * 4) / 6)
    assert calls.count(("import", "rapidocr")) == 1
    assert calls.count(("init", {"params": {"use_det": True}})) == 1
    assert sum(call[0] == "call" for call in calls if isinstance(call, tuple)) == 2
    assert second.text == first.text


def test_rapidocr_korean_configuration_uses_v5_mobile_onnx_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_options: dict[str, object] = {}

    class FakeRapidOCR:
        def __init__(self, **options: object) -> None:
            captured_options.update(options)

        def __call__(self, image: np.ndarray) -> object:
            return SimpleNamespace(boxes=[], txts=[], scores=[])

    fake_module = SimpleNamespace(
        RapidOCR=FakeRapidOCR,
        LangRec=SimpleNamespace(KOREAN="korean"),
        OCRVersion=SimpleNamespace(PPOCRV5="v5"),
        ModelType=SimpleNamespace(MOBILE="mobile"),
        EngineType=SimpleNamespace(ONNXRUNTIME="onnxruntime"),
    )
    monkeypatch.setattr(
        "vision_agent.ocr.importlib.import_module",
        lambda name: fake_module,
    )

    result = RapidOcrEngine(language="korean", allow_download=True).recognize(
        np.zeros((32, 64, 3), dtype=np.uint8)
    )

    assert result.is_available is True
    assert captured_options == {
        "params": {
            "Rec.lang_type": "korean",
            "Rec.ocr_version": "v5",
            "Rec.model_type": "mobile",
            "Rec.engine_type": "onnxruntime",
        }
    }


def test_explicit_english_language_is_mapped_to_rapidocr_enum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_options: dict[str, object] = {}

    class FakeRapidOCR:
        def __init__(self, **options: object) -> None:
            captured_options.update(options)

        def __call__(self, image: np.ndarray) -> object:
            return SimpleNamespace(boxes=[], txts=[], scores=[])

    fake_module = SimpleNamespace(
        RapidOCR=FakeRapidOCR,
        LangRec=SimpleNamespace(EN="en"),
        OCRVersion=SimpleNamespace(PPOCRV5="v5"),
        ModelType=SimpleNamespace(MOBILE="mobile"),
        EngineType=SimpleNamespace(ONNXRUNTIME="onnxruntime"),
    )
    monkeypatch.setattr(
        "vision_agent.ocr.importlib.import_module",
        lambda name: fake_module,
    )

    result = RapidOcrEngine(language="english", allow_download=True).recognize(
        np.zeros((32, 64, 3), dtype=np.uint8)
    )

    assert result.is_available is True
    assert captured_options["params"]["Rec.lang_type"] == "en"


def test_unknown_ocr_language_is_explicitly_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vision_agent.ocr.importlib.import_module",
        lambda name: SimpleNamespace(RapidOCR=object),
    )

    result = RapidOcrEngine(language="made-up", allow_download=True).recognize(
        np.zeros((32, 64, 3), dtype=np.uint8)
    )

    assert result.is_available is False
    assert "unsupported OCR language" in (result.error or "")


def test_korean_backend_refuses_implicit_model_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instantiated = False

    class FakeRapidOCR:
        def __init__(self, **options: object) -> None:
            nonlocal instantiated
            instantiated = True

    monkeypatch.setattr(
        "vision_agent.ocr.importlib.import_module",
        lambda name: SimpleNamespace(RapidOCR=FakeRapidOCR),
    )

    result = RapidOcrEngine().recognize(np.zeros((32, 64, 3), dtype=np.uint8))

    assert result.is_available is False
    assert "Rec.model_path" in (result.error or "")
    assert instantiated is False


def test_local_korean_model_path_allows_offline_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_options: dict[str, object] = {}

    class FakeRapidOCR:
        def __init__(self, **options: object) -> None:
            captured_options.update(options)

        def __call__(self, image: np.ndarray) -> object:
            return SimpleNamespace(boxes=None, txts=None, scores=None)

    fake_module = SimpleNamespace(
        RapidOCR=FakeRapidOCR,
        LangRec=SimpleNamespace(KOREAN="ko"),
        OCRVersion=SimpleNamespace(PPOCRV5="v5"),
        ModelType=SimpleNamespace(MOBILE="mobile"),
        EngineType=SimpleNamespace(ONNXRUNTIME="onnxruntime"),
    )
    monkeypatch.setattr(
        "vision_agent.ocr.importlib.import_module",
        lambda name: fake_module,
    )

    result = RapidOcrEngine(rec_model_path="/models/korean.onnx").recognize(
        np.zeros((32, 64, 3), dtype=np.uint8)
    )

    assert result.is_available is True
    assert result.lines == ()
    assert captured_options["params"]["Rec.model_path"] == "/models/korean.onnx"


def test_invalid_image_does_not_import_optional_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str) -> object:
        raise AssertionError("backend import must not happen")

    monkeypatch.setattr("vision_agent.ocr.importlib.import_module", fail_import)

    result = RapidOcrEngine().recognize(np.zeros((0, 0, 3), dtype=np.uint8))

    assert result.error == "invalid_image"
    assert result.is_available is True


def test_missing_backend_is_explicit_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def missing_import(name: str) -> object:
        nonlocal calls
        calls += 1
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("vision_agent.ocr.importlib.import_module", missing_import)
    engine = RapidOcrEngine(language="default")

    first = engine.recognize(np.zeros((32, 64, 3), dtype=np.uint8))
    second = engine.recognize(np.zeros((32, 64, 3), dtype=np.uint8))

    assert first.is_available is False
    assert second.is_available is False
    assert calls == 1


def test_backend_runtime_error_is_returned_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenRapidOCR:
        def __call__(self, image: np.ndarray) -> object:
            raise RuntimeError("bad model")

    monkeypatch.setattr(
        "vision_agent.ocr.importlib.import_module",
        lambda name: SimpleNamespace(RapidOCR=lambda **kwargs: BrokenRapidOCR()),
    )

    result = RapidOcrEngine(language="default").recognize(np.zeros((32, 64, 3), dtype=np.uint8))

    assert result.is_available is True
    assert result.lines == ()
    assert result.error == "RuntimeError: bad model"
