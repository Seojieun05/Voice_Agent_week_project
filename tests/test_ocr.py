from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
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


class _FakeDownloadFile:
    @classmethod
    def run(cls, *args: object, **kwargs: object) -> None:
        del cls, args, kwargs
        raise AssertionError("unexpected RapidOCR download")


def _offline_importer(
    rapidocr_module: object,
    download_class: object = _FakeDownloadFile,
) -> Callable[[str], object]:
    download_module = SimpleNamespace(DownloadFile=download_class)

    def fake_import(name: str) -> object:
        if name == "rapidocr.utils.download_file":
            return download_module
        return rapidocr_module

    return fake_import


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
    engine = RapidOcrEngine(language="default", allow_download=True, use_det=True)
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
        _offline_importer(fake_module),
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
        _offline_importer(fake_module),
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
    tmp_path: Path,
) -> None:
    instantiated = False
    rapidocr_root = tmp_path / "rapidocr"
    models_dir = rapidocr_root / "models"
    models_dir.mkdir(parents=True)
    for filename in (
        "PP-OCRv6_det_small.onnx",
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    ):
        (models_dir / filename).write_bytes(b"cached")

    class FakeRapidOCR:
        def __init__(self, **options: object) -> None:
            nonlocal instantiated
            instantiated = True

    fake_module = SimpleNamespace(
        __file__=str(rapidocr_root / "__init__.py"),
        RapidOCR=FakeRapidOCR,
        LangRec=SimpleNamespace(KOREAN="ko"),
        OCRVersion=SimpleNamespace(PPOCRV5="v5"),
        ModelType=SimpleNamespace(MOBILE="mobile"),
        EngineType=SimpleNamespace(ONNXRUNTIME="onnxruntime"),
    )
    monkeypatch.setattr(
        "vision_agent.ocr.importlib.import_module",
        _offline_importer(fake_module),
    )

    result = RapidOcrEngine().recognize(np.zeros((32, 64, 3), dtype=np.uint8))

    assert result.is_available is False
    assert "Rec.model_path" in (result.error or "")
    assert instantiated is False


def test_missing_default_detector_model_stops_before_constructor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    instantiated = False
    rapidocr_root = tmp_path / "rapidocr"
    models_dir = rapidocr_root / "models"
    models_dir.mkdir(parents=True)
    for filename in (
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "PP-OCRv6_rec_small.onnx",
    ):
        (models_dir / filename).write_bytes(b"cached")

    class FakeRapidOCR:
        def __init__(self, **options: object) -> None:
            nonlocal instantiated
            instantiated = True

    fake_module = SimpleNamespace(
        __file__=str(rapidocr_root / "__init__.py"),
        RapidOCR=FakeRapidOCR,
    )
    monkeypatch.setattr(
        "vision_agent.ocr.importlib.import_module",
        _offline_importer(fake_module),
    )

    result = RapidOcrEngine(language="default").recognize(np.zeros((32, 64, 3), dtype=np.uint8))

    assert result.is_available is False
    assert "Det.model_path has no local model file" in (result.error or "")
    assert instantiated is False


def test_cached_default_models_are_pinned_before_offline_constructor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_params: dict[str, object] = {}
    rapidocr_root = tmp_path / "rapidocr"
    models_dir = rapidocr_root / "models"
    models_dir.mkdir(parents=True)
    filenames = (
        "PP-OCRv6_det_small.onnx",
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "PP-OCRv6_rec_small.onnx",
    )
    for filename in filenames:
        (models_dir / filename).write_bytes(b"cached")

    class FakeRapidOCR:
        def __init__(self, **options: object) -> None:
            captured_params.update(options["params"])  # type: ignore[arg-type]

        def __call__(self, image: np.ndarray) -> object:
            return SimpleNamespace(boxes=None, txts=None, scores=None)

    fake_module = SimpleNamespace(
        __file__=str(rapidocr_root / "__init__.py"),
        RapidOCR=FakeRapidOCR,
    )
    monkeypatch.setattr(
        "vision_agent.ocr.importlib.import_module",
        _offline_importer(fake_module),
    )
    engine = RapidOcrEngine(language="default")

    assert captured_params == {}

    result = engine.recognize(np.zeros((32, 64, 3), dtype=np.uint8))

    assert result.is_available is True
    assert captured_params == {
        f"{component}.model_path": str(models_dir / filename)
        for component, filename in zip(("Det", "Cls", "Rec"), filenames, strict=True)
    }


def test_offline_constructor_blocks_internal_rapidocr_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    instantiated = False
    original_download_calls = 0
    rapidocr_root = tmp_path / "rapidocr"
    models_dir = rapidocr_root / "models"
    models_dir.mkdir(parents=True)
    for filename in (
        "PP-OCRv6_det_small.onnx",
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "PP-OCRv6_rec_small.onnx",
    ):
        (models_dir / filename).write_bytes(b"cached")

    class AttemptingDownloadFile:
        @classmethod
        def run(cls, *args: object, **kwargs: object) -> None:
            nonlocal original_download_calls
            del cls, args, kwargs
            original_download_calls += 1

    class FakeRapidOCR:
        def __init__(self, **options: object) -> None:
            nonlocal instantiated
            del options
            instantiated = True
            AttemptingDownloadFile.run(object())

    fake_module = SimpleNamespace(
        __file__=str(rapidocr_root / "__init__.py"),
        RapidOCR=FakeRapidOCR,
    )
    monkeypatch.setattr(
        "vision_agent.ocr.importlib.import_module",
        _offline_importer(fake_module, AttemptingDownloadFile),
    )

    result = RapidOcrEngine(language="default").recognize(np.zeros((32, 64, 3), dtype=np.uint8))

    assert instantiated is True
    assert original_download_calls == 0
    assert result.is_available is False
    assert "RapidOCR download blocked" in (result.error or "")


def test_allow_download_keeps_rapidocr_downloader_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download_calls = 0

    class CountingDownloadFile:
        @classmethod
        def run(cls, *args: object, **kwargs: object) -> None:
            nonlocal download_calls
            del cls, args, kwargs
            download_calls += 1

    class FakeRapidOCR:
        def __init__(self, **options: object) -> None:
            del options
            CountingDownloadFile.run(object())

        def __call__(self, image: np.ndarray) -> object:
            del image
            return SimpleNamespace(boxes=None, txts=None, scores=None)

    fake_module = SimpleNamespace(RapidOCR=FakeRapidOCR)
    monkeypatch.setattr(
        "vision_agent.ocr.importlib.import_module",
        _offline_importer(fake_module, CountingDownloadFile),
    )

    result = RapidOcrEngine(language="default", allow_download=True).recognize(
        np.zeros((32, 64, 3), dtype=np.uint8)
    )

    assert result.is_available is True
    assert download_calls == 1


def test_installed_model_registry_resolves_offline_filenames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_params: dict[str, object] = {}
    models_dir = tmp_path / "custom-models"
    models_dir.mkdir()
    filenames = {
        "det": "custom-det.onnx",
        "cls": "custom-cls.onnx",
        "rec": "custom-rec.onnx",
    }
    for filename in filenames.values():
        (models_dir / filename).write_bytes(b"cached")

    config = SimpleNamespace(
        Global=SimpleNamespace(model_root_dir=models_dir),
        Det=SimpleNamespace(
            engine_type="engine",
            ocr_version="version",
            task_type="det",
            lang_type="language",
            model_type="type",
        ),
        Cls=SimpleNamespace(
            engine_type="engine",
            ocr_version="version",
            task_type="cls",
            lang_type="language",
            model_type="type",
        ),
        Rec=SimpleNamespace(
            engine_type="engine",
            ocr_version="version",
            task_type="rec",
            lang_type="language",
            model_type="type",
        ),
    )

    class FakeRapidOCR:
        def _load_config(self, config_path: object, params: object) -> object:
            del config_path, params
            return config

        def __init__(self, **options: object) -> None:
            captured_params.update(options["params"])  # type: ignore[arg-type]

        def __call__(self, image: np.ndarray) -> object:
            del image
            return SimpleNamespace(boxes=None, txts=None, scores=None)

    class FakeFileInfo:
        def __init__(self, **values: object) -> None:
            self.task_type = values["task_type"]

    class FakeInferSession:
        @classmethod
        def get_model_url(cls, file_info: FakeFileInfo) -> dict[str, str]:
            del cls
            return {"model_dir": f"https://models.invalid/{filenames[file_info.task_type]}"}

    fake_module = SimpleNamespace(RapidOCR=FakeRapidOCR)
    download_module = SimpleNamespace(DownloadFile=_FakeDownloadFile)
    base_module = SimpleNamespace(FileInfo=FakeFileInfo, InferSession=FakeInferSession)

    def fake_import(name: str) -> object:
        if name == "rapidocr.utils.download_file":
            return download_module
        if name == "rapidocr.inference_engine.base":
            return base_module
        return fake_module

    monkeypatch.setattr("vision_agent.ocr.importlib.import_module", fake_import)

    result = RapidOcrEngine(language="default").recognize(np.zeros((32, 64, 3), dtype=np.uint8))

    assert result.is_available is True
    assert captured_params == {
        "Det.model_path": str(models_dir / filenames["det"]),
        "Cls.model_path": str(models_dir / filenames["cls"]),
        "Rec.model_path": str(models_dir / filenames["rec"]),
    }


def test_installed_rapidocr_default_models_do_not_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("rapidocr")
    download_module = importlib.import_module("rapidocr.utils.download_file")
    download_class = download_module.DownloadFile
    download_calls = 0

    def fail_download(cls: object, *args: object, **kwargs: object) -> None:
        nonlocal download_calls
        del cls, args, kwargs
        download_calls += 1
        raise AssertionError("real RapidOCR downloader was called")

    monkeypatch.setattr(download_class, "run", classmethod(fail_download))

    result = RapidOcrEngine(language="default", allow_download=False).recognize(
        np.zeros((32, 64, 3), dtype=np.uint8)
    )

    assert result.is_available is True
    assert result.error is None
    assert download_calls == 0


def test_missing_explicit_model_path_stops_before_constructor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    instantiated = False

    class FakeRapidOCR:
        def __init__(self, **options: object) -> None:
            nonlocal instantiated
            instantiated = True

    fake_module = SimpleNamespace(
        RapidOCR=FakeRapidOCR,
        LangRec=SimpleNamespace(KOREAN="ko"),
        OCRVersion=SimpleNamespace(PPOCRV5="v5"),
        ModelType=SimpleNamespace(MOBILE="mobile"),
        EngineType=SimpleNamespace(ONNXRUNTIME="onnxruntime"),
    )
    monkeypatch.setattr(
        "vision_agent.ocr.importlib.import_module",
        _offline_importer(fake_module),
    )

    result = RapidOcrEngine(rec_model_path=tmp_path / "missing.onnx").recognize(
        np.zeros((32, 64, 3), dtype=np.uint8)
    )

    assert result.is_available is False
    assert "Rec.model_path is not a local model file" in (result.error or "")
    assert instantiated is False


def test_local_korean_model_path_allows_offline_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_options: dict[str, object] = {}
    rapidocr_root = tmp_path / "rapidocr"
    models_dir = rapidocr_root / "models"
    models_dir.mkdir(parents=True)
    for filename in (
        "PP-OCRv6_det_small.onnx",
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    ):
        (models_dir / filename).write_bytes(b"cached")
    recognition_model = tmp_path / "korean.onnx"
    recognition_model.write_bytes(b"explicit")

    class FakeRapidOCR:
        def __init__(self, **options: object) -> None:
            captured_options.update(options)

        def __call__(self, image: np.ndarray) -> object:
            return SimpleNamespace(boxes=None, txts=None, scores=None)

    fake_module = SimpleNamespace(
        __file__=str(rapidocr_root / "__init__.py"),
        RapidOCR=FakeRapidOCR,
        LangRec=SimpleNamespace(KOREAN="ko"),
        OCRVersion=SimpleNamespace(PPOCRV5="v5"),
        ModelType=SimpleNamespace(MOBILE="mobile"),
        EngineType=SimpleNamespace(ONNXRUNTIME="onnxruntime"),
    )
    monkeypatch.setattr(
        "vision_agent.ocr.importlib.import_module",
        _offline_importer(fake_module),
    )

    result = RapidOcrEngine(rec_model_path=recognition_model).recognize(
        np.zeros((32, 64, 3), dtype=np.uint8)
    )

    assert result.is_available is True
    assert result.lines == ()
    assert captured_options["params"]["Rec.model_path"] == str(recognition_model)
    assert captured_options["params"]["Det.model_path"] == str(
        models_dir / "PP-OCRv6_det_small.onnx"
    )
    assert captured_options["params"]["Cls.model_path"] == str(
        models_dir / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
    )


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

    result = RapidOcrEngine(language="default", allow_download=True).recognize(
        np.zeros((32, 64, 3), dtype=np.uint8)
    )

    assert result.is_available is True
    assert result.lines == ()
    assert result.error == "RuntimeError: bad model"
