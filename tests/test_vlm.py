from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vision_agent.vlm import TransformersVisionLanguageModel, VisionLanguageResult


class _FakePipeline:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.output


def test_transformers_adapter_uses_image_text_to_text_messages() -> None:
    fake = _FakePipeline([{"generated_text": "  빨간 자판기가 보입니다.  "}])
    factory_calls: list[dict[str, object]] = []

    def factory(**kwargs: Any) -> _FakePipeline:
        factory_calls.append(kwargs)
        return fake

    model = TransformersVisionLanguageModel(
        "test-model",
        pipeline_factory=factory,
        assumed_confidence=0.7,
    )
    result = model.describe(np.zeros((8, 8, 3), dtype=np.uint8), "설명")

    assert factory_calls == [{"task": "image-text-to-text", "model": "test-model"}]
    assert result.description == "빨간 자판기가 보입니다."
    assert result.confidence == pytest.approx(0.7)
    assert fake.calls[0]["max_new_tokens"] == 64

    second = model.describe(np.zeros((8, 8, 3), dtype=np.uint8), "다시 설명")
    assert second.description == "빨간 자판기가 보입니다."
    assert len(factory_calls) == 1


def test_transformers_adapter_parses_chat_style_generated_text() -> None:
    fake = _FakePipeline(
        [
            {
                "generated_text": [
                    {"role": "user", "content": [{"type": "text", "text": "질문"}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "파란 기계가 보입니다."}],
                    },
                ]
            }
        ]
    )
    model = TransformersVisionLanguageModel(
        "test-model",
        pipeline_factory=lambda **_: fake,
    )

    result = model.describe(np.zeros((8, 8, 3), dtype=np.uint8), "설명")

    assert result.description == "파란 기계가 보입니다."


def test_transformers_adapter_blocks_unapproved_remote_model_download(tmp_path: Path) -> None:
    missing_path = tmp_path / "not-local"
    model = TransformersVisionLanguageModel(str(missing_path), allow_download=False)

    result = model.describe(np.zeros((8, 8, 3), dtype=np.uint8), "설명")

    assert result.description is None
    assert "VLM 다운로드가 비활성화" in (result.error or "")


def test_transformers_adapter_reports_missing_optional_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    monkeypatch.setitem(sys.modules, "transformers", None)
    model = TransformersVisionLanguageModel(str(local_model))

    result = model.describe(np.zeros((8, 8, 3), dtype=np.uint8), "설명")

    assert result.description is None
    assert "Transformers VLM이 설치되지 않았습니다" in (result.error or "")


def test_transformers_adapter_returns_error_for_invalid_image() -> None:
    model = TransformersVisionLanguageModel("test-model", pipeline_factory=lambda **_: None)

    result = model.describe(np.zeros((8, 8), dtype=np.uint8), "설명")

    assert result.description is None
    assert result.error == "invalid_image"


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_vlm_result_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError):
        VisionLanguageResult("설명", confidence=confidence)
