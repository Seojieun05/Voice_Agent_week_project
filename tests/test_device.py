import pytest

from vision_agent.pipeline import resolve_device


@pytest.mark.parametrize(
    ("cuda_available", "expected"),
    [
        (False, "cpu"),
        (True, "0"),
    ],
)
def test_automatic_device_selection(cuda_available: bool, expected: str) -> None:
    assert resolve_device(None, cuda_available=cuda_available) == expected


def test_explicit_cpu_is_allowed_without_cuda() -> None:
    assert resolve_device("cpu", cuda_available=False) == "cpu"


@pytest.mark.parametrize("requested", ["0", "cuda:0"])
def test_explicit_cuda_device_fails_when_cuda_is_unavailable(requested: str) -> None:
    with pytest.raises(RuntimeError, match=r"CUDA.*--device cpu"):
        resolve_device(requested, cuda_available=False)
