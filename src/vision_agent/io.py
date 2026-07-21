from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = path.open("w", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
