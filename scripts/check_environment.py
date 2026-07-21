#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys


def main() -> int:
    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    try:
        import cv2

        info["opencv"] = cv2.__version__
    except ImportError:
        info["opencv"] = None

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
        info["cuda_available"] = False

    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = None

    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0 if info["ultralytics"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
