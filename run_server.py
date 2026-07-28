#!/usr/bin/env python
"""Convenience script to run the server with .env loaded."""

import os
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("dotenv not installed. Run: pip install -e '.[dev]'", file=sys.stderr)
    sys.exit(1)

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment from {env_path}")

args = [sys.executable, "-m", "uvicorn", "vision_agent.server:app",
        "--host", "0.0.0.0", "--port", "8000", "--workers", "1"] + sys.argv[1:]
sys.exit(subprocess.call(args))
