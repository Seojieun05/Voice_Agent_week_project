#!/bin/bash
# Convenience script to run the server with .env loaded

set -a
if [ -f .env ]; then
    source .env
fi
set +a

.venv/bin/python -m uvicorn vision_agent.server:app --host 0.0.0.0 --port 8000 --workers 1 "$@"
