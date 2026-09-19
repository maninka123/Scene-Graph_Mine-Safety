#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/build_demo_bundle.py
(cd web && npm install && npm run build)
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
