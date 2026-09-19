$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
python scripts/build_demo_bundle.py
Push-Location web
npm install
npm run build
Pop-Location
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
