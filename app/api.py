from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.demo_service import ROOT, load_demo, run_interactive


class AnalysisRequest(BaseModel):
    nodes: list[dict[str, Any]]
    edge_distance_m: float = Field(default=2.5, ge=0.25, le=8.0)


app = FastAPI(
    title="MineGraph Studio API",
    version="2.0.0",
    description="Local API for the paper-aligned perception-to-reasoning demonstrator.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "MineGraph Studio API"}


@app.get("/api/demo")
def demo() -> dict[str, Any]:
    try:
        return load_demo()
    except Exception as exc:  # pragma: no cover - surfaced to the browser with context
        raise HTTPException(status_code=500, detail=f"Could not load demo assets: {exc}") from exc


@app.post("/api/analyse")
def analyse(request: AnalysisRequest) -> dict[str, Any]:
    try:
        return run_interactive(request.nodes, request.edge_distance_m)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not analyse scene: {exc}") from exc


DIST = ROOT / "web" / "dist"
if DIST.exists():
    assets = DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        candidate = (DIST / path).resolve()
        if path and candidate.is_file() and DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(DIST / "index.html")
