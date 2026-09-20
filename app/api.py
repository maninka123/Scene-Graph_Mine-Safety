from __future__ import annotations

from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.demo_service import ROOT, load_demo, run_interactive
from app.qwen_service import model_status, reason_scene
from app.upload_service import create_job, perception_status, public_job


class AnalysisRequest(BaseModel):
    nodes: list[dict[str, Any]]
    edge_distance_m: float = Field(default=8.0, ge=0.25, le=8.0)


class ReasoningRequest(BaseModel):
    graph: dict[str, Any]
    alerts: list[dict[str, Any]] = Field(default_factory=list)


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


@app.get("/api/perception/status")
def segmentation_status() -> dict[str, Any]:
    return perception_status()


@app.post("/api/point-clouds", status_code=202)
async def upload_point_cloud(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
    try:
        content = await file.read()
        return create_job(file.filename or "point-cloud.pcd", content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/point-clouds/{job_id}")
def point_cloud_job(job_id: str) -> dict[str, Any]:
    try:
        return public_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Upload job not found") from exc


@app.get("/api/model/status")
def qwen_status() -> dict[str, Any]:
    return model_status()


@app.post("/api/reason")
def reason(request: ReasoningRequest) -> dict[str, Any]:
    try:
        return reason_scene(request.graph, request.alerts)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Qwen reasoning failed: {exc}") from exc


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
