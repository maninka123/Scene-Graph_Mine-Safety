from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import streamlit as st

PANDAS_IMPORT_ERROR = None
try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - environment dependent
    pd = None
    PANDAS_IMPORT_ERROR = exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MinkUNET.minkunet.config import load_config

from AppUI.app_core.pipeline_runner import (
    AppRunError,
    create_run_dir,
    run_single_pipeline,
    run_temporal_pipeline,
)
from AppUI.app_core.model_catalog import (
    compatibility_message,
    discover_local_models,
    get_device_report,
)
from AppUI.app_core.plotting import (
    bar_class_counts,
    bar_movement_states,
    line_frame_metrics,
    safe_frame_metrics_df,
    scatter3d_semantic,
)
from AppUI.app_core.timestamp_utils import (
    frame_rows,
    list_pcd_files,
    parse_pcd_timestamp,
    range_and_gap_select,
)


DEFAULT_INFERENCE_CONFIG = "MinkUNET/configs/inference.yaml"
DEFAULT_CHECKPOINT = "MinkUNET/checkpoints/semantic_best.pt"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.0rem; padding-bottom: 1.5rem;}
        .app-card {
            border: 1px solid rgba(120, 130, 150, 0.35);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            background: linear-gradient(180deg, rgba(21,34,56,0.60), rgba(16,25,42,0.35));
        }
        .stAlert > div {
            border-radius: 10px;
        }
        .caption-soft {
            color: #9cb3c9;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_class_meta(config_path: str) -> Dict:
    cfg = load_config(config_path)
    return {
        "class_names": list(cfg["model"]["class_names"]),
        "class_colors": dict(cfg["model"]["class_colors"]),
    }


def _parse_json_map(raw: str, value_type: str) -> Optional[Dict]:
    text = raw.strip()
    if not text:
        return None
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    out = {}
    for key, value in parsed.items():
        if value_type == "float":
            out[str(key)] = float(value)
        else:
            out[str(key)] = int(value)
    return out


def _save_plot_html(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _preview_from_npz(npz_path: Path, class_meta: Dict, title: str, max_points: int = 20000):
    data = np.load(npz_path)
    fig = scatter3d_semantic(
        points=data["points"],
        labels=data["labels"],
        class_names=class_meta["class_names"],
        class_colors=class_meta["class_colors"],
        title=title,
        max_points=max_points,
    )
    return fig


def _txt(value) -> str:
    return "" if value is None else str(value).strip()


def _severity_from_item(item: Dict) -> str:
    level = _txt(item.get("severity", "")) or _txt(item.get("risk_level", ""))
    low = level.lower()
    if "high" in low:
        return "high"
    if "medium" in low or "med" in low:
        return "medium"
    if "low" in low:
        return "low"
    return "unknown"


def _severity_badge(level: str) -> str:
    if level == "high":
        return "\U0001F534 HIGH"
    if level == "medium":
        return "\U0001F7E0 MEDIUM"
    if level == "low":
        return "\U0001F7E2 LOW"
    return "\u26AA UNKNOWN"


def _category_from_text(*texts: str) -> str:
    blob = " ".join(_txt(t).lower() for t in texts)
    if any(k in blob for k in ["human", "worker", "person", "operator"]):
        return "human"
    if any(k in blob for k in ["hazard", "risk", "unsafe", "collision", "proximity", "danger", "severity"]):
        return "safety"
    if any(k in blob for k in ["moving", "movement", "speed", "trajectory", "displacement", "track"]):
        return "movement"
    if any(k in blob for k in ["conveyor", "shearer", "chock", "operation", "production", "equipment"]):
        return "operations"
    if any(k in blob for k in ["wall", "roof", "infrastructure", "structure", "environment"]):
        return "infrastructure"
    return "observation"


def _empty_issue_buckets() -> Dict[str, List[Dict]]:
    return {
        "safety": [],
        "operations": [],
        "human": [],
        "infrastructure": [],
        "movement": [],
        "observation": [],
    }


def _extract_single_issues(llm_json: Dict) -> Dict[str, List[Dict]]:
    buckets = _empty_issue_buckets()

    for hz in llm_json.get("hazards", []):
        if not isinstance(hz, dict):
            continue
        category = _category_from_text(hz.get("type"), hz.get("reason"))
        obj_ids = hz.get("object_ids", [])
        buckets[category].append(
            {
                "title": _txt(hz.get("type", "Hazard")),
                "detail": f"Objects: {obj_ids} | {_txt(hz.get('reason'))}",
                "severity": _severity_from_item(hz),
            }
        )

    for ins in llm_json.get("insights", []):
        if not isinstance(ins, dict):
            continue
        semantic = _txt(ins.get("semantic_label", ""))
        label = _txt(ins.get("generated_label", "Unknown"))
        obj_id = ins.get("object_id", "?")
        reason = _txt(ins.get("reasoning", ""))
        category = _category_from_text(semantic, label, reason)
        buckets[category].append(
            {
                "title": f"Object {obj_id}: {label}",
                "detail": (f"Semantic: {semantic} | " if semantic else "") + reason,
                "severity": _severity_from_item(ins),
            }
        )
    return buckets


def _extract_temporal_issues(llm_json: Dict) -> Dict[str, List[Dict]]:
    buckets = _empty_issue_buckets()

    for hz in llm_json.get("temporal_hazards", []):
        if not isinstance(hz, dict):
            continue
        category = _category_from_text(hz.get("type"), hz.get("reason"))
        tracks = hz.get("track_ids", [])
        buckets[category].append(
            {
                "title": _txt(hz.get("type", "Hazard")),
                "detail": f"Tracks: {tracks} | {_txt(hz.get('reason'))}",
                "severity": _severity_from_item(hz),
            }
        )

    for item in llm_json.get("object_identifications", []):
        if not isinstance(item, dict):
            continue
        track_id = item.get("track_id", "?")
        semantic = _txt(item.get("primary_semantic_label", ""))
        label = _txt(item.get("identified_as", "Unknown"))
        reason = _txt(item.get("reasoning", ""))
        category = _category_from_text(semantic, label, reason)
        buckets[category].append(
            {
                "title": f"Track {track_id}: {label}",
                "detail": (f"Semantic: {semantic} | " if semantic else "") + reason,
                "severity": _severity_from_item(item),
            }
        )

    movement = _txt(llm_json.get("movement_analysis", ""))
    if movement:
        buckets["movement"].append({"title": "Movement Analysis", "detail": movement, "severity": "unknown"})
    return buckets


def _render_issue_sections(buckets: Dict[str, List[Dict]]):
    meta = {
        "safety": ("\U0001F6E1\uFE0F", "Safety-Related Issues"),
        "operations": ("\u2699\uFE0F", "Operational Issues"),
        "human": ("\U0001F477", "Human-Related Issues"),
        "infrastructure": ("\U0001F3D7\uFE0F", "Infrastructure / Environment"),
        "movement": ("\U0001F9ED", "Movement & Tracking"),
        "observation": ("\U0001F4CC", "General Observations"),
    }
    order = ["safety", "operations", "human", "infrastructure", "movement", "observation"]
    for key in order:
        icon, title = meta[key]
        items = buckets.get(key, [])
        if not items:
            continue
        st.markdown(f"#### {icon} {title}")
        for item in items:
            badge = _severity_badge(item.get("severity", "unknown"))
            st.markdown(f"**{item.get('title', 'Issue')}**  \n{badge}  \n{item.get('detail', '')}")


def _render_llm_single_structured(llm_json: Dict):
    if not isinstance(llm_json, dict):
        st.warning("LLM JSON format is not an object. Showing raw content.")
        st.json(llm_json)
        return
    st.markdown("### \U0001F916 LLM Reasoning Report")
    parse_warning = _txt(llm_json.get("_parse_warning", ""))
    if parse_warning:
        st.warning(f"\u26A0\uFE0F {parse_warning}")
    safety = _txt(llm_json.get("safety_assessment", ""))
    if safety:
        if any(k in safety.lower() for k in ["unsafe", "risk", "hazard", "danger", "critical"]):
            st.warning(f"\U0001F6E1\uFE0F Safety Assessment: {safety}")
        else:
            st.success(f"\u2705 Safety Assessment: {safety}")
    else:
        st.info("\u2139\uFE0F No safety assessment text returned.")

    buckets = _extract_single_issues(llm_json)
    _render_issue_sections(buckets)

    with st.expander("Show Raw LLM JSON"):
        st.json(llm_json)


def _render_llm_temporal_structured(llm_json: Dict):
    if not isinstance(llm_json, dict):
        st.warning("LLM JSON format is not an object. Showing raw content.")
        st.json(llm_json)
        return
    st.markdown("### \U0001F916 LLM Temporal Reasoning Report")
    parse_warning = _txt(llm_json.get("_parse_warning", ""))
    if parse_warning:
        st.warning(f"\u26A0\uFE0F {parse_warning}")
    safety = _txt(llm_json.get("temporal_safety_assessment", ""))
    movement = _txt(llm_json.get("movement_analysis", ""))

    c1, c2 = st.columns(2)
    with c1:
        if safety:
            if any(k in safety.lower() for k in ["unsafe", "risk", "hazard", "danger", "critical"]):
                st.warning(f"\U0001F6E1\uFE0F Safety Assessment: {safety}")
            else:
                st.success(f"\u2705 Safety Assessment: {safety}")
        else:
            st.info("\u2139\uFE0F No safety assessment text returned.")
    with c2:
        if movement:
            st.info(f"\u2699\uFE0F Movement / Operation Summary: {movement}")
        else:
            st.info("\u2139\uFE0F No movement summary returned.")

    buckets = _extract_temporal_issues(llm_json)
    _render_issue_sections(buckets)

    with st.expander("Show Raw LLM JSON"):
        st.json(llm_json)


def _render_single_results(run_summary: Dict, class_meta: Dict):
    outputs = run_summary.get("outputs", {})
    metrics = run_summary.get("metrics", {})
    frame_dir = Path(run_summary["frame_dir"])

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Points", f"{metrics.get('point_count', 0):,}")
    c2.metric("Objects", str(metrics.get("object_count", 0)))
    c3.metric("Graph Nodes", str(metrics.get("graph_nodes", 0)))
    c4.metric("Graph Edges", str(metrics.get("graph_edges", 0)))
    c5.metric("Seg Time (s)", f"{metrics.get('segmentation_inference_seconds', 0.0):.3f}")
    c6.metric("LLM Time (s)", f"{metrics.get('llm_inference_seconds', 0.0):.3f}")
    st.caption(f"LLM status: {metrics.get('llm_status', 'unknown')}")

    st.markdown("### Segmentation Preview")
    st.caption(f"Segmentation inference time: {metrics.get('segmentation_inference_seconds', 0.0):.3f} s")
    npz_path = Path(outputs["semantic_npz"])
    if npz_path.exists():
        fig = _preview_from_npz(npz_path, class_meta=class_meta, title="Semantic Segmentation (Single Frame)")
        st.plotly_chart(fig, use_container_width=True)
        _save_plot_html(fig, frame_dir / "plots" / "single_segmentation_preview.html")
    else:
        st.warning("Missing semantic prediction file for preview.")

    class_counts = metrics.get("class_counts", {})
    if class_counts:
        fig_bar = bar_class_counts(class_counts, title="Class Point Distribution")
        st.plotly_chart(fig_bar, use_container_width=True)
        _save_plot_html(fig_bar, frame_dir / "plots" / "single_class_distribution.html")

    llm_path = outputs.get("llm_response")
    llm_raw = outputs.get("llm_raw")
    llm_status = metrics.get("llm_status", "unknown")
    if llm_path and Path(llm_path).exists():
        st.markdown("### LLM Output")
        st.caption(f"LLM inference time: {metrics.get('llm_inference_seconds', 0.0):.3f} s")
        llm_json = _read_json(Path(llm_path))
        _render_llm_single_structured(llm_json)
    elif llm_status == "skipped":
        st.info("LLM step was skipped for this run.")
    else:
        st.warning("LLM JSON output not available. Reasoning likely failed to parse.")
        if llm_raw and Path(llm_raw).exists():
            with st.expander("Show Raw LLM Output"):
                raw_text = Path(llm_raw).read_text(encoding="utf-8", errors="ignore")
                st.code(raw_text[:12000], language="text")

    st.markdown("### Output Folder")
    st.code(run_summary["frame_dir"], language="text")


def _render_temporal_results(run_summary: Dict, class_meta: Dict):
    outputs = run_summary.get("outputs", {})
    metrics = run_summary.get("metrics", {})
    run_dir = Path(run_summary["run_dir"])

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Frames", str(run_summary.get("num_frames", 0)))
    c2.metric("Total Objects", str(metrics.get("total_objects", 0)))
    c3.metric("Avg Inference (s)", f"{metrics.get('avg_inference_seconds', 0.0):.3f}")
    c4.metric("Estimated FPS", f"{metrics.get('fps_estimate', 0.0):.2f}")
    c5.metric("Seg Total (s)", f"{metrics.get('segmentation_total_seconds', 0.0):.3f}")
    c6.metric("LLM Time (s)", f"{metrics.get('llm_inference_seconds', 0.0):.3f}")
    st.caption(f"LLM status: {metrics.get('llm_status', 'unknown')}")

    frame_metrics_csv = Path(outputs["frame_metrics_csv"])
    if frame_metrics_csv.exists():
        frame_df = pd.read_csv(frame_metrics_csv)
    else:
        frame_df = safe_frame_metrics_df([])

    if not frame_df.empty:
        st.markdown("### Temporal Performance")
        fig_line = line_frame_metrics(frame_df)
        st.plotly_chart(fig_line, use_container_width=True)
        _save_plot_html(fig_line, run_dir / "plots" / "temporal_inference_metrics.html")

        st.markdown("### Per-frame Metrics")
        st.dataframe(frame_df[["frame_index", "frame_name", "num_points", "num_objects", "inference_seconds"]], use_container_width=True)

    class_counts = metrics.get("class_counts", {})
    if class_counts:
        st.markdown("### Aggregated Class Distribution")
        fig_bar = bar_class_counts(class_counts, title="Temporal Class Point Distribution")
        st.plotly_chart(fig_bar, use_container_width=True)
        _save_plot_html(fig_bar, run_dir / "plots" / "temporal_class_distribution.html")

    temporal_graph_path = Path(outputs["temporal_graph"])
    if temporal_graph_path.exists():
        temporal_graph = _read_json(temporal_graph_path)
        fig_mv = bar_movement_states(temporal_graph)
        st.plotly_chart(fig_mv, use_container_width=True)
        _save_plot_html(fig_mv, run_dir / "plots" / "temporal_movement_states.html")
    else:
        st.warning("Temporal graph file not found.")

    llm_path = outputs.get("llm_response")
    llm_raw = outputs.get("llm_raw")
    llm_status = metrics.get("llm_status", "unknown")
    if llm_path and Path(llm_path).exists():
        st.markdown("### LLM Output")
        st.caption(f"LLM inference time: {metrics.get('llm_inference_seconds', 0.0):.3f} s")
        _render_llm_temporal_structured(_read_json(Path(llm_path)))
    elif llm_status == "skipped":
        st.info("LLM step was skipped for this run.")
    else:
        st.warning("LLM JSON output not available. Reasoning likely failed to parse.")
        if llm_raw and Path(llm_raw).exists():
            with st.expander("Show Raw LLM Output"):
                raw_text = Path(llm_raw).read_text(encoding="utf-8", errors="ignore")
                st.code(raw_text[:12000], language="text")

    st.markdown("### Output Folder")
    st.code(run_summary["run_dir"], language="text")


def _common_sidebar():
    st.sidebar.markdown("## Pipeline Controls")
    mode = st.sidebar.radio("Mode", ["Single Frame", "Temporal Sequence"], index=0)
    run_name = st.sidebar.text_input("Run Name (optional)", "")

    st.sidebar.markdown("## Segmentation")
    mink_cfg = st.sidebar.text_input("MinkUNET Config", DEFAULT_INFERENCE_CONFIG)
    checkpoint = st.sidebar.text_input("Checkpoint", DEFAULT_CHECKPOINT)
    temporal_window = st.sidebar.slider("Temporal Smoothing Window", min_value=1, max_value=9, value=3, step=1)

    st.sidebar.markdown("## Graph")
    graph_dist = st.sidebar.slider("Graph Distance Threshold", min_value=0.5, max_value=10.0, value=2.5, step=0.1)

    st.sidebar.markdown("## LLM")
    run_llm = st.sidebar.checkbox("Run LLM Reasoning", value=True)
    local_models = discover_local_models(PROJECT_ROOT)
    model_index = {m["model_id"]: m for m in local_models}
    model_options = [m["model_id"] for m in local_models]

    selected_row = None
    if model_options:
        default_idx = model_options.index(DEFAULT_LLM_MODEL) if DEFAULT_LLM_MODEL in model_options else 0
        picked = st.sidebar.selectbox(
            "Available Models (workspace)",
            options=model_options + ["Custom model id..."],
            index=default_idx,
        )
        if picked == "Custom model id...":
            llm_model = st.sidebar.text_input("Custom LLM Model", DEFAULT_LLM_MODEL)
            selected_row = None
        else:
            llm_model = picked
            selected_row = model_index.get(picked)
    else:
        st.sidebar.warning("No cached local models detected in `LLM/` folders.", icon="⚠️")
        llm_model = st.sidebar.text_input("LLM Model", DEFAULT_LLM_MODEL)

    device_report = get_device_report()
    compat = compatibility_message(selected_row, device_report)

    with st.sidebar.expander("Advanced Class-DBSCAN Overrides"):
        eps_json = st.text_area(
            "class_dbscan_eps (JSON, optional)",
            value='{"human": 0.25, "equipment": 0.35}',
            height=100,
        )
        minpts_json = st.text_area(
            "class_min_points (JSON, optional)",
            value='{"human": 20, "equipment": 80}',
            height=100,
        )

    st.sidebar.info(
        "💡 Need more models?\n\n"
        "Use a HuggingFace model id in `Custom model id` and run once to download, "
        "or download to `LLM/Model_Cache` manually.",
    )

    return {
        "mode": mode,
        "run_name": run_name,
        "mink_cfg": mink_cfg,
        "checkpoint": checkpoint,
        "temporal_window": temporal_window,
        "graph_dist": graph_dist,
        "run_llm": run_llm,
        "llm_model": llm_model,
        "eps_json": eps_json,
        "minpts_json": minpts_json,
        "local_models": local_models,
        "selected_model_row": selected_row,
        "device_report": device_report,
        "compatibility": compat,
    }


def _render_model_runtime_panel(controls: Dict):
    st.markdown("## LLM Model Availability & Runtime Check")
    left, right = st.columns([1.5, 1.0])

    with left:
        local_models = controls.get("local_models", [])
        if local_models:
            rows = []
            for m in local_models:
                rows.append(
                    {
                        "Model": m["model_id"],
                        "Weights Size (GB)": m["weights_gb"],
                        "Estimated VRAM Required (GB)": m["required_vram_gb"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No cached models found yet in workspace `LLM/` folders.", icon="ℹ️")

    with right:
        device = controls.get("device_report") or {}
        st.info(f"🖥️ Current device: {device.get('device_text', 'Unknown')}")
        comp = controls.get("compatibility") or {}
        title = comp.get("title", "Compatibility")
        message = comp.get("message", "")
        icon = comp.get("icon", "ℹ️")
        status = comp.get("status", "info")
        text = f"{icon} {title}\n\n{message}"
        if status == "success":
            st.success(text)
        elif status == "warning":
            st.warning(text)
        elif status == "error":
            st.error(text)
        else:
            st.info(text)


def main():
    st.set_page_config(page_title="Mine Scene Intelligence App", layout="wide")
    _inject_theme()

    st.title("Mine Scene Intelligence App")
    st.caption("MinkUNET segmentation + scene graph + local LLM reasoning (isolated app pipeline)")
    st.info("All app outputs are saved only under `AppUI/app_results/`.")

    if PANDAS_IMPORT_ERROR is not None or pd is None:
        st.error(
            "Pandas failed to import due to a NumPy/Pandas binary mismatch in this environment.\n\n"
            "Fix in your active venv:\n"
            "1) `pip install --upgrade pip`\n"
            "2) `pip uninstall -y numpy pandas`\n"
            "3) `pip install numpy pandas`\n\n"
            f"Original error: {PANDAS_IMPORT_ERROR}"
        )
        st.stop()

    controls = _common_sidebar()
    mode = controls["mode"]
    _render_model_runtime_panel(controls)

    try:
        class_meta = _load_class_meta(controls["mink_cfg"])
    except Exception as exc:
        st.error(f"Could not load config `{controls['mink_cfg']}`: {exc}")
        return

    try:
        class_dbscan_eps = _parse_json_map(controls["eps_json"], value_type="float")
        class_min_points = _parse_json_map(controls["minpts_json"], value_type="int")
    except Exception as exc:
        st.error(f"Invalid advanced JSON setting: {exc}")
        return

    if mode == "Single Frame":
        st.markdown("## Single-frame Run")
        folder = st.text_input("PCD Folder", "Datasets/Data_all")
        files = list_pcd_files(folder)
        if not files:
            st.warning("No `PC_*.pcd` files found in this folder.")
            return

        labels = [p.name for p in files]
        selected_name = st.selectbox("Select Point Cloud", labels, index=0)
        selected_file = next(p for p in files if p.name == selected_name)
        ts = parse_pcd_timestamp(selected_file)
        st.markdown(f"<div class='caption-soft'>Timestamp: {ts.isoformat() if ts else 'N/A'}</div>", unsafe_allow_html=True)

        run_btn = st.button("Run Single-frame Pipeline", type="primary")
        if run_btn:
            run_dir = create_run_dir(mode="single", run_name=controls["run_name"] or "single")
            progress = st.progress(0.0, text="Starting run...")
            status = st.empty()

            def cb(event: Dict):
                progress.progress(float(event.get("progress", 0.0)), text=str(event.get("message", "")))
                status.info(f"[{event.get('stage', 'stage')}] {event.get('message', '')}")

            try:
                run_summary = run_single_pipeline(
                    pcd_path=str(selected_file),
                    run_dir=run_dir,
                    minkunet_config=controls["mink_cfg"],
                    checkpoint_path=controls["checkpoint"],
                    temporal_window=controls["temporal_window"],
                    graph_dist_threshold=controls["graph_dist"],
                    run_llm=controls["run_llm"],
                    llm_model=controls["llm_model"],
                    class_dbscan_eps=class_dbscan_eps,
                    class_min_points=class_min_points,
                    progress_callback=cb,
                )
                status.success("Single-frame pipeline completed.")
                st.session_state["last_run"] = run_summary
            except AppRunError as exc:
                status.error(f"Run failed: {exc}")
                return
            except Exception as exc:
                status.error(f"Unexpected error: {exc}")
                return

        if st.session_state.get("last_run", {}).get("mode") == "single":
            st.markdown("---")
            st.markdown("## Latest Single-frame Results")
            _render_single_results(st.session_state["last_run"], class_meta=class_meta)

    else:
        st.markdown("## Temporal Run")
        folder = st.text_input("PCD Sequence Folder", "Datasets/Data_all")
        files = list_pcd_files(folder)
        if not files:
            st.warning("No `PC_*.pcd` files found in this folder.")
            return

        max_idx = len(files) - 1
        idx_range = st.slider("Frame Index Range", min_value=0, max_value=max_idx, value=(0, max_idx), step=1)
        gap_seconds = st.slider("Minimum Gap Between Consecutive Frames (seconds)", min_value=1, max_value=120, value=10, step=1)
        max_track_distance = st.slider("Max Track Distance", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        movement_threshold = st.slider("Movement Threshold", min_value=0.05, max_value=2.0, value=0.2, step=0.05)
        enable_live_preview = st.checkbox("Live Segmentation Preview While Running", value=True)

        selected_files = range_and_gap_select(files, index_range=idx_range, gap_seconds=gap_seconds)
        st.info(f"Selected {len(selected_files)} frames from {len(files)} available.")
        st.dataframe(pd.DataFrame(frame_rows(selected_files)), use_container_width=True, height=220)

        run_btn = st.button("Run Temporal Pipeline", type="primary")
        if run_btn:
            if not selected_files:
                st.warning("No frames selected after applying range and gap.")
                return

            run_dir = create_run_dir(mode="temporal", run_name=controls["run_name"] or "temporal")
            progress = st.progress(0.0, text="Starting run...")
            status = st.empty()
            preview_slot = st.empty()

            def cb(event: Dict):
                progress.progress(float(event.get("progress", 0.0)), text=str(event.get("message", "")))
                status.info(f"[{event.get('stage', 'stage')}] {event.get('message', '')}")
                npz = event.get("preview_npz")
                if enable_live_preview and npz:
                    npz_path = Path(str(npz))
                    if npz_path.exists():
                        fig = _preview_from_npz(
                            npz_path,
                            class_meta=class_meta,
                            title=f"Live Preview: {event.get('frame_name', npz_path.parent.name)}",
                            max_points=12000,
                        )
                        preview_slot.plotly_chart(fig, use_container_width=True)

            try:
                run_summary = run_temporal_pipeline(
                    selected_files=selected_files,
                    run_dir=run_dir,
                    minkunet_config=controls["mink_cfg"],
                    checkpoint_path=controls["checkpoint"],
                    temporal_window=controls["temporal_window"],
                    graph_dist_threshold=controls["graph_dist"],
                    max_track_distance=max_track_distance,
                    movement_threshold=movement_threshold,
                    run_llm=controls["run_llm"],
                    llm_model=controls["llm_model"],
                    class_dbscan_eps=class_dbscan_eps,
                    class_min_points=class_min_points,
                    progress_callback=cb,
                )
                status.success("Temporal pipeline completed.")
                st.session_state["last_run"] = run_summary
            except AppRunError as exc:
                status.error(f"Run failed: {exc}")
                return
            except Exception as exc:
                status.error(f"Unexpected error: {exc}")
                return

        if st.session_state.get("last_run", {}).get("mode") == "temporal":
            st.markdown("---")
            st.markdown("## Latest Temporal Results")
            _render_temporal_results(st.session_state["last_run"], class_meta=class_meta)


if __name__ == "__main__":
    main()
