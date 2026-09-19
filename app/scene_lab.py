from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mine_safety.config import load_config
from mine_safety.graph.scene import build_scene_graph
from mine_safety.rules import SafetyRuleEngine

APP_ICON = PROJECT_ROOT / "assets" / "app_icon.svg"
FIGURES = PROJECT_ROOT / "assets" / "figures"
st.set_page_config(page_title="MineGraph Safety Lab", page_icon=str(APP_ICON), layout="wide")


def inject_design() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        .stApp {
          background:
            radial-gradient(circle at 15% 0%, rgba(41,151,255,.14), transparent 32rem),
            radial-gradient(circle at 90% 10%, rgba(48,209,88,.09), transparent 28rem),
            var(--background-color);
          font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
        }
        .block-container { max-width: 1480px; padding-top: 1.35rem; padding-bottom: 3rem; }
        h1 { letter-spacing: -.035em !important; line-height: 1.04 !important; font-weight: 720 !important; }
        h2, h3 { letter-spacing: -.018em !important; }
        .hero {
          padding: 1.35rem 1.5rem; border-radius: 24px;
          background: color-mix(in srgb, var(--secondary-background-color) 78%, transparent);
          border: 1px solid rgba(128,128,128,.20);
          box-shadow: 0 18px 48px rgba(0,0,0,.10), inset 0 1px rgba(255,255,255,.28);
          backdrop-filter: blur(24px) saturate(155%);
        }
        .eyebrow { color: #2997ff; font-weight: 700; font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; }
        .hero-copy { max-width: 760px; opacity: .76; font-size: 1.02rem; line-height: 1.55; }
        .notice {
          margin-top: .9rem; padding: .8rem 1rem; border-radius: 14px;
          background: rgba(255,159,10,.10); border: 1px solid rgba(255,159,10,.30);
          font-size: .9rem; line-height: 1.45;
        }
        .figure-card {
          padding: .75rem; border-radius: 20px; background: #fff;
          border: 1px solid rgba(128,128,128,.18); box-shadow: 0 10px 30px rgba(0,0,0,.06);
        }
        .stage-card {
          padding: 1rem 1.05rem; border-radius: 18px; min-height: 126px;
          border: 1px solid rgba(128,128,128,.18); background: rgba(255,255,255,.78);
          color: #1d1d1f !important;
        }
        .stage-card h3, .stage-card p { color: #1d1d1f !important; }
        .stage-ready { color: #16833a; font-weight: 700; }
        .stage-wip { color: #a65d00; font-weight: 700; }
        [data-testid="stMetric"] {
          padding: .85rem 1rem; border-radius: 18px;
          border: 1px solid rgba(128,128,128,.18);
          background: color-mix(in srgb, var(--secondary-background-color) 82%, transparent);
          box-shadow: inset 0 1px rgba(255,255,255,.20);
        }
        [data-testid="stMetricValue"] { letter-spacing: -.03em; }
        .stButton button, .stDownloadButton button { border-radius: 999px; transition: transform 100ms ease-out; }
        .stButton button:active, .stDownloadButton button:active { transform: scale(.97); }
        [data-baseweb="tab-list"] { gap: .25rem; background: rgba(128,128,128,.08); padding: .28rem; border-radius: 14px; }
        [data-baseweb="tab"] { border-radius: 10px; padding: .55rem .9rem; }
        [data-baseweb="tab-highlight"] { display: none; }
        @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; transition: none !important; } }
        @media (prefers-reduced-transparency: reduce) { .hero { backdrop-filter: none; } }
        @media (prefers-contrast: more) { .hero, [data-testid="stMetric"] { border-width: 2px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


CLASSES = [
    "personnel", "shearer_cutter", "mining_vehicle", "hydraulic_support",
    "roof_canopy", "conveyor", "mining_face", "tunnel_wall", "floor", "anomaly",
]
COLORS = {
    "personnel": "#30D158", "shearer_cutter": "#FF453A", "mining_vehicle": "#FF9F0A",
    "hydraulic_support": "#BF5AF2", "roof_canopy": "#64D2FF", "conveyor": "#FFD60A",
    "mining_face": "#AC8E68", "tunnel_wall": "#8E8E93", "floor": "#636366", "anomaly": "#FF375F",
}


def scenario(name: str) -> pd.DataFrame:
    base = [
        {"id": "equipment-01", "label": "mining_vehicle", "x": 0.0, "y": 0.0, "z": 0.5,
         "size_x": 3.0, "size_y": 1.8, "size_z": 1.8, "vx": 0.5, "vy": 0.0, "vz": 0.0,
         "heading_x": 1.0, "heading_y": 0.0, "heading_z": 0.0, "active": True, "entropy": 0.05},
    ]
    if name == "Safe passage":
        base += [{"id": "person-01", "label": "personnel", "x": 5.5, "y": 2.5, "z": 0.9,
                  "size_x": .5, "size_y": .5, "size_z": 1.8, "vx": 0.0, "vy": .2, "vz": 0.0,
                  "heading_x": 0.0, "heading_y": 1.0, "heading_z": 0.0, "active": True, "entropy": .08}]
    elif name == "Proximity breach":
        base += [{"id": "person-01", "label": "personnel", "x": 1.7, "y": 0.2, "z": 0.9,
                  "size_x": .5, "size_y": .5, "size_z": 1.8, "vx": 0.0, "vy": 0.0, "vz": 0.0,
                  "heading_x": 1.0, "heading_y": 0.0, "heading_z": 0.0, "active": True, "entropy": .12}]
    elif name == "Blind spot":
        base += [{"id": "person-01", "label": "personnel", "x": -2.2, "y": 0.1, "z": 0.9,
                  "size_x": .5, "size_y": .5, "size_z": 1.8, "vx": 0.0, "vy": 0.0, "vz": 0.0,
                  "heading_x": 1.0, "heading_y": 0.0, "heading_z": 0.0, "active": True, "entropy": .10}]
    elif name == "Congested zone":
        for index, (x, y) in enumerate([(2.8, .4), (2.5, -1.1), (3.2, 1.2)], 1):
            base += [{"id": f"person-{index:02d}", "label": "personnel", "x": x, "y": y, "z": .9,
                      "size_x": .5, "size_y": .5, "size_z": 1.8, "vx": 0.0, "vy": 0.0, "vz": 0.0,
                      "heading_x": 1.0, "heading_y": 0.0, "heading_z": 0.0, "active": True, "entropy": .10}]
    else:
        base += [{"id": "anomaly-01", "label": "anomaly", "x": 2.8, "y": 1.0, "z": 1.7,
                  "size_x": .8, "size_y": .7, "size_z": .6, "vx": 0.0, "vy": 0.0, "vz": 0.0,
                  "heading_x": 0.0, "heading_y": 0.0, "heading_z": 1.0, "active": True, "entropy": .89}]
    return pd.DataFrame(base)


def add_object(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    prefix = "person" if label == "personnel" else label.replace("_", "-")
    used = {str(value) for value in frame["id"].tolist()}
    index = 1
    while f"{prefix}-{index:02d}" in used:
        index += 1
    row = {
        "id": f"{prefix}-{index:02d}", "label": label, "x": 0.0, "y": 0.0, "z": 0.8,
        "size_x": 0.6, "size_y": 0.6, "size_z": 1.6, "vx": 0.0, "vy": 0.0, "vz": 0.0,
        "heading_x": 1.0, "heading_y": 0.0, "heading_z": 0.0, "active": True,
        "entropy": 0.8 if label == "anomaly" else 0.1,
    }
    return pd.concat([frame, pd.DataFrame([row])], ignore_index=True)


def rows_to_nodes(frame: pd.DataFrame) -> list[dict]:
    nodes = []
    for _, row in frame.iterrows():
        node = {
            "id": str(row["id"]), "label": str(row["label"]),
            "centroid": (float(row["x"]), float(row["y"]), float(row["z"])),
            "bbox_dimensions": (float(row["size_x"]), float(row["size_y"]), float(row["size_z"])),
            "velocity": (float(row["vx"]), float(row["vy"]), float(row["vz"])),
            "orientation": (float(row["heading_x"]), float(row["heading_y"]), float(row["heading_z"])),
            "active": bool(row["active"]), "entropy": max(float(row["entropy"]), 0.0),
            "is_anomaly": str(row["label"]) == "anomaly", "confidence": 0.0 if row["label"] == "anomaly" else .95,
        }
        node["volume_m3"] = float(np.prod(node["bbox_dimensions"]))
        nodes.append(node)
    return nodes


def graph_figure(graph) -> go.Figure:
    lookup = {node.id: node for node in graph.nodes}
    fig = go.Figure()
    for edge in graph.edges:
        source, target = lookup[edge.source], lookup[edge.target]
        fig.add_trace(go.Scatter3d(
            x=[source.centroid[0], target.centroid[0]], y=[source.centroid[1], target.centroid[1]],
            z=[source.centroid[2], target.centroid[2]], mode="lines",
            line={"color": "rgba(142,142,147,.22)", "width": 2}, hoverinfo="skip", showlegend=False,
        ))
    for label in sorted({node.label for node in graph.nodes}):
        items = [node for node in graph.nodes if node.label == label]
        fig.add_trace(go.Scatter3d(
            x=[node.centroid[0] for node in items], y=[node.centroid[1] for node in items],
            z=[node.centroid[2] for node in items], mode="markers+text",
            text=[node.id for node in items], textposition="top center", name=label.replace("_", " ").title(),
            marker={"size": [max(9, min(22, 8 + np.cbrt(max(node.volume_m3, .01)) * 6)) for node in items],
                    "color": COLORS.get(label, "#2997FF"), "opacity": .92,
                    "line": {"color": "rgba(255,255,255,.65)", "width": 1}},
            hovertemplate="<b>%{text}</b><br>x=%{x:.2f} m<br>y=%{y:.2f} m<br>z=%{z:.2f} m<extra></extra>",
        ))
    fig.update_layout(
        height=590, margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        scene={"bgcolor": "rgba(0,0,0,0)", "aspectmode": "data",
               "xaxis_title": "Longwall x (m)", "yaxis_title": "Cross-cut y (m)",
               "zaxis_title": "Height z (m)"},
        legend={"orientation": "h", "y": 1.04},
    )
    return fig


inject_design()
config = load_config()
if "objects" not in st.session_state:
    st.session_state.objects = scenario("Safe passage")
if "editor_revision" not in st.session_state:
    st.session_state.editor_revision = 0

if hasattr(st, "logo"):
    st.logo(str(APP_ICON), size="large")

st.markdown(
    """<section class="hero"><div class="eyebrow">Interactive research companion</div>
    <h1>MineGraph Safety Lab</h1>
    <p class="hero-copy">Build one underground scene graph, inspect its spatial relationships, and test the
    paper-aligned deterministic safety layer. Every finding stays traceable to the objects you place.</p>
    <div class="notice"><b>Work in progress · Demonstration only.</b> This interface is not connected to ROS,
    Gazebo, mine sensors, a control system, or an operational simulator. It is a single-graph educational
    sandbox and must not be used for safety decisions.</div></section>""",
    unsafe_allow_html=True,
)
st.write("")

with st.sidebar:
    st.subheader("Scenario controls")
    preset = st.selectbox("Starting point", ["Safe passage", "Proximity breach", "Blind spot", "Congested zone", "Anomaly nearby"])
    if st.button("Load preset", width="stretch"):
        st.session_state.objects = scenario(preset)
        st.session_state.editor_revision += 1
        st.rerun()
    st.divider()
    st.subheader("Add an object")
    quick_class = st.selectbox("Object class", CLASSES, format_func=lambda value: value.replace("_", " ").title())
    if st.button("Add to scene", width="stretch"):
        st.session_state.objects = add_object(st.session_state.objects, quick_class)
        st.session_state.editor_revision += 1
        st.rerun()
    mean_intensity = st.slider("Scene intensity", 0, 255, 90, help="Paper threshold: rolling mean below 25/255")
    edge_distance = st.slider("Graph edge range (m)", 1.0, 12.0, float(config["graph"]["edge_distance_m"]), .5)
    st.caption("Edit positions, dimensions, velocities, headings, activity, and entropy in the table.")

objects = st.data_editor(
    st.session_state.objects,
    num_rows="dynamic", width="stretch", hide_index=True,
    key=f"scene_editor_{st.session_state.editor_revision}",
    column_config={
        "label": st.column_config.SelectboxColumn("Class", options=CLASSES, required=True),
        "active": st.column_config.CheckboxColumn("Active"),
        "entropy": st.column_config.NumberColumn("Entropy", min_value=0.0, step=.01, format="%.2f"),
    },
)
st.session_state.objects = objects

try:
    graph = build_scene_graph(rows_to_nodes(objects), edge_distance_m=edge_distance, mean_intensity=float(mean_intensity))
    alerts = SafetyRuleEngine(config["rules"]).evaluate(graph)
except (KeyError, TypeError, ValueError) as exc:
    st.error(f"Fix the scene table before testing: {exc}")
    st.stop()

metrics = st.columns(4)
metrics[0].metric("Objects", len(graph.nodes))
metrics[1].metric("Directed relations", len(graph.edges))
metrics[2].metric("Rule alerts", len(alerts))
metrics[3].metric("Mean intensity", f"{mean_intensity}/255")

builder_tab, graph_tab, reasoning_tab, llm_tab, pipeline_tab = st.tabs(
    ["Scene builder", "Graph explorer", "Rule reasoning", "Contextual reasoning", "Pipeline map"]
)

with builder_tab:
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        st.plotly_chart(graph_figure(graph), width="stretch", config={"displaylogo": False})
    with right:
        st.subheader("Compose a scenario")
        st.write("Add a row for each entity, choose its class, then set position, physical size, velocity, and heading. The graph updates immediately.")
        st.info("Tip: three personnel inside a 4 m equipment zone trigger the paper's density rule; place personnel behind moving equipment to test the blind-spot rule.")
        payload = graph.model_dump_json(indent=2)
        st.download_button("Download scene graph JSON", payload, "scene_graph.json", "application/json", width="stretch")

with graph_tab:
    st.subheader("Object-to-object evidence")
    edge_rows = [{
        "source": edge.source, "relation": edge.relation, "target": edge.target,
        "distance_m": round(edge.distance_m, 3), "safety_flags": ", ".join(edge.safety_flags),
    } for edge in graph.edges]
    st.dataframe(edge_rows, width="stretch", hide_index=True)
    with st.expander("Validated graph JSON"):
        st.json(graph.model_dump(mode="json"))

with reasoning_tab:
    if not alerts:
        st.success("No deterministic rule threshold is currently violated.")
    for alert in alerts:
        icon = "🔴" if alert.severity in {"critical", "high"} else "🟠"
        with st.container(border=True):
            st.subheader(f"{icon} {alert.rule.replace('_', ' ').title()}")
            st.write(alert.message)
            st.caption(f"Objects: {', '.join(alert.object_ids) or 'whole scene'}")
            st.json(alert.evidence, expanded=False)
    st.caption("Deterministic alerts are authoritative within this sandbox. Thresholds are research defaults and require site-specific validation.")
    st.info("Low visibility is not evaluated here because the paper requires a rolling 3-second intensity window; this lab contains only one snapshot.")

with llm_tab:
    st.subheader("Contextual and longitudinal reasoning")
    stage_columns = st.columns(3, gap="medium")
    with stage_columns[0]:
        st.markdown('<div class="stage-card"><div class="stage-ready">Available now</div><h3>Deterministic rules</h3><p>Spatial checks run against this graph and write flags back to its edges.</p></div>', unsafe_allow_html=True)
    with stage_columns[1]:
        st.markdown('<div class="stage-card"><div class="stage-wip">Not run in this lab</div><h3>Contextual Qwen</h3><p>Requires a 10-second temporal graph and local model weights.</p></div>', unsafe_allow_html=True)
    with stage_columns[2]:
        st.markdown('<div class="stage-card"><div class="stage-wip">Not run in this lab</div><h3>Longitudinal GraphRAG</h3><p>Requires linked historical memories and a local Qdrant archive.</p></div>', unsafe_allow_html=True)
    st.warning("Work in progress · the interactive page does not execute model inference")
    st.write("The repository includes the paper's full Appendix A/B prompts, distinct strict JSON schemas, object/memory-ID grounding, one regeneration attempt, linked-memory filtering, and local Qwen/Qdrant adapters.")
    st.code("mine-safety examples/scene_graph.json --llm", language="bash")
    st.caption("Models are downloaded only when you explicitly enable the local reasoning path. No cloud endpoint is used.")

with pipeline_tab:
    st.subheader("Paper pipeline")
    st.image(str(FIGURES / "Graphical Abstract_final.png"), width="stretch")
    st.caption("Graphical abstract supplied with the paper assets. The live sandbox implements the highlighted single-graph path only.")
    st.markdown("**Current sandbox path:** manual objects → one scene graph → deterministic spatial rules.  **Full repository path:** colourised point cloud → MinkUNet → entropy anomalies → scene/temporal graph → rules + local Qwen → selective GraphRAG.")
    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        st.image(str(FIGURES / "NN model_v2.png"), caption="Sparse 3D neural architecture", width="stretch")
        st.image(str(FIGURES / "LLM Pipline v2.png"), caption="Selective longitudinal reasoning", width="stretch")
    with right:
        st.image(str(FIGURES / "anaomaly figure v2.png"), caption="Entropy-to-anomaly proposal pipeline", width="stretch")
        st.image(str(FIGURES / "hazards in tunnel.png"), caption="Graph-grounded hazard examples", width="stretch")
    st.caption("Author-asset note: where a figure label differs from the manuscript, the executable code follows the paper text—0.01 m voxels and only recurring/escalating longitudinal patterns.")
