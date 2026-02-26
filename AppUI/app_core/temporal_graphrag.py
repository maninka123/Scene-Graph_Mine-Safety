from __future__ import annotations

import ast
import importlib.util
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_timestamp_any(ts_value: Any) -> Optional[datetime]:
    if ts_value is None:
        return None
    if isinstance(ts_value, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts_value))
        except Exception:
            return None

    ts = str(ts_value).strip()
    if not ts:
        return None

    try:
        return datetime.fromisoformat(ts)
    except Exception:
        pass

    try:
        if "_" in ts and len(ts.split("_")) == 2:
            date_part, time_part = ts.split("_")
            hms = time_part[:6]
            frac = time_part[6:]
            base = datetime.strptime(date_part + hms, "%Y%m%d%H%M%S")
            if frac:
                usec = int(frac[:6].ljust(6, "0"))
                base = base.replace(microsecond=usec)
            return base
    except Exception:
        pass
    return None


def _format_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "Unknown"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", str(text).lower())


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    low = str(text).lower()
    return any(term in low for term in terms)


def _question_requests_visual(question: str) -> bool:
    return _contains_any(
        question,
        [
            "plot",
            "graph",
            "vector",
            "visual",
            "visualize",
            "chart",
            "show",
            "trajectory",
            "arrow",
            "link",
        ],
    )


def _is_explicit_date_range_question(text: str) -> bool:
    low = str(text).lower()
    if _contains_any(low, ["date range", "time range", "period", "time window", "dates covered", "range covered"]):
        return True
    return re.search(r"\bfrom\b.+\bto\b", low) is not None


def _label_histogram(tracked_objects: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in tracked_objects:
        lbl = str(item.get("primary_label", "unknown")).lower().strip()
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def _extract_time_bounds(graph: Dict[str, Any]) -> Tuple[Optional[datetime], Optional[datetime]]:
    times: List[datetime] = []
    tracked = graph.get("tracked_objects", [])
    if isinstance(tracked, list):
        for track in tracked:
            if not isinstance(track, dict):
                continue
            for ts in track.get("timestamps", []):
                parsed = _parse_timestamp_any(ts)
                if parsed is not None:
                    times.append(parsed)
    if not times:
        return None, None
    return min(times), max(times)


def discover_temporal_graphs(search_roots: Sequence[Path]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    seen: set = set()
    for root in search_roots:
        if not root.exists():
            continue
        for graph_path in root.rglob("temporal_scene_graph.json"):
            resolved = str(graph_path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)

            graph = _load_json(graph_path)
            if not graph:
                continue

            tracked = graph.get("tracked_objects", [])
            tracked = tracked if isinstance(tracked, list) else []
            movement_events = graph.get("movement_events", [])
            movement_events = movement_events if isinstance(movement_events, list) else []
            summary = graph.get("summary", {})
            summary = summary if isinstance(summary, dict) else {}

            start_dt, end_dt = _extract_time_bounds(graph)
            labels = _label_histogram(tracked)

            source_kind = "app_results" if "AppUI" in graph_path.parts else "results"
            doc = {
                "graph_id": resolved,
                "graph_path": resolved,
                "run_dir": str(graph_path.parent.resolve()),
                "run_name": graph_path.parent.name,
                "source_kind": source_kind,
                "start_dt": start_dt,
                "end_dt": end_dt,
                "start_label": _format_dt(start_dt),
                "end_label": _format_dt(end_dt),
                "summary": summary,
                "labels": labels,
                "total_tracks": _safe_int(summary.get("total_tracks", len(tracked))),
                "moving_objects": _safe_int(summary.get("moving_objects", len(movement_events))),
                "graph": graph,
            }
            docs.append(doc)

    docs.sort(
        key=lambda d: (
            d.get("end_dt") is None,
            d.get("end_dt") or datetime.min,
            d.get("run_name", ""),
        ),
        reverse=True,
    )
    return docs


def summarize_temporal_coverage(docs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not docs:
        return {
            "count": 0,
            "start_dt": None,
            "end_dt": None,
            "start_label": "Unknown",
            "end_label": "Unknown",
        }
    starts = [d.get("start_dt") for d in docs if d.get("start_dt") is not None]
    ends = [d.get("end_dt") for d in docs if d.get("end_dt") is not None]
    start_dt = min(starts) if starts else None
    end_dt = max(ends) if ends else None
    return {
        "count": len(docs),
        "start_dt": start_dt,
        "end_dt": end_dt,
        "start_label": _format_dt(start_dt),
        "end_label": _format_dt(end_dt),
    }


def _score_graph(question_tokens: set, doc: Dict[str, Any]) -> float:
    score = 0.0
    labels = set(doc.get("labels", {}).keys())
    overlap = len(question_tokens.intersection(labels))
    score += overlap * 2.0

    blob_parts = [doc.get("run_name", "")]
    for key, value in doc.get("summary", {}).items():
        blob_parts.append(f"{key}:{value}")
    graph_blob = " ".join(str(x).lower() for x in blob_parts)
    for token in question_tokens:
        if token in graph_blob:
            score += 0.3

    movement_need = any(t in question_tokens for t in {"moving", "movement", "speed", "velocity", "trajectory"})
    risk_need = any(t in question_tokens for t in {"risk", "hazard", "unsafe", "safe", "anomaly"})
    if movement_need:
        score += min(4.0, float(doc.get("moving_objects", 0)) * 0.15)
    if risk_need:
        score += min(3.0, float(doc.get("moving_objects", 0)) * 0.1)

    if doc.get("end_dt") is not None:
        score += 0.5
    return score


def _anomaly_thresholds(tracks: Sequence[Dict[str, Any]]) -> Tuple[float, float]:
    disps = [_safe_float(t.get("total_displacement", 0.0)) for t in tracks]
    speeds = [_safe_float(t.get("avg_speed_m_per_s", 0.0)) for t in tracks if t.get("avg_speed_m_per_s") is not None]
    if disps:
        mean_d = sum(disps) / len(disps)
        var_d = sum((x - mean_d) ** 2 for x in disps) / len(disps)
        disp_thr = max(1.0, mean_d + math.sqrt(var_d))
    else:
        disp_thr = 1.0
    if speeds:
        mean_s = sum(speeds) / len(speeds)
        var_s = sum((x - mean_s) ** 2 for x in speeds) / len(speeds)
        speed_thr = max(0.25, mean_s + math.sqrt(var_s))
    else:
        speed_thr = 0.5
    return disp_thr, speed_thr


def _is_track_anomaly(track: Dict[str, Any], disp_thr: float, speed_thr: float) -> bool:
    disp = _safe_float(track.get("total_displacement", 0.0))
    speed = _safe_float(track.get("avg_speed_m_per_s", 0.0)) if track.get("avg_speed_m_per_s") is not None else 0.0
    state = str(track.get("movement_state", "")).lower()
    return disp >= disp_thr or speed >= speed_thr or "fast" in state


def _score_track(question_tokens: set, track: Dict[str, Any], disp_thr: float, speed_thr: float) -> float:
    score = 0.0
    lbl = str(track.get("primary_label", "unknown")).lower()
    state = str(track.get("movement_state", "")).lower()
    if lbl in question_tokens:
        score += 3.0

    if any(t in question_tokens for t in {"human", "worker", "person"}) and lbl == "human":
        score += 4.0
    if any(t in question_tokens for t in {"equipment", "shearer", "conveyor", "support"}) and lbl in {"equipment", "conveyor"}:
        score += 2.5
    if any(t in question_tokens for t in {"moving", "movement", "speed", "velocity"}) and state != "stationary":
        score += 2.0
    if any(t in question_tokens for t in {"risk", "hazard", "unsafe", "anomaly"}) and _is_track_anomaly(track, disp_thr, speed_thr):
        score += 2.0

    score += min(2.0, _safe_float(track.get("total_displacement", 0.0)) * 0.4)
    if "fast" in state:
        score += 1.0
    return score


def retrieve_temporal_subgraph(
    question: str,
    docs: Sequence[Dict[str, Any]],
    selected_graph_ids: Sequence[str],
    max_graphs: int = 4,
    max_tracks_per_graph: int = 8,
    max_events_per_graph: int = 6,
    max_clearance_edges_per_graph: int = 6,
) -> Dict[str, Any]:
    question_tokens = set(_tokenize(question))
    selected_set = {str(x) for x in selected_graph_ids}
    candidates = [d for d in docs if not selected_set or d.get("graph_id") in selected_set]
    if not candidates:
        return {
            "selected_docs": [],
            "context_text": "",
            "evidence_lines": [],
            "coverage": summarize_temporal_coverage([]),
        }

    scored_docs = [(d, _score_graph(question_tokens, d)) for d in candidates]
    scored_docs.sort(key=lambda item: (item[1], item[0].get("end_dt") or datetime.min), reverse=True)
    chosen_docs = [item[0] for item in scored_docs[: max(1, min(max_graphs, len(scored_docs)))]]

    evidence_lines: List[str] = []
    context_lines: List[str] = []
    context_lines.append("Retrieved temporal evidence for question answering:")

    selected_doc_payloads: List[Dict[str, Any]] = []
    visual_tracks: List[Dict[str, Any]] = []
    total_tracks_sum = 0
    total_moving_sum = 0
    total_human_tracks_sum = 0
    latest_doc_end = None
    latest_doc_stats: Dict[str, Any] = {}
    for idx, doc in enumerate(chosen_docs, start=1):
        graph = doc.get("graph", {})
        tracked = graph.get("tracked_objects", [])
        tracked = tracked if isinstance(tracked, list) else []
        movement_events = graph.get("movement_events", [])
        movement_events = movement_events if isinstance(movement_events, list) else []
        spatial = graph.get("spatial_relationships", {})
        spatial = spatial if isinstance(spatial, dict) else {}

        disp_thr, speed_thr = _anomaly_thresholds(tracked)

        scored_tracks = [(t, _score_track(question_tokens, t, disp_thr, speed_thr)) for t in tracked if isinstance(t, dict)]
        scored_tracks.sort(key=lambda item: item[1], reverse=True)
        top_tracks = [item[0] for item in scored_tracks[: max_tracks_per_graph]]
        top_track_ids = {_safe_int(t.get("track_id", -1), -1) for t in top_tracks}

        ranked_events = []
        for ev in movement_events:
            if not isinstance(ev, dict):
                continue
            track_id = _safe_int(ev.get("track_id", -1), -1)
            disp = _safe_float(ev.get("displacement", 0.0))
            speed = _safe_float(ev.get("speed_m_per_s", 0.0)) if ev.get("speed_m_per_s") is not None else 0.0
            score = disp + speed * 2.0
            if track_id in top_track_ids:
                score += 3.0
            ranked_events.append((ev, score))
        ranked_events.sort(key=lambda item: item[1], reverse=True)
        top_events = [item[0] for item in ranked_events[:max_events_per_graph]]

        all_edges: List[Dict[str, Any]] = []
        for frame_key in ("first_frame", "last_frame"):
            edges = spatial.get(frame_key, [])
            if isinstance(edges, list):
                for edge in edges:
                    if isinstance(edge, dict):
                        e = dict(edge)
                        e["_frame"] = frame_key
                        all_edges.append(e)
        all_edges.sort(key=lambda e: _safe_float(e.get("distance", 9999.0), 9999.0))
        top_clearance = all_edges[:max_clearance_edges_per_graph]

        anomalies = [t for t in tracked if isinstance(t, dict) and _is_track_anomaly(t, disp_thr, speed_thr)]
        anomalies.sort(key=lambda t: _safe_float(t.get("total_displacement", 0.0)), reverse=True)
        anomalies = anomalies[:4]

        human_tracks = 0
        moving_human_tracks = 0
        for track in tracked:
            if not isinstance(track, dict):
                continue
            lbl = str(track.get("primary_label", "unknown")).lower()
            if lbl != "human":
                continue
            human_tracks += 1
            state = str(track.get("movement_state", "stationary")).lower()
            if state != "stationary":
                moving_human_tracks += 1

        total_tracks_sum += _safe_int(doc.get("total_tracks", len(tracked)))
        total_moving_sum += _safe_int(doc.get("moving_objects", len(movement_events)))
        total_human_tracks_sum += human_tracks
        doc_end = doc.get("end_dt")
        if doc_end is not None and (latest_doc_end is None or doc_end > latest_doc_end):
            latest_doc_end = doc_end
            latest_doc_stats = {
                "run_name": doc.get("run_name"),
                "human_tracks": human_tracks,
                "moving_human_tracks": moving_human_tracks,
                "total_tracks": _safe_int(doc.get("total_tracks", len(tracked))),
                "moving_objects": _safe_int(doc.get("moving_objects", len(movement_events))),
            }

        context_lines.append(
            f"[Graph {idx}] Run={doc.get('run_name')} "
            f"Range={doc.get('start_label')} -> {doc.get('end_label')} "
            f"Tracks={doc.get('total_tracks')} Moving={doc.get('moving_objects')}"
        )

        if top_tracks:
            context_lines.append("  Top tracks:")
            for track in top_tracks:
                track_id = _safe_int(track.get("track_id", -1), -1)
                lbl = str(track.get("primary_label", "unknown"))
                state = str(track.get("movement_state", "unknown"))
                disp = _safe_float(track.get("total_displacement", 0.0))
                speed = track.get("avg_speed_m_per_s", None)
                speed_txt = f", speed={_safe_float(speed):.3f}m/s" if speed is not None else ""
                context_lines.append(f"  - Track {track_id}: {lbl}, {state}, disp={disp:.2f}m{speed_txt}")

                traj = track.get("trajectory", [])
                norm_traj: List[List[float]] = []
                if isinstance(traj, list):
                    for point in traj:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            x = _safe_float(point[0])
                            y = _safe_float(point[1])
                            z = _safe_float(point[2]) if len(point) >= 3 else 0.0
                            norm_traj.append([x, y, z])
                if len(norm_traj) >= 2 and len(visual_tracks) < 24:
                    visual_tracks.append(
                        {
                            "run_name": str(doc.get("run_name", "")),
                            "track_id": track_id,
                            "label": lbl,
                            "state": state,
                            "displacement": disp,
                            "trajectory": norm_traj[:24],
                        }
                    )

        if top_events:
            context_lines.append("  Velocity temporal edges (movement events):")
            for ev in top_events:
                track_id = _safe_int(ev.get("track_id", -1), -1)
                lbl = str(ev.get("label", "unknown"))
                disp = _safe_float(ev.get("displacement", 0.0))
                speed = ev.get("speed_m_per_s", None)
                state = str(ev.get("state", "unknown"))
                speed_txt = f", { _safe_float(speed):.3f}m/s" if speed is not None else ""
                context_lines.append(f"  - Track {track_id} ({lbl}): {state}, disp={disp:.2f}m{speed_txt}")

        if top_clearance:
            context_lines.append("  Clearance edges (nearest distances):")
            for edge in top_clearance:
                src = _safe_int(edge.get("source", -1), -1)
                tgt = _safe_int(edge.get("target", -1), -1)
                dist = _safe_float(edge.get("distance", 0.0))
                frame_tag = str(edge.get("_frame", "frame"))
                context_lines.append(f"  - {frame_tag}: obj {src} <-> obj {tgt}, dist={dist:.2f}m")

        if anomalies:
            context_lines.append("  Nearby anomaly nodes:")
            for track in anomalies:
                track_id = _safe_int(track.get("track_id", -1), -1)
                lbl = str(track.get("primary_label", "unknown"))
                disp = _safe_float(track.get("total_displacement", 0.0))
                state = str(track.get("movement_state", "unknown"))
                context_lines.append(f"  - Track {track_id}: {lbl}, {state}, disp={disp:.2f}m")

        evidence_lines.append(
            f"{doc.get('run_name')}: {doc.get('start_label')} -> {doc.get('end_label')}, "
            f"tracks={doc.get('total_tracks')}, moving={doc.get('moving_objects')}"
        )
        selected_doc_payloads.append(
            {
                "graph_id": doc.get("graph_id"),
                "run_name": doc.get("run_name"),
                "start_label": doc.get("start_label"),
                "end_label": doc.get("end_label"),
                "total_tracks": doc.get("total_tracks"),
                "moving_objects": doc.get("moving_objects"),
                "human_tracks": human_tracks,
                "moving_human_tracks": moving_human_tracks,
            }
        )

    if not latest_doc_stats and selected_doc_payloads:
        first = selected_doc_payloads[0]
        latest_doc_stats = {
            "run_name": first.get("run_name"),
            "human_tracks": _safe_int(first.get("human_tracks", 0)),
            "moving_human_tracks": _safe_int(first.get("moving_human_tracks", 0)),
            "total_tracks": _safe_int(first.get("total_tracks", 0)),
            "moving_objects": _safe_int(first.get("moving_objects", 0)),
        }

    return {
        "selected_docs": selected_doc_payloads,
        "context_text": "\n".join(context_lines),
        "evidence_lines": evidence_lines,
        "coverage": summarize_temporal_coverage(chosen_docs),
        "aggregate_stats": {
            "runs": len(selected_doc_payloads),
            "total_tracks_sum": total_tracks_sum,
            "total_moving_sum": total_moving_sum,
            "total_human_tracks_sum": total_human_tracks_sum,
            "latest_doc": latest_doc_stats,
        },
        "visual_payload": {
            "title": "Temporal Motion Vector Graph (retrieved evidence)",
            "tracks": visual_tracks,
        },
    }


def _select_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _safe_token_ids(tokenizer):
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if eos_id else None
    if eos_id is not None:
        eos_id = int(eos_id)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id
        try:
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass
    if pad_id is not None:
        pad_id = int(pad_id)
    return eos_id, pad_id


def _get_or_load_model(model_name: str, cache_dir: Path):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    model_kwargs = {
        "torch_dtype": _select_torch_dtype(),
        "cache_dir": str(cache_dir),
    }
    if has_accelerate:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not has_accelerate:
        run_device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(run_device)

    _MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def _generate_response_text(tokenizer, model, messages: List[Dict[str, str]], max_new_tokens: int = 700) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
        text = "\n".join(parts) + "\nassistant:"

    eos_id, pad_id = _safe_token_ids(tokenizer)
    if pad_id is None:
        pad_id = 0

    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = getattr(model, "device", torch.device("cpu"))

    inputs = tokenizer([text], return_tensors="pt").to(model_device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
    new_tokens = generated_ids[0][len(inputs.input_ids[0]) :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _balanced_json_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    start = None
    depth = 0
    in_str = False
    escape = False
    for idx, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : idx + 1])
                    start = None
    return list(reversed(candidates))


def _parse_response_json(text: str) -> Optional[Dict[str, Any]]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    for candidate in _balanced_json_candidates(cleaned):
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        try:
            py_candidate = re.sub(r"\bnull\b", "None", candidate, flags=re.IGNORECASE)
            py_candidate = re.sub(r"\btrue\b", "True", py_candidate, flags=re.IGNORECASE)
            py_candidate = re.sub(r"\bfalse\b", "False", py_candidate, flags=re.IGNORECASE)
            data = ast.literal_eval(py_candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


def _normalize_list_of_strings(values: Any, limit: int = 6) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for item in values:
        text = str(item).strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _fallback_answer(question: str, retrieval: Dict[str, Any]) -> Dict[str, Any]:
    selected_docs = retrieval.get("selected_docs", [])
    evidence = retrieval.get("evidence_lines", [])
    coverage = retrieval.get("coverage", {})
    if not selected_docs:
        answer = "No temporal graph evidence is available for the selected scope."
        confidence = "low"
    else:
        answer = (
            f"I could not generate a reliable free-form LLM response for this question. "
            f"I used {len(selected_docs)} temporal run(s) from {coverage.get('start_label', 'Unknown')} "
            f"to {coverage.get('end_label', 'Unknown')}. "
            f"Please ask a more specific question (count, movement, hazard, or date range)."
        )
        confidence = "low"
    return {
        "answer": answer,
        "highlights": evidence[:4],
        "evidence": evidence[:6],
        "confidence": confidence,
    }


def _deterministic_answer(question: str, retrieval: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q = str(question).strip().lower()
    if not q:
        return None

    selected_docs = retrieval.get("selected_docs", [])
    if not selected_docs:
        return None
    coverage = retrieval.get("coverage", {})
    stats = retrieval.get("aggregate_stats", {})
    latest = stats.get("latest_doc", {}) if isinstance(stats, dict) else {}

    count_query = _contains_any(q, ["how many", "count", "number of", "total"])
    people_query = _contains_any(q, ["people", "person", "human", "worker", "operator"])
    moving_query = _contains_any(q, ["moving", "movement", "move", "velocity", "speed"])
    track_query = _contains_any(q, ["track", "object"])
    date_query = _is_explicit_date_range_question(q) or _contains_any(q, ["what dates", "which dates", "when was", "when is"])
    visual_query = _question_requests_visual(q) and _contains_any(q, ["movement", "moving", "trajectory", "vector", "path", "arrow", "link"])

    if count_query and people_query:
        runs = _safe_int(stats.get("runs", len(selected_docs)))
        latest_humans = _safe_int(latest.get("human_tracks", 0))
        latest_run = str(latest.get("run_name", selected_docs[0].get("run_name", "latest run")))
        total_humans = _safe_int(stats.get("total_human_tracks_sum", 0))
        if runs <= 1:
            answer = f"{latest_humans} human track(s) were detected in the selected temporal run ({latest_run})."
        else:
            answer = (
                f"{total_humans} human track(s) across {runs} selected runs "
                f"(sum of run-local tracks). Latest run {latest_run} has {latest_humans}."
            )
        return {
            "answer": answer,
            "highlights": [
                f"Date range: {coverage.get('start_label', 'Unknown')} -> {coverage.get('end_label', 'Unknown')}",
                f"Latest run: {latest_run}",
            ],
            "evidence": retrieval.get("evidence_lines", [])[:6],
            "confidence": "high",
        }

    if count_query and moving_query:
        runs = _safe_int(stats.get("runs", len(selected_docs)))
        total_moving = _safe_int(stats.get("total_moving_sum", 0))
        latest_moving = _safe_int(latest.get("moving_objects", 0))
        latest_run = str(latest.get("run_name", selected_docs[0].get("run_name", "latest run")))
        if runs <= 1:
            answer = f"{latest_moving} moving object track(s) were detected in the selected temporal run ({latest_run})."
        else:
            answer = (
                f"{total_moving} moving object track(s) across {runs} selected runs "
                f"(sum of run-local counts). Latest run {latest_run} has {latest_moving}."
            )
        return {
            "answer": answer,
            "highlights": [f"Date range: {coverage.get('start_label', 'Unknown')} -> {coverage.get('end_label', 'Unknown')}"],
            "evidence": retrieval.get("evidence_lines", [])[:6],
            "confidence": "high",
        }

    if count_query and track_query:
        runs = _safe_int(stats.get("runs", len(selected_docs)))
        total_tracks = _safe_int(stats.get("total_tracks_sum", 0))
        latest_tracks = _safe_int(latest.get("total_tracks", 0))
        latest_run = str(latest.get("run_name", selected_docs[0].get("run_name", "latest run")))
        if runs <= 1:
            answer = f"{latest_tracks} object track(s) were detected in the selected temporal run ({latest_run})."
        else:
            answer = (
                f"{total_tracks} object track(s) across {runs} selected runs "
                f"(sum of run-local counts). Latest run {latest_run} has {latest_tracks}."
            )
        return {
            "answer": answer,
            "highlights": [f"Date range: {coverage.get('start_label', 'Unknown')} -> {coverage.get('end_label', 'Unknown')}"],
            "evidence": retrieval.get("evidence_lines", [])[:6],
            "confidence": "high",
        }

    if visual_query:
        visual_payload = retrieval.get("visual_payload", {})
        tracks = visual_payload.get("tracks", []) if isinstance(visual_payload, dict) else []
        answer = (
            f"Rendered a motion vector graph with arrows for {len(tracks)} retrieved track trajectories "
            f"from {coverage.get('start_label', 'Unknown')} to {coverage.get('end_label', 'Unknown')}."
        )
        return {
            "answer": answer,
            "highlights": [
                "Arrows indicate movement direction (start -> end).",
                "Lines show track trajectories in XY space.",
            ],
            "evidence": retrieval.get("evidence_lines", [])[:6],
            "confidence": "high",
        }

    if date_query:
        answer = (
            f"Selected temporal data covers {coverage.get('start_label', 'Unknown')} "
            f"to {coverage.get('end_label', 'Unknown')}."
        )
        return {
            "answer": answer,
            "highlights": [f"Runs selected: {len(selected_docs)}"],
            "evidence": retrieval.get("evidence_lines", [])[:6],
            "confidence": "high",
        }
    return None


def _normalize_answer_payload(data: Dict[str, Any], retrieval: Dict[str, Any]) -> Dict[str, Any]:
    answer = str(data.get("answer", "")).strip()
    highlights = _normalize_list_of_strings(data.get("highlights", []), limit=6)
    evidence = _normalize_list_of_strings(data.get("evidence", []), limit=8)
    confidence = str(data.get("confidence", "medium")).lower().strip()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"

    if not answer:
        fallback = _fallback_answer("", retrieval)
        answer = fallback["answer"]
        if not highlights:
            highlights = fallback["highlights"]
        if not evidence:
            evidence = fallback["evidence"]
    if not evidence:
        evidence = retrieval.get("evidence_lines", [])[:6]

    return {
        "answer": answer,
        "highlights": highlights,
        "evidence": evidence,
        "confidence": confidence,
    }


def answer_temporal_question(
    question: str,
    docs: Sequence[Dict[str, Any]],
    selected_graph_ids: Sequence[str],
    model_name: str,
    chat_history: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    question = str(question or "").strip()
    if not question:
        return {
            "answer": "Please enter a question.",
            "highlights": [],
            "evidence": [],
            "confidence": "low",
            "selected_docs": [],
            "coverage": summarize_temporal_coverage([]),
        }

    retrieval = retrieve_temporal_subgraph(
        question=question,
        docs=docs,
        selected_graph_ids=selected_graph_ids,
    )
    if not retrieval.get("selected_docs"):
        fallback = _fallback_answer(question, retrieval)
        fallback["selected_docs"] = []
        fallback["coverage"] = retrieval.get("coverage", summarize_temporal_coverage([]))
        fallback["visual_payload"] = retrieval.get("visual_payload") if _question_requests_visual(question) else None
        return fallback

    deterministic = _deterministic_answer(question, retrieval)
    if deterministic is not None:
        deterministic["selected_docs"] = retrieval.get("selected_docs", [])
        deterministic["coverage"] = retrieval.get("coverage", summarize_temporal_coverage([]))
        deterministic["visual_payload"] = retrieval.get("visual_payload") if _question_requests_visual(question) else None
        return deterministic

    history = list(chat_history or [])[-4:]
    history_lines: List[str] = []
    for turn in history:
        q = str(turn.get("question", "")).strip()
        a = str(turn.get("answer", "")).strip()
        if q and a:
            history_lines.append(f"Q: {q}\nA: {a}")
    history_text = "\n\n".join(history_lines) if history_lines else "None"

    system_prompt = (
        "You are a mining temporal-graph assistant.\n"
        "Use only the provided retrieved evidence.\n"
        "Do not reveal chain-of-thought. Do not include thinking traces.\n"
        "Return ONLY strict JSON with keys:\n"
        '{"answer": string, "highlights": [string], "evidence": [string], "confidence": "low|medium|high"}'
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Conversation context (follow-up history):\n{history_text}\n\n"
        f"Retrieved GraphRAG Evidence:\n{retrieval.get('context_text', '')}\n\n"
        "Answer requirements:\n"
        "- Keep answer concise and actionable.\n"
        "- Mention specific tracks/events when relevant.\n"
        "- If evidence is weak, say uncertainty.\n"
        "- Return JSON only."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    cache_dir = Path.cwd() / "LLM" / "Model_Cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer, model = _get_or_load_model(model_name, cache_dir=cache_dir)
        raw_text = _generate_response_text(tokenizer, model, messages, max_new_tokens=700)
        parsed = _parse_response_json(raw_text)
    except Exception:
        parsed = None

    if parsed is None:
        result = _fallback_answer(question, retrieval)
    else:
        result = _normalize_answer_payload(parsed, retrieval)

    result["selected_docs"] = retrieval.get("selected_docs", [])
    result["coverage"] = retrieval.get("coverage", summarize_temporal_coverage([]))
    result["visual_payload"] = retrieval.get("visual_payload") if _question_requests_visual(question) else None
    return result
