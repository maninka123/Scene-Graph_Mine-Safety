from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_demo_contract_and_real_assets():
    response = client.get("/api/demo")
    assert response.status_code == 200
    payload = response.json()
    assert payload["demo"]["point_count"] == 48_000
    assert payload["demo"]["id"] == "PC_20260122_153641662"
    assert len(payload["point_cloud"]["positions"]) == 12_000
    assert len(payload["nodes"]) == 8
    assert any(stage["id"] == "minkunet" for stage in payload["pipeline"])


def test_live_scene_graph_and_rule_analysis():
    demo = client.get("/api/demo").json()
    equipment = next(node for node in demo["nodes"] if node["label"] == "equipment")
    worker = {
        "id": "personnel-test-1",
        "label": "personnel",
        "centroid": equipment["centroid"],
        "bbox_dimensions": [0.5, 0.5, 1.75],
        "orientation": [1, 0, 0],
        "volume_m3": 0.44,
        "voxel_count": 1,
        "confidence": 1,
        "entropy": 0,
        "is_anomaly": False,
        "active": True,
        "velocity": [0, 0, 0],
        "movement_state": "stationary",
    }
    response = client.post(
        "/api/analyse",
        json={"nodes": [*demo["nodes"], worker], "edge_distance_m": 2.5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert any(alert["rule"] == "proximity_violation" for alert in payload["alerts"])
    assert payload["live_timings"]["total_postprocess_ms"] >= 0


def test_qwen_status_contract_does_not_load_model():
    response = client.get("/api/model/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == "Qwen/Qwen2.5-3B-Instruct"
    assert payload["prompt_profile"] == "Appendix A contextual safety reasoning"
    assert "cuda_available" in payload
    assert "loaded" in payload


def test_perception_upload_status_contract():
    response = client.get("/api/perception/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["checkpoint"] == "legacy_six_class/semantic_best.pt"
    assert payload["classes"] == ["wall", "equipment", "human", "conveyor", "roof", "other"]
    assert payload["paper_checkpoint"] is False
    assert ".pcd" in payload["accepted_formats"]


def test_point_cloud_upload_rejects_unsupported_files():
    response = client.post(
        "/api/point-clouds",
        files={"file": ("not-a-cloud.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 422
    assert ".pcd" in response.json()["detail"]
