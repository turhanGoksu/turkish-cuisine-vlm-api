"""Tests for GET /health."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_reports_no_model_before_startup(client: TestClient) -> None:
    """Without the lifespan hook having run, the model is honestly reported missing."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_health_reports_a_loaded_model(client: TestClient) -> None:
    """Once something is in app.state, readiness flips to true."""
    app.state.model = object()
    try:
        response = client.get("/health")
    finally:
        app.state.model = None

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
