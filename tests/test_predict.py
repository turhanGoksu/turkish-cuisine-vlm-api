"""Tests for POST /predict."""

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.main import MAX_UPLOAD_BYTES
from app.ml import DEFAULT_QUESTION
from app.models import Prediction

from .conftest import StubModel, make_image_bytes


def test_returns_the_model_answer(client: TestClient, stub_model: StubModel) -> None:
    response = client.post(
        "/predict", files={"file": ("kebap.png", make_image_bytes(), "image/png")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "kebap.png",
        "question": DEFAULT_QUESTION,
        "answer": "Bu, lahmacun.",
        "duration_ms": 42,
    }


def test_passes_a_custom_question_to_the_model(
    client: TestClient, stub_model: StubModel
) -> None:
    response = client.post(
        "/predict",
        files={"file": ("kebap.png", make_image_bytes(), "image/png")},
        data={"question": "Kaç kalori?"},
    )

    assert response.status_code == 200
    assert response.json()["question"] == "Kaç kalori?"
    assert stub_model.calls == [((64, 64), "Kaç kalori?")]


def test_stores_the_prediction(
    client: TestClient, stub_model: StubModel, session: Session
) -> None:
    client.post(
        "/predict", files={"file": ("kebap.png", make_image_bytes(), "image/png")}
    )

    stored = session.scalars(select(Prediction)).all()
    assert len(stored) == 1
    assert stored[0].filename == "kebap.png"
    assert stored[0].answer == "Bu, lahmacun."
    assert stored[0].created_at is not None


def test_rejects_a_file_that_is_not_an_image(
    client: TestClient, stub_model: StubModel
) -> None:
    """A broken upload is the caller's mistake, so it must be a 400 and not a 500."""
    response = client.post(
        "/predict", files={"file": ("notes.png", b"this is not an image", "image/png")}
    )

    assert response.status_code == 400


def test_rejects_an_oversized_upload(client: TestClient, stub_model: StubModel) -> None:
    oversized = b"\0" * (MAX_UPLOAD_BYTES + 1)

    response = client.post(
        "/predict", files={"file": ("huge.png", oversized, "image/png")}
    )

    assert response.status_code == 413


def test_reports_503_while_the_model_is_still_loading(client: TestClient) -> None:
    """No stub_model fixture here, so the service looks like it is still starting."""
    response = client.post(
        "/predict", files={"file": ("kebap.png", make_image_bytes(), "image/png")}
    )

    assert response.status_code == 503
