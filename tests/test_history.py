"""Tests for GET /history."""

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import Prediction


def _add(session: Session, filename: str, minutes_ago: int) -> None:
    session.add(
        Prediction(
            filename=filename,
            question="Bu yemek nedir?",
            answer="Bu, lahmacun.",
            duration_ms=100,
            created_at=datetime.now(UTC) - timedelta(minutes=minutes_ago),
        )
    )
    session.flush()


def test_returns_an_empty_list_when_nothing_is_stored(client: TestClient) -> None:
    response = client.get("/history")

    assert response.status_code == 200
    assert response.json() == []


def test_returns_newest_first(client: TestClient, session: Session) -> None:
    _add(session, "oldest.png", minutes_ago=30)
    _add(session, "newest.png", minutes_ago=1)
    _add(session, "middle.png", minutes_ago=10)

    names = [row["filename"] for row in client.get("/history").json()]

    assert names == ["newest.png", "middle.png", "oldest.png"]


def test_limit_and_offset_page_through_results(
    client: TestClient, session: Session
) -> None:
    for index in range(5):
        _add(session, f"{index}.png", minutes_ago=index)

    first_page = client.get("/history", params={"limit": 2}).json()
    second_page = client.get("/history", params={"limit": 2, "offset": 2}).json()

    assert [row["filename"] for row in first_page] == ["0.png", "1.png"]
    assert [row["filename"] for row in second_page] == ["2.png", "3.png"]


def test_rejects_an_out_of_range_limit(client: TestClient) -> None:
    """The ceiling stops a single request from asking for the whole table."""
    assert client.get("/history", params={"limit": 0}).status_code == 422
    assert client.get("/history", params={"limit": 101}).status_code == 422
