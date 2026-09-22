"""Shared test fixtures.

The tests run against the real PostgreSQL service rather than an in-memory
substitute, because the schema depends on behaviour SQLite does not reproduce:
timezone-aware timestamps and enforced VARCHAR lengths. Testing against SQLite
would turn those two deliberate decisions into untested assumptions.

The model, by contrast, is always replaced with a stub. Loading it would cost
4.4 GB and a minute per test run, and what these tests check is the behaviour of
the API around the model, not the quality of its answers.

Run them with:

    docker compose run --rm test
"""

from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import delete
from sqlalchemy.orm import Session

import app.main as main_module
from app.database import Base, get_engine, get_session
from app.main import app, get_model
from app.models import Prediction


def make_image_bytes(size: tuple[int, int] = (64, 64), fmt: str = "PNG") -> bytes:
    """Return the encoded bytes of a small solid-colour test image."""
    buffer = BytesIO()
    Image.new("RGB", size, (200, 150, 100)).save(buffer, format=fmt)
    return buffer.getvalue()


class StubModel:
    """Stands in for DietitianModel with the same call signature."""

    def __init__(self, answer: str = "Bu, lahmacun.", duration_ms: int = 42) -> None:
        self._answer = answer
        self._duration_ms = duration_ms
        self.calls: list[tuple[tuple[int, int], str]] = []

    def answer(self, image: Image.Image, question: str) -> tuple[str, int]:
        self.calls.append((image.size, question))
        return self._answer, self._duration_ms


@pytest.fixture(scope="session", autouse=True)
def _tables() -> None:
    """Make sure the schema exists before any test runs."""
    Base.metadata.create_all(bind=get_engine())


@pytest.fixture()
def session() -> Iterator[Session]:
    """Yield a Session whose every write is rolled back afterwards.

    The session is bound to a connection with an open transaction, and
    join_transaction_mode="create_savepoint" means the application's own
    commit() releases a savepoint instead of ending that outer transaction.
    Nothing a test writes — including the initial clear-out below — survives it,
    so tests are isolated from each other and from any real data in the database.
    """
    connection = get_engine().connect()
    transaction = connection.begin()
    db = Session(
        bind=connection,
        autoflush=False,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    # Start every test from an empty table without actually deleting anything:
    # this statement is part of the transaction that gets rolled back.
    db.execute(delete(Prediction))
    db.flush()
    try:
        yield db
    finally:
        db.close()
        transaction.rollback()
        connection.close()


@pytest.fixture()
def client(session: Session, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient wired to the throwaway session.

    TestClient is deliberately not used as a context manager: entering one runs
    the lifespan hook, which would load the real model.
    """

    @contextmanager
    def _test_session_scope() -> Iterator[Session]:
        yield session

    # POST /predict writes through session_scope() rather than through a
    # dependency, so that it does not hold a pooled connection during
    # generation. That is the right call for production and the one place here
    # that cannot be redirected with dependency_overrides.
    monkeypatch.setattr(main_module, "session_scope", _test_session_scope)
    app.dependency_overrides[get_session] = lambda: session

    yield TestClient(app)

    app.dependency_overrides.clear()


@pytest.fixture()
def stub_model() -> Iterator[StubModel]:
    """Install a stub model. Tests that omit this fixture see an unloaded service."""
    model = StubModel()
    app.dependency_overrides[get_model] = lambda: model
    yield model
    app.dependency_overrides.pop(get_model, None)
