"""Database engine, session factory, and the FastAPI session dependency.

Two objects with very different lifetimes live here:

* the Engine owns the connection pool and is created once per process;
* a Session is a unit of work and is created once per request, because a Session
  is not thread-safe and carries an open transaction that must not be shared.

Both are built lazily rather than at import time, so importing this module never
opens a socket and tests can import it without a database running.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Base class every ORM model inherits from; it collects the table metadata."""


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create the process-wide engine, or return the one already created.

    Raises:
        RuntimeError: If DATABASE_URL is not configured.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. docker-compose provides it; see .env.example."
        )

    return create_engine(
        url,
        # Checks a pooled connection is still alive before handing it out. Without
        # this, restarting the database container leaves the pool full of dead
        # sockets and the next few requests fail with an obscure connection error.
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide factory that produces per-request sessions."""
    return sessionmaker(
        bind=get_engine(),
        autoflush=False,
        # Keeps attributes readable after commit. Without this, SQLAlchemy expires
        # every attribute on commit and reading row.id afterwards would fire another
        # query — or fail outright once the session is closed.
        expire_on_commit=False,
    )


@contextmanager
def session_scope() -> Iterator[Session]:
    """Open a Session and guarantee it is closed again.

    Used where a session is needed for a short moment rather than for the whole
    request, so that a pooled connection is not held open during slow work.
    """
    session = get_session_factory()()
    try:
        yield session
    finally:
        # Returns the connection to the pool. Skipping this leaks connections until
        # the pool is exhausted and every request hangs waiting for a free one.
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding one Session per request, always closed."""
    with session_scope() as session:
        yield session
