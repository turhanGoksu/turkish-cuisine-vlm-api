"""FastAPI application exposing the Turkish cuisine vision-language model.

Endpoints:
    GET  /health   Liveness and model-readiness information.
    POST /predict  Upload a food image and get a Turkish answer about it.
    GET  /history  List previously stored predictions, newest first.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import Base, get_engine, get_session, session_scope
from app.ml import DEFAULT_QUESTION, DietitianModel, InvalidImageError, decode_image
from app.models import Prediction
from app.schemas import HealthResponse, PredictionRecord, PredictionResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# Images are read fully into memory before decoding, so the limit is what keeps a
# single upload from exhausting the container's RAM.
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare the database and load the model once, before serving traffic.

    The database is set up first because it fails in milliseconds if it is
    misconfigured, whereas loading the model takes a minute or more. Failing fast
    beats discovering a wrong connection string after a long startup.
    """
    logger.info("Starting up: creating tables if they do not exist")
    Base.metadata.create_all(bind=get_engine())

    logger.info("Starting up: loading model (this takes a while on CPU)")
    app.state.model = DietitianModel.load()

    yield

    logger.info("Shutting down: releasing model and database connections")
    app.state.model = None
    get_engine().dispose()


app = FastAPI(
    title="Turkish Cuisine VLM API",
    description=(
        "Serves a LoRA-adapted Qwen2-VL model that recognises Turkish dishes and "
        "answers nutrition questions about them in Turkish."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def get_model(request: Request) -> DietitianModel:
    """Return the loaded model, or fail with 503 if startup has not finished."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model is not loaded yet. Try again shortly.",
        )
    return model


def store_prediction(record: Prediction) -> None:
    """Persist one prediction, logging but not raising on failure.

    Writing to the history table is a side effect, not what the caller asked for.
    An answer that took a minute of CPU to produce must not be thrown away because
    a secondary write failed.
    """
    try:
        with session_scope() as session:
            session.add(record)
            session.commit()
    except SQLAlchemyError:
        logger.exception("Could not store prediction for %s", record.filename)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    """Report whether the service is up and whether the model is ready.

    Declared async on purpose: it does no blocking work, so it stays on the event
    loop and keeps answering even while the thread pool is busy with inference.
    """
    return HealthResponse(
        status="ok",
        model_loaded=getattr(request.app.state, "model", None) is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(
    file: UploadFile = File(..., description="A photo of a Turkish dish."),
    question: str = Form(
        DEFAULT_QUESTION, description="What to ask about the image, in Turkish."
    ),
    model: DietitianModel = Depends(get_model),
) -> PredictionResponse:
    """Answer a question about an uploaded food image and record the answer.

    Declared with a plain def rather than async def: generation is CPU-bound, so
    FastAPI runs this in its thread pool instead of blocking the event loop.

    Note that no database session is injected here. A session holds a pooled
    connection, and holding one open for the minute that generation takes would
    exhaust the pool under very little load. The session is opened afterwards,
    only for the few milliseconds the write needs.
    """
    # Read at most one byte past the limit: enough to detect an oversized upload
    # without ever holding the whole of it in memory.
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"The image must be smaller than {limit_mb} MB.",
        )

    try:
        # The client-supplied content type is only a hint; decoding is the real check.
        image = decode_image(data)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    answer, duration_ms = model.answer(image=image, question=question)
    filename = file.filename or "unnamed"
    logger.info("Answered %s in %d ms", filename, duration_ms)

    store_prediction(
        Prediction(
            filename=filename,
            question=question,
            answer=answer,
            duration_ms=duration_ms,
        )
    )

    return PredictionResponse(
        filename=filename,
        question=question,
        answer=answer,
        duration_ms=duration_ms,
    )


@app.get("/history", response_model=list[PredictionRecord], tags=["history"])
def history(
    limit: int = Query(20, ge=1, le=100, description="How many records to return."),
    offset: int = Query(0, ge=0, description="How many records to skip."),
    session: Session = Depends(get_session),
) -> list[Prediction]:
    """List stored predictions, newest first.

    A plain def again: the database driver is synchronous, so this blocks. Unlike
    /predict it is fast, which is why injecting the session per request is fine here.
    """
    statement = (
        select(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    # Returning ORM objects directly works because PredictionRecord is configured
    # with from_attributes=True; FastAPI reads the attributes off each row.
    return list(session.scalars(statement))
