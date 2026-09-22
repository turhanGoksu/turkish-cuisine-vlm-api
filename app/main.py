"""FastAPI application exposing the Turkish cuisine vision-language model.

Endpoints:
    GET  /health   Liveness and model-readiness information.
    POST /predict  Upload a food image and get a Turkish answer about it.

GET /history is added once the database layer is in place.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from app.ml import DEFAULT_QUESTION, DietitianModel, InvalidImageError, decode_image
from app.schemas import HealthResponse, PredictionResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# Images are read fully into memory before decoding, so the limit is what keeps a
# single upload from exhausting the container's RAM.
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup and release it at shutdown.

    Loading here rather than at import time keeps importing this module cheap, which
    matters for tests, and lets /health report readiness truthfully. Anything after
    the yield runs during shutdown.
    """
    logger.info("Starting up: loading model (this takes a while on CPU)")
    app.state.model = DietitianModel.load()
    yield
    logger.info("Shutting down: releasing model")
    app.state.model = None


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
    """Answer a question about an uploaded food image.

    Declared with a plain def rather than async def: generation is CPU-bound, so
    FastAPI runs this in its thread pool instead of blocking the event loop.
    """
    # Read at most one byte past the limit: enough to detect an oversized upload
    # without ever holding the whole of it in memory.
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"The image must be smaller than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    try:
        # The client-supplied content type is only a hint; decoding is the real check.
        image = decode_image(data)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    answer, duration_ms = model.answer(image=image, question=question)
    logger.info("Answered %s in %d ms", file.filename, duration_ms)

    return PredictionResponse(
        filename=file.filename or "unnamed",
        question=question,
        answer=answer,
        duration_ms=duration_ms,
    )
