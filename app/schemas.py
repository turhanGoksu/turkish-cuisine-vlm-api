"""Pydantic schemas describing the JSON contract of the API.

These models are the single source of truth for three things at once: runtime
validation of outgoing data, JSON serialisation, and the OpenAPI documentation
served at /docs. Note that the uploaded image itself is *not* described here —
it arrives as multipart/form-data and is handled by FastAPI's UploadFile.

The served model is a generative vision-language model, not a classifier, so a
prediction is free-form Turkish text rather than a label with a probability.
Exposing a "confidence" score would mean inventing a number the model never
produces.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

_ANSWER_EXAMPLE = (
    "Bu yemek Adana Kebap'tır. Yaklaşık 450 kalori içerir ve protein açısından "
    "zengindir, ancak yağ oranı yüksek olduğu için porsiyon kontrolüne dikkat edin."
)


class HealthResponse(BaseModel):
    """Liveness and readiness information for the service."""

    status: str = Field(description="Always 'ok' when the service is able to respond.")
    model_loaded: bool = Field(
        description="True once the vision-language model is loaded into memory."
    )

    model_config = ConfigDict(
        # 'model_' is a protected prefix in Pydantic v2; allow it for model_loaded.
        protected_namespaces=(),
        json_schema_extra={"example": {"status": "ok", "model_loaded": True}},
    )


class PredictionResponse(BaseModel):
    """The result returned by POST /predict for a single uploaded image."""

    filename: str = Field(description="Original name of the uploaded file.")
    question: str = Field(description="The question that was asked about the image.")
    answer: str = Field(description="The model's answer, in Turkish.")
    duration_ms: int = Field(
        ge=0,
        description=(
            "Wall-clock inference time in milliseconds. Recorded because the model "
            "runs on CPU, where generation is slow enough to be worth measuring."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "kebap.jpg",
                "question": "Bu yemek nedir ve besin değerleri nasıldır?",
                "answer": _ANSWER_EXAMPLE,
                "duration_ms": 41230,
            }
        }
    )


class PredictionRecord(BaseModel):
    """A stored prediction as returned by GET /history."""

    id: int = Field(description="Database primary key of the stored prediction.")
    filename: str
    question: str
    answer: str
    duration_ms: int = Field(ge=0)
    created_at: datetime = Field(description="UTC timestamp of when the row was written.")

    # Lets FastAPI build this schema straight from a SQLAlchemy row object
    # (reading attributes) instead of requiring a dict.
    model_config = ConfigDict(from_attributes=True)
