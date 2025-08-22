from pydantic import BaseModel

from src.config.constants import IntentEnum


class Enrichment(BaseModel):
    polarity: str | None = None
    sub_reason: str | None = None


class IntentResult(BaseModel):
    label: str
    key: IntentEnum
    probability: float
    enrichment: Enrichment | None = None


class PredictionResponse(BaseModel):
    model_version: str
    language: str = "en"
    intents: list[IntentResult]
    review_status: str
