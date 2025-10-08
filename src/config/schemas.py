# src/config/schemas.py
"""Data schemas and type definitions for the application."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

# --- Type definitions for labels ---
ParentLabel = Literal["FEEDBACK", "SENSITIVE_EXIT", "NOISE_SPAM", "OTHER"]
FeedbackSubtype = Literal["COMPLIMENT", "COMPLAINT"]
SensitiveExitSubtype = Literal["BABY_LOSS", "OPTOUT"]
OtherSubtype = Literal["ACCOUNT_UPDATE", "INFORMATION_QUERY", "CONFIRMATION"]


# --- Data structure for training samples ---
class SampleRow(TypedDict):
    """Defines the structure of a row in the JSONL training data."""

    text: str
    label: ParentLabel
    feedback_subtype: NotRequired[FeedbackSubtype]
    sensitive_exit_subtype: NotRequired[SensitiveExitSubtype]
    other_subtype: NotRequired[OtherSubtype]


# --- API Response Schemas (used by the classifier) ---
class Enrichment(TypedDict, total=False):
    """Holds enrichment data, like subtype or sentiment score."""

    sub_reason: str
    score: float


class IntentResult(TypedDict):
    """Represents a single predicted intent."""

    label: str
    key: ParentLabel | str
    probability: float
    enrichment: Enrichment


class PredictionResponse(TypedDict):
    """The final structure of a prediction returned by the classifier."""

    model_version: str
    intents: list[IntentResult]
    review_status: Literal["CLASSIFIED", "NEEDS_REVIEW"]
