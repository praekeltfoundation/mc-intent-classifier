# src/config/thresholds.py
"""Minimal, Pydantic-based confidence thresholds for the intent classifier."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from src.config.schemas import ParentLabel


class Thresholds(BaseModel):
    """Minimal acceptance thresholds and review band."""

    service_feedback: float = Field(0.55, ge=0.0, le=1.0)
    sensitive_exit: float = Field(0.45, ge=0.0, le=1.0)
    other: float = Field(0.50, ge=0.0, le=1.0)
    noise: float = Field(0.70, ge=0.0, le=1.0)
    review_band: float = Field(0.40, ge=0.0, le=1.0, alias="review_band")
    sentiment_review_band: float = Field(
        0.75, ge=0.0, le=1.0, alias="sentiment_review_band"
    )

    @classmethod
    def defaults(cls) -> Thresholds:
        """Explicit constructor for defaults (mypy-friendly)."""
        return cls(
            service_feedback=0.55,
            sensitive_exit=0.45,
            other=0.50,
            noise=0.70,
            review_band=0.40,
            sentiment_review_band=0.75,
        )

    def for_parent(self, parent: ParentLabel | str) -> float:
        """Return the acceptance cutoff for the given parent family."""
        if parent == "FEEDBACK":
            return self.service_feedback
        if parent == "SENSITIVE_EXIT":
            return self.sensitive_exit
        if parent == "NOISE_SPAM":
            return self.noise
        return self.other


def load_thresholds(path: Path | str | None) -> Thresholds:
    """Load thresholds from JSON, or return defaults if missing/invalid."""
    if not path:
        # print("DEBUG: No path provided, using defaults.") # Optional Debug
        return Thresholds.defaults()
    p = Path(path)
    if not p.exists():
        # print(f"DEBUG: Path {p} not found, using defaults.") # Optional Debug
        return Thresholds.defaults()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # print(f"DEBUG: Loaded data from {p}: {data}") # Optional Debug

        # --- Handle potential structure mismatch ---
        # If your JSON has {"per_parent": {...}, "review_band": ..., "sentiment_review_band": ...}
        # you need to flatten 'per_parent' before passing to Pydantic

        if "per_parent" in data:
            parent_thresholds = data.pop("per_parent")
            # Map keys based on your for_parent logic
            data["service_feedback"] = parent_thresholds.get("FEEDBACK", 0.55)
            data["sensitive_exit"] = parent_thresholds.get("SENSITIVE_EXIT", 0.45)
            data["noise"] = parent_thresholds.get("NOISE_SPAM", 0.70)
            data["other"] = parent_thresholds.get(
                "OTHER", 0.50
            )  # Assuming 'other' maps to 'OTHER' intent

        # print(f"DEBUG: Data after potential mapping: {data}") # Optional Debug
        return Thresholds(**data)

    except (ValidationError, json.JSONDecodeError, TypeError):
        # print(f"DEBUG: Error loading thresholds: {e}, using defaults.") # Optional Debug
        # Optionally log the error here
        return Thresholds.defaults()
