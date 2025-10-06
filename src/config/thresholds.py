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
    review_band: float = Field(0.40, ge=0.0, le=1.0)

    @classmethod
    def defaults(cls) -> Thresholds:
        """Explicit constructor for defaults (mypy-friendly)."""
        return cls(
            service_feedback=0.55,
            sensitive_exit=0.45,
            other=0.50,
            noise=0.70,
            review_band=0.40,
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
        return Thresholds.defaults()
    p = Path(path)
    if not p.exists():
        return Thresholds.defaults()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return Thresholds(**data)
    except (ValidationError, json.JSONDecodeError):
        return Thresholds.defaults()
