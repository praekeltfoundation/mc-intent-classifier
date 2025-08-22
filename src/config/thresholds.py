import json
from pathlib import Path

from pydantic import BaseModel


class Thresholds(BaseModel):
    service_feedback: float = 0.55
    sensitive_exit: float = 0.45
    other: float = 0.50
    noise: float = 0.70
    review_band: float = 0.40


def load_thresholds(path: Path | None = None) -> Thresholds:
    """Load thresholds from JSON or fallback to defaults."""
    if path and path.exists():
        return Thresholds(**json.loads(path.read_text()))
    return Thresholds()
