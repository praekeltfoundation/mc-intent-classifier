# src/classifiers/ensemble.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import joblib
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from transformers import TextClassificationPipeline, pipeline

from src.config.constants import BEREAVEMENT_PATTERNS, OPTOUT_PATTERNS
from src.config.schemas import Enrichment, IntentResult, ParentLabel, PredictionResponse
from src.config.thresholds import load_thresholds
from src.utils.logger import get_logger
from src.utils.normalise import normalise_text

logger = get_logger(__name__)


class EnsembleClassifier:
    """Classifier combining a dense encoder, ML head, and rule-based enrichment."""

    def __init__(self, artifacts_dir: Path, thresholds_path: Path | None = None):
        self.artifacts_dir = artifacts_dir
        self.manifest = self._load_json(artifacts_dir / "manifest.json")
        self.thresholds = load_thresholds(thresholds_path)
        self.encoder: SentenceTransformer = SentenceTransformer(
            self.manifest["encoder"]
        )
        self.clf: ClassifierMixin = joblib.load(artifacts_dir / "clf.pkl")
        self.label_encoder: LabelEncoder = joblib.load(
            artifacts_dir / "label_encoder.pkl"
        )
        self.sentiment_pipeline: TextClassificationPipeline = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Safely loads a JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found at {path}")
        return cast(dict[str, Any], json.loads(path.read_text()))

    def _enrich(self, parent: str, text: str) -> Enrichment:
        """Determines the sub-intent based on the predicted parent label."""
        if parent == "FEEDBACK":
            sentiment = self.sentiment_pipeline(text, top_k=1)[0]
            sub_reason = (
                "COMPLIMENT" if sentiment["label"] == "POSITIVE" else "COMPLAINT"
            )
            return Enrichment(sub_reason=sub_reason, score=sentiment["score"])

        if parent == "SENSITIVE_EXIT":
            if any(p.search(text) for p in BEREAVEMENT_PATTERNS):
                return Enrichment(sub_reason="BABY_LOSS")
            if any(p.search(text) for p in OPTOUT_PATTERNS):
                return Enrichment(sub_reason="OPTOUT")
        return Enrichment()

    def predict(self, text: str, top_k: int = 1) -> PredictionResponse:
        """Predicts the intent and sub-intent for a given text."""
        norm_text = normalise_text(text)
        embedding: NDArray[np.float64] = self.encoder.encode([norm_text])
        probabilities: NDArray[np.float64] = self.clf.predict_proba(embedding)[0]
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        candidates: list[IntentResult] = []
        for i in top_indices:
            label: ParentLabel = self.label_encoder.classes_[i]
            prob = probabilities[i]
            if prob >= self.thresholds.for_parent(label):
                candidates.append(
                    IntentResult(
                        label=label,
                        key=label,
                        probability=round(float(prob), 4),
                        enrichment=self._enrich(label, norm_text),
                    )
                )
        if not candidates:
            top_index = np.argmax(probabilities)
            label = self.label_encoder.classes_[top_index]
            prob = probabilities[top_index]
            candidates.append(
                IntentResult(
                    label=label,
                    key=label,
                    probability=round(float(prob), 4),
                    enrichment=self._enrich(label, norm_text),
                )
            )

        review_status: Literal["CLASSIFIED", "NEEDS_REVIEW"] = (
            "NEEDS_REVIEW"
            if candidates[0]["probability"] < self.thresholds.review_band
            else "CLASSIFIED"
        )
        return PredictionResponse(
            model_version=self.manifest["version"],
            intents=candidates,
            review_status=review_status,
        )
