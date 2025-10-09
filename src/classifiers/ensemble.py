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

from src.config.schemas import Enrichment, IntentResult, ParentLabel, PredictionResponse
from src.config.thresholds import load_thresholds
from src.utils.logger import get_logger
from src.utils.normalise import normalise_text

logger = get_logger(__name__)


class EnsembleClassifier:
    """Hierarchical classifier combining a parent model and specialized sub-models."""

    def __init__(self, artifacts_dir: Path, thresholds_path: Path | None = None):
        self.artifacts_dir = artifacts_dir
        self.manifest = self._load_json(artifacts_dir / "manifest.json")
        self.thresholds = load_thresholds(thresholds_path)

        # Load main sentence encoder
        self.encoder: SentenceTransformer = SentenceTransformer(
            self.manifest["encoder"]
        )

        # Load parent classifier
        self.clf_parent: ClassifierMixin = joblib.load(artifacts_dir / "clf_parent.pkl")
        self.le_parent: LabelEncoder = joblib.load(artifacts_dir / "le_parent.pkl")

        # Load specialized sub-intent models
        self.clf_sensitive_exit: ClassifierMixin | None = self._load_optional_model(
            "clf_sensitive_exit.pkl"
        )
        self.le_sensitive_exit: LabelEncoder | None = self._load_optional_model(
            "le_sensitive_exit.pkl"
        )

        # Load pre-trained multilingual sentiment model for FEEDBACK
        self.sentiment_pipeline: TextClassificationPipeline = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        )

    def _load_optional_model(self, filename: str) -> Any | None:
        """Load a model artifact if it exists, otherwise return None."""
        model_path = self.artifacts_dir / filename
        if model_path.exists():
            return joblib.load(model_path)
        logger.warning(f"Optional model artifact not found: {filename}")
        return None

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Safely loads a JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found at {path}")
        return cast(dict[str, Any], json.loads(path.read_text()))

    def _enrich(
        self, parent: str, text: str, embedding: NDArray[np.float64]
    ) -> Enrichment:
        """Determines the sub-intent using the appropriate specialist model."""
        if parent == "FEEDBACK":
            # Using pre-trained multilingual model for robust sentiment analysis
            sentiment = self.sentiment_pipeline(text, top_k=1)[0]
            # Map 'positive' to COMPLIMENT, and 'neutral'/'negative' to COMPLAINT
            sub_reason = (
                "COMPLIMENT" if sentiment["label"] == "positive" else "COMPLAINT"
            )
            return Enrichment(
                sub_reason=sub_reason,
                score=sentiment["score"],
                sentiment_label=sentiment["label"],
            )

        if (
            parent == "SENSITIVE_EXIT"
            and self.clf_sensitive_exit
            and self.le_sensitive_exit
        ):
            # Using our custom-trained, specialized ML model
            sub_pred_idx = self.clf_sensitive_exit.predict(embedding)[0]
            sub_reason = self.le_sensitive_exit.inverse_transform([sub_pred_idx])[0]
            sub_prob = self.clf_sensitive_exit.predict_proba(embedding)[0][sub_pred_idx]
            return Enrichment(sub_reason=sub_reason, score=float(sub_prob))

        return Enrichment()

    def predict(self, text: str) -> PredictionResponse:
        """Predicts intent using a hierarchical approach."""
        norm_text = normalise_text(text)
        embedding: NDArray[np.float64] = self.encoder.encode([norm_text])

        # 1. Get parent probabilities
        probabilities: NDArray[np.float64] = self.clf_parent.predict_proba(embedding)[0]

        # 2. Apply "default-to-OTHER" logic based on thresholds
        confident_preds: list[tuple[ParentLabel, float]] = []
        for i, class_name in enumerate(self.le_parent.classes_):
            prob = probabilities[i]
            if class_name != "OTHER" and prob >= self.thresholds.for_parent(class_name):
                confident_preds.append((class_name, prob))

        if confident_preds:
            # If specific intents meet thresholds, pick the highest scoring one
            parent_label, parent_prob = max(confident_preds, key=lambda item: item[1])
        else:
            # Otherwise, default to OTHER
            other_idx = np.where(self.le_parent.classes_ == "OTHER")[0][0]
            parent_label = "OTHER"
            parent_prob = probabilities[other_idx]

        # 3. Get sub-intent from the appropriate specialist
        enrichment = self._enrich(parent_label, norm_text, embedding)

        final_pred = IntentResult(
            label=parent_label,
            key=parent_label,
            probability=round(float(parent_prob), 4),
            enrichment=enrichment,
        )

        # 4. Determine review status using the new two-rule system
        is_low_confidence = final_pred["probability"] < self.thresholds.review_band
        is_neutral_feedback = (
            final_pred["key"] == "FEEDBACK"
            and final_pred["enrichment"].get("sentiment_label") == "neutral"
        )

        review_status: Literal["CLASSIFIED", "NEEDS_REVIEW"] = (
            "NEEDS_REVIEW" if is_low_confidence or is_neutral_feedback else "CLASSIFIED"
        )

        return PredictionResponse(
            model_version=self.manifest["version"],
            intents=[final_pred],
            review_status=review_status,
        )
