import json
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.base import ClassifierMixin
from transformers import TextClassificationPipeline, pipeline

from src.config.constants import BEREAVEMENT_PATTERNS, LABEL_MAP, IntentEnum
from src.config.schemas import Enrichment, IntentResult, PredictionResponse
from src.config.thresholds import Thresholds, load_thresholds
from src.utils.logger import get_logger
from src.utils.normalise import normalise_text

logger = get_logger(__name__)


class EnsembleClassifier:
    """Classifier combining linear head + enrichment rules."""

    def __init__(self, artifacts_dir: Path, thresholds_path: Optional[Path] = None):
        self.artifacts_dir: Path = artifacts_dir

        # Manifest
        manifest_path = artifacts_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError("Manifest not found in artifacts directory")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())

        # Model + encoder
        self.clf: ClassifierMixin = joblib.load(artifacts_dir / "clf.pkl")
        self.encoder: SentenceTransformer = SentenceTransformer(
            self.manifest["encoder"]
        )

        # Thresholds
        self.thresholds: Thresholds = load_thresholds(thresholds_path)

        # Sentiment
        self.sentiment_pipeline: TextClassificationPipeline = pipeline(
            task="text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

        logger.info("EnsembleClassifier loaded successfully")

    def _enrich(self, intent: IntentEnum, text: str) -> Optional[Enrichment]:
        """Adds enrichment for service feedback & sensitive exits."""
        if intent == IntentEnum.SERVICE_FEEDBACK:
            try:
                res = self.sentiment_pipeline(text, top_k=1)[0]
                return Enrichment(polarity=res["label"].lower())
            except Exception:
                return Enrichment(polarity="neutral")

        if intent == IntentEnum.SENSITIVE_EXIT:
            if BEREAVEMENT_PATTERNS.search(text):
                return Enrichment(sub_reason="bereavement_high_likelihood")
            return Enrichment(sub_reason="opt_out")

        return None

    def classify(self, text: str) -> Tuple[str, float]:
        """Simple interface for Flask endpoint."""
        result = self.predict(text)
        best = result.intents[0]
        return best.label, best.probability

    def predict(self, text: str, top_k: int = 2) -> PredictionResponse:
        """Full prediction with thresholds + enrichment."""
        norm_text = normalise_text(text)
        emb = self.encoder.encode([norm_text], show_progress_bar=False)
        clf_probs: NDArray[np.float64] = self.clf.predict_proba(emb)[0]

        candidates: list[IntentResult] = []
        for i, prob in enumerate(clf_probs):
            intent = list(IntentEnum)[i]
            thr = getattr(self.thresholds, intent.value, 0.5)
            if prob >= thr:
                candidates.append(
                    IntentResult(
                        label=LABEL_MAP[intent],
                        key=intent,
                        probability=round(float(prob), 4),
                        enrichment=self._enrich(intent, text),
                    )
                )

        # fallback if no candidates
        if not candidates:
            i = int(np.argmax(clf_probs))
            intent = list(IntentEnum)[i]
            candidates.append(
                IntentResult(
                    label=LABEL_MAP[intent],
                    key=intent,
                    probability=round(float(clf_probs[i]), 4),
                    enrichment=self._enrich(intent, text),
                )
            )

        # sort + trim
        candidates = sorted(candidates, key=lambda x: x.probability, reverse=True)[
            :top_k
        ]

        review_status = "CLASSIFIED"
        if candidates[0].probability < self.thresholds.review_band:
            review_status = "NEEDS_REVIEW"

        return PredictionResponse(
            model_version=self.manifest["version"],
            intents=candidates,
            review_status=review_status,
        )
