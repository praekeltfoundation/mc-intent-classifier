# src/application.py
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

import sentry_sdk
from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue
from flask_basicauth import BasicAuth
from prometheus_flask_exporter import PrometheusMetrics

from src.classifiers.ensemble import EnsembleClassifier
from src.utils.logger import get_logger
from src.utils.normalise import normalise_text

# --- Setup ---
version = importlib.metadata.version("mc-intent-classifier")
logger = get_logger(__name__)
ROOT_DIR = Path(__file__).parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
THRESHOLDS_PATH = ARTIFACTS_DIR / "thresholds.json"

# --- App Initialization ---
app = Flask(__name__)
app.config["BASIC_AUTH_USERNAME"] = os.environ.get("NLU_USERNAME")
app.config["BASIC_AUTH_PASSWORD"] = os.environ.get("NLU_PASSWORD")

if dsn := os.environ.get("SENTRY_DSN"):
    sentry_sdk.init(dsn=dsn, traces_sample_rate=0.1)
    logger.info("Sentry initialized.")

basic_auth = BasicAuth(app)
metrics = PrometheusMetrics(app)
metrics.info("app_info", "MC Intent Classifier Service", version=version)

# --- Load Classifier ---
classifier: EnsembleClassifier | None = None
try:
    classifier = EnsembleClassifier(
        artifacts_dir=ARTIFACTS_DIR, thresholds_path=THRESHOLDS_PATH
    )
    logger.info(f"Classifier loaded. Version: {classifier.manifest['version']}")
except Exception:
    logger.exception("CRITICAL: Failed to load the classifier on startup.")


# --- API Endpoints ---
@app.route("/health")
@metrics.do_not_track()  # type: ignore
def health_check() -> ResponseReturnValue:
    """Health check endpoint."""
    status = {
        "status": "ok",
        "version": version,
        "model_loaded": classifier is not None,
    }
    return jsonify(status)


@app.route("/nlu/babyloss/")
@basic_auth.required  # type: ignore
def nlu_babyloss() -> ResponseReturnValue:
    """Detects the 'baby loss' sub-intent with detailed metrics."""
    if not classifier:
        return jsonify({"error": "Classifier not available"}), 503

    question = request.args.get("question")
    if not question:
        return jsonify({"error": "The 'question' parameter is required."}), 400

    prediction = classifier.predict(question)
    top_intent = prediction["intents"][0]

    is_babyloss = (
        top_intent["key"] == "SENSITIVE_EXIT"
        and top_intent["enrichment"].get("sub_reason") == "BABY_LOSS"
    )

    response_payload = {
        "babyloss": is_babyloss,
        "model_version": prediction["model_version"],
        "parent_label": top_intent["key"],
        "sub_intent": top_intent["enrichment"].get("sub_reason"),
        "probability": top_intent["probability"],
        "review_status": prediction["review_status"],
    }
    return jsonify(response_payload)


@app.route("/nlu/feedback/")
@basic_auth.required  # type: ignore
def nlu_feedback() -> ResponseReturnValue:
    """
    Detects 'compliment' or 'complaint' sub-intents with detailed metrics.
    This endpoint assumes the user is already in a "feedback" state.
    It bypasses the parent classifier and calls the sentiment model directly.
    """
    if not classifier:
        return jsonify({"error": "Classifier not available"}), 503

    question = request.args.get("question")
    if not question:
        return jsonify({"error": "The 'question' parameter is required."}), 400

    # 1. Normalize the text (just like the classifier does)
    norm_text = normalise_text(question)

    # 2. Call the sentiment pipeline directly (bypassing the parent model)
    sentiment = classifier.sentiment_pipeline(norm_text, top_k=1)[0]
    sentiment_label = sentiment["label"]
    sentiment_score = round(sentiment["score"], 4)

    # 3. Apply the correct logic to map sentiment to intent
    intent_response = "None"
    if sentiment_label == "positive":
        intent_response = "COMPLIMENT"
    elif sentiment_label == "negative":
        intent_response = "COMPLAINT"

    # 4. Determine review_status using DATA-DRIVEN thresholds
    is_neutral_feedback = sentiment_label == "neutral"

    # Use the new, data-driven threshold from your tuning script
    is_low_confidence = sentiment_score < classifier.thresholds.sentiment_review_band

    review_status = (
        "NEEDS_REVIEW" if (is_low_confidence or is_neutral_feedback) else "CLASSIFIED"
    )

    # 5. Build the final payload
    response_payload = {
        "intent": intent_response,
        "model_version": classifier.manifest["version"],
        "parent_label": "FEEDBACK",
        "probability": sentiment_score,
        "review_status": review_status,
        "sentiment_label": sentiment_label,
    }
    return jsonify(response_payload)
