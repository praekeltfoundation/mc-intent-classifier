import importlib.metadata
import os
from pathlib import Path

import sentry_sdk
from flask import Flask, request
from flask_basicauth import BasicAuth
from prometheus_flask_exporter import PrometheusMetrics

from src.intent_classifier import IntentClassifier

version = importlib.metadata.version("mc-intent-classifier")

dirname = os.path.dirname(__file__)
DATA_PATH = Path(f"{dirname}/data")
NLU_FILE_PATH = DATA_PATH / "nlu.yaml"
EMBEDDINGS_FILE_PATH = DATA_PATH / "intent_embeddings.json"

app = Flask(__name__)
app.config["BASIC_AUTH_USERNAME"] = os.environ.get("NLU_USERNAME")
app.config["BASIC_AUTH_PASSWORD"] = os.environ.get("NLU_PASSWORD")


if os.environ.get("SENTRY_DSN"):
    sentry_sdk.init(
        dsn=os.environ.get("SENTRY_DSN"),
        traces_sample_rate=0.0,
    )

basic_auth = BasicAuth(app)

metrics = PrometheusMetrics(app)
metrics.info("app_info", "Application info", version=version)

classifier = IntentClassifier(
    embeddings_path=EMBEDDINGS_FILE_PATH, nlu_path=NLU_FILE_PATH
)


@app.route("/nlu/")
@basic_auth.required
def nlu():
    question = request.args.get("question")

    if not question:
        return {"error": "question is required"}, 400

    intent, confidence = classifier.classify(question)
    return {
        "question": question,
        "intent": intent,
        "confidence": confidence,
    }
