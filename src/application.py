import importlib.metadata
import os
from pathlib import Path
from typing import Union

import sentry_sdk
from flask import Flask, Response, jsonify, request
from flask_basicauth import BasicAuth
from prometheus_flask_exporter import PrometheusMetrics

from src.classifiers.ensemble import EnsembleClassifier

# Version info
version = importlib.metadata.version("mc-intent-classifier")

# Paths
dirname = os.path.dirname(__file__)
DATA_PATH = Path(f"{dirname}/data")
ARTIFACTS_DIR = Path(dirname) / "artifacts"
THRESHOLDS_PATH = ARTIFACTS_DIR / "thresholds.json"

# Flask app
app = Flask(__name__)
app.config["BASIC_AUTH_USERNAME"] = os.environ.get("NLU_USERNAME")
app.config["BASIC_AUTH_PASSWORD"] = os.environ.get("NLU_PASSWORD")

# Sentry
if os.environ.get("SENTRY_DSN"):
    sentry_sdk.init(dsn=os.environ.get("SENTRY_DSN"), traces_sample_rate=0.0)

# Auth + Metrics
basic_auth = BasicAuth(app)
metrics = PrometheusMetrics(app)
metrics.info("app_info", "Application info", version=version)

# Load classifier
classifier = EnsembleClassifier(
    artifacts_dir=ARTIFACTS_DIR, thresholds_path=THRESHOLDS_PATH
)


@app.route("/nlu/")
@basic_auth.required  # type: ignore
def nlu() -> Union[Response, tuple[dict[str, str], int]]:
    question = request.args.get("question")
    if not question:
        return {"error": "question is required"}, 400

    response = classifier.predict(question)
    return jsonify(response.model_dump())  # pydantic ensures schema


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
