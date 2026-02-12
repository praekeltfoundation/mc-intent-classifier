import importlib.metadata
import sentry_sdk
from flask import Flask, request
from prometheus_flask_exporter import PrometheusMetrics
from pydantic import ValidationError

from src.config import load_config
from src.tasks import build_classify_and_update_chain
from src.turn_webhook import TurnWebhook
from src.utils import validate_turn_signature

version = importlib.metadata.version("mc-intent-classifier")

app = Flask(__name__)
app.config.from_mapping(load_config())


if app.config.get("SENTRY_DSN"):
    sentry_sdk.init(
        dsn=app.config.get("SENTRY_DSN"),
        traces_sample_rate=0.0,
    )

metrics = PrometheusMetrics(app)
metrics.info("app_info", "Application info", version=version)


@app.route("/nlu/", methods=["POST"])
def nlu():
    signature_error = validate_turn_signature(
        request, app.config.get("TURN_HMAC_SECRET")
    )
    if signature_error:
        return signature_error

    payload = request.get_json(silent=True)
    if payload is None:
        return {"error": "json body required"}, 400

    try:
        webhook = TurnWebhook.model_validate(payload)
    except ValidationError as exc:
        return {"error": "invalid payload", "details": exc.errors()}, 400

    text_messages = [
        message
        for message in webhook.messages
        if message.type == "text" and message.text and message.text.body
    ]

    for message in text_messages:
        build_classify_and_update_chain(message.id, message.text.body).apply_async()

    if not text_messages:
        return {"status": "ignored", "count": 0}

    return {"status": "queued", "count": len(text_messages)}
