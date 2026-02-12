import base64
import hmac
from hashlib import sha256

from flask import request

from src.application import app
from src.utils import validate_turn_signature


def _sign(payload: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode(), payload, sha256).digest()
    return base64.b64encode(digest).decode()


def test_validate_turn_signature_missing_secret():
    body = b'{"messages":[]}'
    with app.test_request_context("/nlu/", method="POST", data=body):
        error = validate_turn_signature(request, "")
        assert error == ({"error": "TURN_HMAC_SECRET must be set"}, 500)


def test_validate_turn_signature_missing_header():
    body = b'{"messages":[]}'
    with app.test_request_context("/nlu/", method="POST", data=body):
        error = validate_turn_signature(request, "test-secret")
        assert error == ({"error": "X-Turn-Hook-Signature header required"}, 401)


def test_validate_turn_signature_invalid_signature():
    body = b'{"messages":[]}'
    with app.test_request_context(
        "/nlu/",
        method="POST",
        data=body,
        headers={"X-Turn-Hook-Signature": "invalid"},
    ):
        error = validate_turn_signature(request, "test-secret")
        assert error == ({"error": "invalid hook signature"}, 401)


def test_validate_turn_signature_success():
    body = b'{"messages":[]}'
    signature = _sign(body, "test-secret")
    with app.test_request_context(
        "/nlu/",
        method="POST",
        data=body,
        headers={"X-Turn-Hook-Signature": signature},
    ):
        error = validate_turn_signature(request, "test-secret")
        assert error is None
