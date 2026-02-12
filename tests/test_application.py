import base64
import hmac
import json
import os
import sys
from hashlib import sha256

import pytest

from src.application import app

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """A test client for the app."""
    app.config["TURN_HMAC_SECRET"] = "test-secret"  # noqa: S105

    with app.test_client() as client:
        yield client


def _sign_payload(payload: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode(), payload, sha256).digest()
    return base64.b64encode(digest).decode()


def test_nlu_success(client, mocker):
    """Test the home route."""

    mock_chain = mocker.Mock()
    mock_build_chain = mocker.patch(
        "src.application.build_classify_and_update_chain",
        return_value=mock_chain,
    )

    payload = {
        "messages": [
            {
                "id": "msg-123",
                "type": "text",
                "text": {"body": "the nurses at the clinic was fantastic"},
            }
        ]
    }
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    signature = _sign_payload(body, app.config["TURN_HMAC_SECRET"])

    response = client.post(
        "/nlu/",
        data=body,
        content_type="application/json",
        headers={"X-Turn-Hook-Signature": signature},
    )
    assert response.status_code == 200
    assert response.json == {"status": "queued", "count": 1}

    mock_build_chain.assert_called_once_with(
        "msg-123", "the nurses at the clinic was fantastic"
    )
    mock_chain.apply_async.assert_called_once_with()


def test_nlu_unauthenticated(client):
    """Test the home route."""

    payload = {
        "messages": [
            {
                "id": "msg-123",
                "type": "text",
                "text": {"body": "the nurses at the clinic was fantastic"},
            }
        ]
    }
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()

    response = client.post(
        "/nlu/",
        data=body,
        content_type="application/json",
    )
    assert response.status_code == 401


def test_nlu_missing_json_body(client):
    """"""

    signature = _sign_payload(b"", app.config["TURN_HMAC_SECRET"])

    response = client.post(
        "/nlu/",
        data=b"",
        content_type="application/json",
        headers={"X-Turn-Hook-Signature": signature},
    )
    assert response.status_code == 400
    assert response.json == {
        "error": "json body required",
    }


def test_nlu_invalid_payload(client):
    """"""

    body = json.dumps(
        {"not_messages": []}, separators=(",", ":"), sort_keys=True
    ).encode()
    signature = _sign_payload(body, app.config["TURN_HMAC_SECRET"])

    response = client.post(
        "/nlu/",
        data=body,
        content_type="application/json",
        headers={"X-Turn-Hook-Signature": signature},
    )
    assert response.status_code == 400
    assert response.json["error"] == "invalid payload"
