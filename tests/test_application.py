import os
import sys
from base64 import b64encode

import pytest

from src.application import app

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """A test client for the app."""
    app.config["BASIC_AUTH_USERNAME"] = "test_username"
    app.config["BASIC_AUTH_PASSWORD"] = "test_password"  # noqa: S105

    with app.test_client() as client:
        yield client


def test_nlu_success(client, mocker):
    """Test the home route."""

    method = mocker.patch("src.intent_classifier.IntentClassifier.classify")
    method.return_value = ("facility_compliment", 100.0)

    credentials = b64encode(b"test_username:test_password").decode("utf-8")

    response = client.get(
        "/nlu/?question=the nurses at the clinic was fantastic",
        headers={"Authorization": f"Basic {credentials}"},
    )
    assert response.status_code == 200
    assert response.json == {
        "confidence": 100.0,
        "intent": "facility_compliment",
        "question": "the nurses at the clinic was fantastic",
    }

    method.assert_called_once_with("the nurses at the clinic was fantastic")


def test_nlu_unauthenticated(client):
    """Test the home route."""

    response = client.get(
        "/nlu/?question=the nurses at the clinic was fantastic",
    )
    assert response.status_code == 401


def test_nlu_invalid(client):
    """"""

    credentials = b64encode(b"test_username:test_password").decode("utf-8")
    response = client.get(
        "/nlu/",
        headers={"Authorization": f"Basic {credentials}"},
    )
    assert response.status_code == 400
    assert response.json == {
        "error": "question is required",
    }
