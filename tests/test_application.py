# tests/test_application.py
import os
import sys
from base64 import b64encode
from collections.abc import Generator

import pytest
from flask.testing import FlaskClient
from pytest_mock import MockerFixture

# This line is preserved from your original file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.application import app
from src.config.schemas import Enrichment, IntentResult, PredictionResponse


# --- This fixture is preserved from your original file ---
@pytest.fixture
def client() -> Generator[FlaskClient, None, None]:
    """A test client for the app."""
    app.config["TESTING"] = True
    app.config["BASIC_AUTH_USERNAME"] = "test_username"
    app.config["BASIC_AUTH_PASSWORD"] = "test_password"  # noqa: S105

    with app.test_client() as client:
        yield client


# --- Helper fixture for authentication headers ---
@pytest.fixture
def basic_auth_headers() -> dict[str, str]:
    """Provides correctly encoded basic authentication headers."""
    credentials = b64encode(b"test_username:test_password").decode("utf-8")
    return {"Authorization": f"Basic {credentials}"}


# --- NEW AND UPDATED TESTS ---


def test_nlu_babyloss_success(
    client: FlaskClient, mocker: MockerFixture, basic_auth_headers: dict[str, str]
) -> None:
    """Tests the /nlu/babyloss/ endpoint for a positive case."""
    mock_classifier = mocker.patch("src.application.classifier")
    mock_classifier.predict.return_value = PredictionResponse(
        model_version="test-v1",
        intents=[
            IntentResult(
                label="SENSITIVE_EXIT",
                key="SENSITIVE_EXIT",
                probability=0.98,
                enrichment=Enrichment(sub_reason="BABY_LOSS"),
            )
        ],
        review_status="CLASSIFIED",
    )

    response = client.get(
        "/nlu/babyloss/?question=I had a miscarriage", headers=basic_auth_headers
    )

    assert response.status_code == 200
    # FIX: Assert against the new, richer response payload
    expected_json = {
        "babyloss": True,
        "model_version": "test-v1",
        "parent_label": "SENSITIVE_EXIT",
        "sub_intent": "BABY_LOSS",
        "probability": 0.98,
        "review_status": "CLASSIFIED",
    }
    assert response.json == expected_json
    mock_classifier.predict.assert_called_once_with("I had a miscarriage")


def test_nlu_feedback_success(
    client: FlaskClient, mocker: MockerFixture, basic_auth_headers: dict[str, str]
) -> None:
    """Tests the /nlu/feedback/ endpoint for a compliment."""
    mock_classifier = mocker.patch("src.application.classifier")
    mock_classifier.predict.return_value = PredictionResponse(
        model_version="test-v1",
        intents=[
            IntentResult(
                label="FEEDBACK",
                key="FEEDBACK",
                probability=0.99,
                enrichment=Enrichment(sub_reason="COMPLIMENT"),
            )
        ],
        review_status="CLASSIFIED",
    )

    response = client.get(
        "/nlu/feedback/?question=the nurses were fantastic", headers=basic_auth_headers
    )

    assert response.status_code == 200
    # FIX: Assert against the new, richer response payload
    expected_json = {
        "intent": "COMPLIMENT",
        "model_version": "test-v1",
        "parent_label": "FEEDBACK",
        "probability": 0.99,
        "review_status": "CLASSIFIED",
    }
    assert response.json == expected_json
    mock_classifier.predict.assert_called_once_with("the nurses were fantastic")


# --- This test is preserved and updated from your original file ---
def test_nlu_unauthenticated(client: FlaskClient) -> None:
    """Tests that an endpoint requires authentication."""
    response = client.get(
        "/nlu/feedback/?question=the nurses at the clinic was fantastic",
    )
    assert response.status_code == 401


# --- This test is preserved and updated from your original file ---
def test_nlu_invalid(
    client: FlaskClient, mocker: MockerFixture, basic_auth_headers: dict[str, str]
) -> None:
    """Tests for a 400 error when the 'question' parameter is missing."""
    mocker.patch("src.application.classifier")

    response = client.get(
        "/nlu/babyloss/",
        headers=basic_auth_headers,
    )
    assert response.status_code == 400
    assert response.json == {"error": "The 'question' parameter is required."}
