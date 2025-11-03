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
from src.utils.normalise import normalise_text


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
    """Tests the /nlu/feedback/ endpoint for a compliment using the new logic."""
    # 1. Mock the classifier object
    mock_classifier = mocker.patch("src.application.classifier")

    # 2. Mock the return value of sentiment_pipeline
    mock_classifier.sentiment_pipeline.return_value = [
        {"label": "positive", "score": 0.99}
    ]

    # 3. Mock the threshold value
    # We need to configure the mock's structure properly
    mock_classifier.thresholds = mocker.MagicMock()
    mock_classifier.thresholds.sentiment_review_band = 0.75  # Example threshold

    # 4. Mock the manifest dictionary (used for model_version)
    mock_classifier.manifest = {"version": "test-v1"}

    # 5. Make the request
    response = client.get(
        "/nlu/feedback/?question=the nurses were fantastic", headers=basic_auth_headers
    )

    # 6. Assertions
    assert response.status_code == 200
    expected_json = {
        "intent": "COMPLIMENT",
        "model_version": "test-v1",
        "parent_label": "FEEDBACK",
        "probability": 0.99,
        "review_status": "CLASSIFIED",  # Because 0.99 > 0.75
        "sentiment_label": "positive",  # Check the extra field
    }
    assert response.json == expected_json

    # 7. Verify the correct methods were called
    mock_classifier.sentiment_pipeline.assert_called_once()
    # Check the input text was normalized before being passed
    call_args, _ = mock_classifier.sentiment_pipeline.call_args
    expected_normalised_text = normalise_text("the nurses were fantastic")
    assert call_args[0] == expected_normalised_text

    # Verify 'predict' was NOT called
    mock_classifier.predict.assert_not_called()


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


@pytest.mark.parametrize(
    "text, expected_mock_label, expected_mock_sub_intent",
    [
        # User's bug report examples
        ("Good service", "FEEDBACK", "COMPLIMENT"),
        ("dirty clinic", "FEEDBACK", "COMPLAINT"),
        ("Service was good", "FEEDBACK", "COMPLIMENT"),
        # A known-good example to ensure it's NOT babyloss
        ("Staff was friendly", "FEEDBACK", "COMPLIMENT"),
        # An 'OTHER' example that should also not be babyloss
        ("Next clinic appointment when?", "OTHER", "INFORMATION_QUERY"),
    ],
)
def test_nlu_babyloss_negative_cases(
    client: FlaskClient,
    mocker: MockerFixture,
    basic_auth_headers: dict[str, str],
    text: str,
    expected_mock_label: str,
    expected_mock_sub_intent: str,
) -> None:
    """
    Tests that common feedback or other messages are NOT misclassified as
    babyloss by the /nlu/babyloss/ endpoint.

    This test mocks the classifier's *correct* prediction (e.g., FEEDBACK)
    and asserts that the endpoint logic correctly returns "babyloss: false".
    """
    mock_classifier = mocker.patch("src.application.classifier")

    # 1. Mock the classifier's *correct* response for this text
    enrichment = Enrichment(sub_reason=expected_mock_sub_intent)

    mock_classifier.predict.return_value = PredictionResponse(
        model_version="test-v1",
        intents=[
            IntentResult(
                label=expected_mock_label,
                key=expected_mock_label,
                probability=0.95,
                enrichment=enrichment,
            )
        ],
        review_status="CLASSIFIED",
    )

    # 2. Call the endpoint we are testing
    response = client.get(f"/nlu/babyloss/?question={text}", headers=basic_auth_headers)

    # 3. Assert the results
    assert response.status_code == 200

    # The /nlu/babyloss/ endpoint logic checks the prediction.
    # Since the top label is FEEDBACK or OTHER, it will not find "BABY_LOSS".
    expected_json = {
        "babyloss": False,
        "model_version": "test-v1",
        "parent_label": expected_mock_label,
        "sub_intent": expected_mock_sub_intent,
        "probability": 0.95,
        "review_status": "CLASSIFIED",
    }
    assert response.json == expected_json
    mock_classifier.predict.assert_called_once_with(text)
