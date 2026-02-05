"""Tests for Turn API client."""

import pytest

from src.turn_client import (
    TurnAPIClient,
    TurnAPIClientError,
    TurnAPIError,
    TurnAPIServerError,
)


@pytest.fixture(autouse=True)
def setup_turn_env(monkeypatch):
    """Set up Turn API environment variables for all tests."""
    monkeypatch.setenv("TURN_API_BASE_URL", "https://api.turn.io")
    monkeypatch.setenv("TURN_API_TOKEN", "test-token")


class TestTurnAPIClient:
    """Tests for TurnAPIClient class."""

    def test_client_initialization_with_env_vars(self):
        """Test client initialization with environment variables."""
        client = TurnAPIClient()
        assert client.base_url == "https://api.turn.io"
        assert client.api_token == "test-token"  # noqa: S105

    def test_client_initialization_strips_trailing_slash(self, monkeypatch):
        """Test that trailing slash is removed from base URL."""
        monkeypatch.setenv("TURN_API_BASE_URL", "https://api.turn.io/")
        monkeypatch.setenv("TURN_API_TOKEN", "test-token")

        client = TurnAPIClient()
        assert client.base_url == "https://api.turn.io"

    def test_client_initialization_missing_base_url(self, monkeypatch):
        """Test that ValueError is raised if base URL is missing."""
        monkeypatch.delenv("TURN_API_BASE_URL", raising=False)
        monkeypatch.setenv("TURN_API_TOKEN", "test-token")

        with pytest.raises(ValueError, match="TURN_API_BASE_URL must be set"):
            TurnAPIClient()

    def test_client_initialization_missing_token(self, monkeypatch):
        """Test that ValueError is raised if token is missing."""
        monkeypatch.setenv("TURN_API_BASE_URL", "https://api.turn.io")
        monkeypatch.delenv("TURN_API_TOKEN", raising=False)

        with pytest.raises(ValueError, match="TURN_API_TOKEN must be set"):
            TurnAPIClient()

    def test_update_message_label_success(self, mocker):
        """Test successful label update."""
        client = TurnAPIClient()

        # Mock successful response
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.text = '{"status": "success", "message_id": "msg-123"}'
        mock_response.json.return_value = {"status": "success", "message_id": "msg-123"}
        mock_post = mocker.patch.object(
            client.session, "post", return_value=mock_response
        )

        result = client.update_message_label("msg-123", "compliment")

        # Verify request
        mock_post.assert_called_once_with(
            "https://api.turn.io/v1/messages/msg-123/labels",
            json={"labels": ["compliment"]},
        )

        # Verify response
        assert result == {"status": "success", "message_id": "msg-123"}

    def test_update_message_label_client_error(self, mocker):
        """Test 4xx client error handling."""
        client = TurnAPIClient()

        # Mock 400 Bad Request
        mock_response = mocker.Mock()
        mock_response.status_code = 400
        mock_response.text = "Invalid label format"
        mocker.patch.object(client.session, "post", return_value=mock_response)

        with pytest.raises(
            TurnAPIClientError, match="Turn API client error \\(HTTP 400\\)"
        ):
            client.update_message_label("msg-123", "invalid-label")

    def test_update_message_label_server_error(self, mocker):
        """Test 5xx server error handling."""
        client = TurnAPIClient()

        # Mock 500 Internal Server Error
        mock_response = mocker.Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mocker.patch.object(client.session, "post", return_value=mock_response)

        with pytest.raises(
            TurnAPIServerError, match="Turn API server error \\(HTTP 500\\)"
        ):
            client.update_message_label("msg-123", "label")

    def test_update_message_label_request_error(self, mocker):
        """Test general request error handling."""
        import requests

        client = TurnAPIClient()

        # Mock request exception
        mocker.patch.object(
            client.session,
            "post",
            side_effect=requests.RequestException("Connection failed"),
        )

        with pytest.raises(
            TurnAPIError, match="Turn API request error for message msg-123"
        ):
            client.update_message_label("msg-123", "label")

    def test_client_headers(self, monkeypatch):
        """Test that client has correct headers."""
        monkeypatch.setenv("TURN_API_BASE_URL", "https://api.turn.io")
        monkeypatch.setenv("TURN_API_TOKEN", "test-token-123")

        client = TurnAPIClient()

        headers = client.session.headers
        assert headers["Authorization"] == "Bearer test-token-123"
        assert headers["Accept"] == "application/vnd.v1+json"
        assert headers["Content-Type"] == "application/json"
