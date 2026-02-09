"""Tests for Celery tasks."""

import pytest

from src.celery_app import celery_app
from src.tasks import (
    build_classify_and_update_chain,
    classify_turn_message,
    update_turn_message_label,
)
from src.turn_client import TurnAPIClientError, TurnAPIServerError

# Enable eager mode for all tests in this module (run tasks synchronously)
celery_app.conf.task_always_eager = True
celery_app.conf.task_eager_propagates = True


class TestCeleryConfiguration:
    """Tests for Celery app configuration."""

    def test_celery_app_exists(self):
        """Test that Celery app is properly initialized."""
        assert celery_app is not None
        assert celery_app.main == "mc-intent-classifier"

    def test_celery_broker_configured(self):
        """Test that broker URL is configured."""
        assert celery_app.conf.broker_url is not None
        # Should either be from env var or default
        assert "amqp://" in celery_app.conf.broker_url

    def test_task_serializer_is_json(self):
        """Test that task serializer is set to JSON."""
        assert celery_app.conf.task_serializer == "json"
        assert "json" in celery_app.conf.accept_content


class TestClassifyTurnMessageTask:
    """Tests for the classify_turn_message task."""

    def test_classify_success(self, mocker):
        """Test successful classification."""
        # Mock classifier
        mock_classifier = mocker.Mock()
        mock_classifier.classify.return_value = ("compliment", 0.95)
        mocker.patch("src.tasks.classifier", mock_classifier)

        # Execute task
        result = classify_turn_message("msg-123", "Thank you so much!")

        # Verify classifier was called
        mock_classifier.classify.assert_called_once_with("Thank you so much!")

        # Verify result
        assert result["message_id"] == "msg-123"
        assert result["intent"] == "compliment"
        assert result["confidence"] == 0.95

    def test_classify_classifier_not_loaded(self, mocker):
        """Test error when classifier is not loaded."""
        # Mock classifier as None (not loaded)
        mocker.patch("src.tasks.classifier", None)

        # Execute task - should raise ValueError
        with pytest.raises(ValueError, match="Intent classifier not loaded"):
            classify_turn_message("msg-789", "Hello world")

    def test_classify_unclassified_intent(self, mocker):
        """Test handling of unclassified intent."""
        # Mock classifier to return Unclassified
        mock_classifier = mocker.Mock()
        mock_classifier.classify.return_value = ("Unclassified", 0.42)
        mocker.patch("src.tasks.classifier", mock_classifier)

        # Execute task
        result = classify_turn_message("msg-unclass", "asdfghjkl")

        # Verify result
        assert result["intent"] == "Unclassified"
        assert result["confidence"] == 0.42


class TestUpdateTurnMessageLabelTask:
    """Tests for the update_turn_message_label task."""

    def test_update_turn_label_success(self, mocker, monkeypatch):
        """Test successful Turn API update."""

        # Mock Turn API client
        mock_turn_client = mocker.Mock()
        mock_turn_client.update_message_label.return_value = {
            "status": "success",
            "message_id": "msg-123",
        }
        mocker.patch("src.tasks.TurnAPIClient", return_value=mock_turn_client)

        # Execute task
        result = update_turn_message_label(
            {"message_id": "msg-123", "intent": "compliment"}
        )

        # Verify Turn API was called
        mock_turn_client.update_message_label.assert_called_once_with(
            "msg-123", "compliment"
        )

        # Verify result
        assert result["message_id"] == "msg-123"
        assert result["intent"] == "compliment"
        assert result["turn_response"]["status"] == "success"

    def test_update_turn_label_client_error(self, mocker, monkeypatch):
        """Test that 4xx client errors do NOT trigger retry."""
        # Set up Turn API env vars
        monkeypatch.setenv("TURN_API_BASE_URL", "https://api.turn.io")
        monkeypatch.setenv("TURN_API_TOKEN", "test-token")

        # Mock Turn API client to raise client error (4xx)
        mock_turn_client = mocker.Mock()
        mock_turn_client.update_message_label.side_effect = TurnAPIClientError(
            "Turn API client error (HTTP 400): Invalid label"
        )
        mocker.patch("src.tasks.TurnAPIClient", return_value=mock_turn_client)

        # Execute task - should raise TurnAPIClientError without retry
        with pytest.raises(TurnAPIClientError, match="Turn API client error"):
            update_turn_message_label({"message_id": "msg-400", "intent": "greeting"})

        # Verify Turn API was called only once (no retry)
        mock_turn_client.update_message_label.assert_called_once()

    def test_update_turn_label_server_error_retry(self, mocker, monkeypatch):
        """Test that 5xx server errors trigger automatic retry."""
        # Set up Turn API env vars
        monkeypatch.setenv("TURN_API_BASE_URL", "https://api.turn.io")
        monkeypatch.setenv("TURN_API_TOKEN", "test-token")

        # Mock Turn API client to raise server error (5xx)
        mock_turn_client = mocker.Mock()
        mock_turn_client.update_message_label.side_effect = TurnAPIServerError(
            "Turn API server error (HTTP 500): Internal server error"
        )
        mocker.patch("src.tasks.TurnAPIClient", return_value=mock_turn_client)

        # Execute task - should raise TurnAPIServerError (retry will be handled by Celery)
        with pytest.raises(TurnAPIServerError, match="Turn API server error"):
            update_turn_message_label({"message_id": "msg-500", "intent": "question"})

        # Verify Turn API was called
        mock_turn_client.update_message_label.assert_called_once()


class TestClassifyAndUpdateChain:
    """Tests for the classify and update chain helper."""

    def test_build_chain_signatures(self):
        """Ensure chain is composed of classify then update tasks."""
        chain_sig = build_classify_and_update_chain("msg-123", "Hello there")

        assert len(chain_sig.tasks) == 2
        assert chain_sig.tasks[0].name == classify_turn_message.name
        assert chain_sig.tasks[1].name == update_turn_message_label.name
        assert chain_sig.tasks[0].args == ("msg-123", "Hello there")
        assert chain_sig.tasks[1].args == ()

    def test_chain_passes_result_to_update(self, mocker):
        """Ensure classifier result is passed to update_turn_message_label."""
        mock_classifier = mocker.Mock()
        mock_classifier.classify.return_value = ("compliment", 0.95)
        mocker.patch("src.tasks.classifier", mock_classifier)

        mock_update = mocker.patch(
            "src.tasks.update_turn_message_label.run",
            return_value={"status": "success"},
        )

        chain_sig = build_classify_and_update_chain("msg-123", "Thank you!")
        chain_sig.apply()

        mock_update.assert_called_once_with(
            {"message_id": "msg-123", "intent": "compliment", "confidence": 0.95}
        )
