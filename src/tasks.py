"""Celery tasks for async processing."""

import logging
from pathlib import Path

from celery import chain

from src.celery_app import celery_app
from src.intent_classifier import IntentClassifier
from src.turn_client import TurnAPIClient, TurnAPIServerError

logger = logging.getLogger(__name__)

# Load classifier once at module level for efficiency
# Model is large and we're classifying thousands of messages
_DATA_DIR = Path(__file__).parent / "data"
_EMBEDDINGS_PATH = _DATA_DIR / "intent_embeddings.json"
_NLU_PATH = _DATA_DIR / "nlu.yaml"

try:
    classifier = IntentClassifier(
        embeddings_path=_EMBEDDINGS_PATH,
        nlu_path=_NLU_PATH,
    )
    logger.info("Intent classifier loaded successfully")
except Exception as e:
    logger.error(f"Failed to load intent classifier: {e}")
    classifier = None


@celery_app.task(
    acks_late=True,  # Only ack after task completes (safer for worker crashes)
)
def classify_turn_message(message_id: str, message_text: str) -> dict:
    """
    Classify incoming message text.

    This task:
    1. Classifies the message text using the IntentClassifier
    2. Does NOT call the Turn API (no label update here)

    Args:
        message_id: Turn message ID to classify
        message_text: Message text to classify

    Returns:
        dict: Classification result with keys:
            - message_id: The Turn message ID
            - intent: Classified intent label
            - confidence: Classification confidence score
            - error: Optional error string

    Raises:
        ValueError: Classifier not loaded or invalid input
    """
    # Validate classifier is loaded
    if classifier is None:
        error_msg = "Intent classifier not loaded. Cannot classify message."
        logger.error(f"{error_msg} Message ID: {message_id}")
        raise ValueError(error_msg)

    try:
        # Step 1: Classify the message
        logger.info(f"Classifying message {message_id}: {message_text[:100]}...")
        intent, confidence = classifier.classify(message_text)
        logger.info(
            f"Classified message {message_id} as '{intent}' with confidence {confidence:.4f}"
        )

        return {
            "message_id": message_id,
            "intent": intent,
            "confidence": confidence,
        }
    except Exception as e:
        # Log other errors but don't retry (likely classification or client errors)
        logger.error(
            f"Error processing message {message_id}: {e}",
            exc_info=True,
        )
        raise


@celery_app.task(
    acks_late=True,  # Only ack after task completes (safer for worker crashes)
    autoretry_for=(TurnAPIServerError,),  # Retry on 5xx server errors
    max_retries=3,
    retry_backoff=True,  # Exponential backoff
    retry_backoff_max=600,  # Max 10 minutes between retries
    retry_jitter=True,  # Add randomness to prevent thundering herd
)
def update_turn_message_label(classification: dict) -> dict:
    """
    Update Turn message label.

    This task:
    1. Updates the message label in Turn via the API
    2. Automatically retries on server errors (5xx) with exponential backoff
    3. Does NOT retry on client errors (4xx) - bad data won't improve on retry

    Args:
        classification: dict with keys message_id and intent

    Returns:
        dict: Update result with keys:
            - message_id: The Turn message ID
            - intent: Intent label sent to Turn
            - turn_response: Response from Turn API

    Raises:
        TurnAPIServerError: Turn API server error (5xx) - will auto-retry
        TurnAPIClientError: Turn API client error (4xx) - will NOT retry
    """
    message_id = classification["message_id"]
    intent = classification["intent"]

    try:
        turn_client = TurnAPIClient()
        turn_response = turn_client.update_message_label(message_id, intent)

        logger.info(
            f"Successfully updated Turn message {message_id} with label: {intent}"
        )

        return {
            "message_id": message_id,
            "intent": intent,
            "turn_response": turn_response,
        }
    except TurnAPIServerError:
        # Re-raise server errors to trigger retry
        logger.warning(
            f"Turn API server error for message {message_id}. "
            f"Retry {update_turn_message_label.request.retries}/{update_turn_message_label.max_retries}"
        )
        raise
    except Exception as e:
        # Log other errors but don't retry (likely client errors)
        logger.error(
            f"Error updating Turn message {message_id}: {e}",
            exc_info=True,
        )
        raise


def build_classify_and_update_chain(message_id: str, message_text: str):
    """Build a Celery chain that classifies then updates the Turn label."""
    return chain(
        classify_turn_message.s(message_id, message_text),
        update_turn_message_label.s(),
    )
