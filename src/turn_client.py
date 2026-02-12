"""Turn API client for updating message labels."""

import logging
import requests

from src.config import load_config
logger = logging.getLogger(__name__)


class TurnAPIError(Exception):
    """Base exception for Turn API errors."""


class TurnAPIClientError(TurnAPIError):
    """Exception for 4xx client errors (won't retry)."""


class TurnAPIServerError(TurnAPIError):
    """Exception for 5xx server errors (can retry)."""


class TurnAPIClient:
    """Client for interacting with Turn API to update message labels."""

    def __init__(self):
        """
        Initialize Turn API client.

        Raises:
            ValueError: If TURN_API_BASE_URL or TURN_API_TOKEN are not set
        """
        config = load_config()
        self.base_url = config["TURN_API_BASE_URL"]
        self.api_token = config["TURN_API_TOKEN"]

        if not self.base_url:
            msg = "TURN_API_BASE_URL must be set"
            raise ValueError(msg)

        if not self.api_token:
            msg = "TURN_API_TOKEN must be set"
            raise ValueError(msg)

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_token}",
                "Accept": "application/vnd.v1+json",
                "Content-Type": "application/json",
            }
        )

    def update_message_label(self, message_id: str, label: str) -> dict:
        """
        Update a message label in Turn.

        Args:
            message_id: Turn message ID
            label: Intent label to assign

        Returns:
            dict: Response from Turn API

        Raises:
            TurnAPIClientError: For 4xx errors (invalid request)
            TurnAPIServerError: For 5xx errors (server issues)
            TurnAPIError: For other errors
        """
        endpoint = f"{self.base_url}/v1/messages/{message_id}/labels"

        payload = {"labels": [label]}

        try:
            logger.info(f"Updating Turn message {message_id} with label: {label}")

            response = self.session.post(endpoint, json=payload, timeout=30)

            # Check for errors
            if response.status_code >= 500:
                error_msg = f"Turn API server error (HTTP {response.status_code}): {response.text}"
                logger.error(error_msg)
                raise TurnAPIServerError(error_msg)

            if response.status_code >= 400:
                error_msg = f"Turn API client error (HTTP {response.status_code}): {response.text}"
                logger.error(error_msg)
                raise TurnAPIClientError(error_msg)

            # Success
            logger.info(f"Successfully updated message {message_id} label to {label}")
            return response.json() if response.text else {}

        except requests.Timeout as e:
            error_msg = f"Turn API request timeout for message {message_id}: {e}"
            logger.error(error_msg)
            raise TurnAPIServerError(error_msg) from e

        except requests.RequestException as e:
            error_msg = f"Turn API request error for message {message_id}: {e}"
            logger.error(error_msg)
            raise TurnAPIError(error_msg) from e
