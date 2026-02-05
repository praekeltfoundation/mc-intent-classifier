"""Celery tasks for async processing."""

import logging

from src.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="src.tasks.hello_world")
def hello_world(name: str = "World") -> dict:
    """
    Simple hello world task for testing Celery connection.

    Args:
        name: Name to greet (default: "World")

    Returns:
        dict: Status message with greeting
    """
    message = f"Hello, {name}!"
    logger.info(f"Hello world task executed: {message}")
    return {"status": "success", "message": message}


# Future tasks will be added here:
# - classify_and_update_turn: Main classification task that calls Turn API
