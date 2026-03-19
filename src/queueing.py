"""Celery task submission helpers for producer processes."""

from celery import chain

from src.celery_app import celery_app

CLASSIFY_TURN_MESSAGE_TASK = "src.tasks.classify_turn_message"
UPDATE_TURN_MESSAGE_LABEL_TASK = "src.tasks.update_turn_message_label"


def build_classify_and_update_chain(message_id: str, message_text: str):
    """Build a Celery chain without importing worker-only task modules."""
    return chain(
        celery_app.signature(
            CLASSIFY_TURN_MESSAGE_TASK,
            args=(message_id, message_text),
        ),
        celery_app.signature(UPDATE_TURN_MESSAGE_LABEL_TASK),
    )
