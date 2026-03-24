"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os

DEFAULT_CELERY_BROKER_URL = "amqp://guest:guest@localhost:5672//"
DEFAULT_CELERY_TASK_TIME_LIMIT = 900
DEFAULT_CELERY_TASK_SOFT_TIME_LIMIT = 840
DEFAULT_CELERY_WARM_ON_STARTUP = False


def load_config() -> dict:
    """Load application configuration from environment variables."""
    return {
        "TURN_HMAC_SECRET": os.environ.get("TURN_HMAC_SECRET"),
        "SENTRY_DSN": os.environ.get("SENTRY_DSN"),
        "TURN_API_BASE_URL": os.environ.get("TURN_API_BASE_URL", "").rstrip("/"),
        "TURN_API_TOKEN": os.environ.get("TURN_API_TOKEN", ""),
        "CELERY_BROKER_URL": os.environ.get(
            "CELERY_BROKER_URL", DEFAULT_CELERY_BROKER_URL
        ),
        "CELERY_TASK_TIME_LIMIT": int(
            os.environ.get("CELERY_TASK_TIME_LIMIT", DEFAULT_CELERY_TASK_TIME_LIMIT)
        ),
        "CELERY_TASK_SOFT_TIME_LIMIT": int(
            os.environ.get(
                "CELERY_TASK_SOFT_TIME_LIMIT", DEFAULT_CELERY_TASK_SOFT_TIME_LIMIT
            )
        ),
        "CELERY_WARM_ON_STARTUP": os.environ.get(
            "CELERY_WARM_ON_STARTUP", str(DEFAULT_CELERY_WARM_ON_STARTUP)
        )
        .lower()
        .strip()
        == "true",
        "CELERY_TASK_ALWAYS_EAGER": os.environ.get("CELERY_TASK_ALWAYS_EAGER", "false")
        .lower()
        .strip()
        == "true",
    }


def load_celery_config(config: dict | None = None) -> dict:
    """Return Celery configuration derived from environment variables."""
    if config is None:
        config = load_config()
    return {
        "broker_url": config["CELERY_BROKER_URL"],
        "task_serializer": "json",
        "accept_content": ["json"],
        "timezone": "UTC",
        "enable_utc": True,
        "task_track_started": True,
        "task_time_limit": config["CELERY_TASK_TIME_LIMIT"],
        "task_soft_time_limit": config["CELERY_TASK_SOFT_TIME_LIMIT"],
        "worker_prefetch_multiplier": 1,  # Fetch one task at a time for long-running tasks
        "worker_max_tasks_per_child": 1000,  # Restart worker after 1000 tasks
    }
