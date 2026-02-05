"""Celery application configuration for async task processing."""

import os

from celery import Celery

# Initialize Celery app
celery_app = Celery("mc-intent-classifier")

# RabbitMQ broker configuration
broker_url = os.environ.get("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")

# Celery configuration
celery_app.conf.update(
    broker_url=broker_url,
    task_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max per task
    task_soft_time_limit=240,  # 4 minutes soft limit
    worker_prefetch_multiplier=1,  # Fetch one task at a time for long-running tasks
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
)

# Task discovery - automatically discover tasks from src.tasks module
celery_app.autodiscover_tasks(["src"])

# For testing - allow synchronous execution
if os.environ.get("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true":
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
