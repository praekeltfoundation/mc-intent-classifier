"""Celery application configuration for async task processing."""

from celery import Celery

from src.config import load_celery_config, load_config

# Initialize Celery app
celery_app = Celery("mc-intent-classifier")

app_config = load_config()
celery_app.conf.update(load_celery_config(app_config))

# Task discovery - automatically discover tasks from src.tasks module
celery_app.autodiscover_tasks(["src"])

# For testing - allow synchronous execution
if app_config["CELERY_TASK_ALWAYS_EAGER"]:
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
