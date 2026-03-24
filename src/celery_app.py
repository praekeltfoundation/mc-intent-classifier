"""Celery application configuration for async task processing."""

from celery import Celery
from celery.signals import worker_ready

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


@worker_ready.connect
def queue_warmup_task(**kwargs) -> None:
    """Queue classifier warm-up after the worker is fully ready."""
    del kwargs

    if (
        app_config["CELERY_TASK_ALWAYS_EAGER"]
        or not app_config["CELERY_WARM_ON_STARTUP"]
    ):
        return

    celery_app.send_task("src.tasks.warm_intent_classifier")
