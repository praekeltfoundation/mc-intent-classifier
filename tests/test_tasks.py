"""Tests for Celery tasks."""

from src.celery_app import celery_app
from src.tasks import hello_world

# Enable eager mode for all tests in this module (run tasks synchronously)
celery_app.conf.task_always_eager = True
celery_app.conf.task_eager_propagates = True


class TestHelloWorldTask:
    """Tests for the hello_world task."""

    def test_hello_world_default(self):
        """Test hello world task with default parameter."""
        result = hello_world.delay()
        assert result.get() == {"status": "success", "message": "Hello, World!"}

    def test_hello_world_custom_name(self):
        """Test hello world task with custom name."""
        result = hello_world.delay("Celery")
        assert result.get() == {"status": "success", "message": "Hello, Celery!"}

    def test_hello_world_synchronous(self):
        """Test hello world task called synchronously."""
        result = hello_world("Testing")
        assert result == {"status": "success", "message": "Hello, Testing!"}


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

    def test_celery_result_backend_configured(self):
        """Test that result backend is configured."""
        assert celery_app.conf.result_backend is not None

    def test_task_serializer_is_json(self):
        """Test that task serializer is set to JSON."""
        assert celery_app.conf.task_serializer == "json"
        assert celery_app.conf.result_serializer == "json"
        assert "json" in celery_app.conf.accept_content
