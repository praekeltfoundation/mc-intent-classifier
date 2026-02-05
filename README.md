# MomConnect Intent Classifier

Model that classifies the intent of inbound messages. It is not intended to be exposed to the outside world, we only have it accisble inside the cluster.

## Development
This project uses [poetry](https://python-poetry.org/docs/#installation) for packaging and dependancy management, so install that first.

Ensure you're also running at least python 3.11, `python --version`.

Then you can install the dependencies
```bash
~ poetry install
```

To run a local worker, set NLU_USERNAME and NLU_PASSWORD environment variables, then start up the flask worker
```bash
~ poetry run flask --app src.application run
```

### Running the Celery Worker

For async task processing, you need to run a Celery worker. First, ensure you have RabbitMQ running (or another message broker configured via `CELERY_BROKER_URL`).

To start the Celery worker:
```bash
~ poetry run celery -A src.celery_app worker --loglevel=info --concurrency=4
```

To test the Celery connection, you can run the hello world task:
```python
from src.tasks import hello_world
result = hello_world.delay("Test")
print(result.get())  # Should print: {'status': 'success', 'message': 'Hello, Test!'}
```

For local development with synchronous execution (no RabbitMQ needed), set:
```bash
export CELERY_TASK_ALWAYS_EAGER=true
```

To run the autoformatting and linting, run
```bash
~ ruff format && ruff check && mypy --install-types
```

For the test runner, we use [pytest](https://docs.pytest.org/):
```bash
~ pytest
```

## Regenerating the embeddings json file

1. Delete the json embeddings file in src/data/
1. Update the nlu.yaml with your changes
1. Run the flask app, this should regenerate the embeddings file. `poetry run flask --app src.application run`

## Editor configuration

If you'd like your editor to handle linting and/or formatting for you, here's how to set it up.

### Visual Studio Code

1. Install the Python and Ruff extensions
1. In settings, check the "Python > Linting: Mypy Enabled" box
1. In settings, set the "Python > Formatting: Provider" to "black" (apparently "ruff format" isn't supported by the Python extension yet and "black" is probably close enough)
1. If you want to have formatting automatically apply, in settings, check the "Editor: Format On Save" checkbox

Alternatively, add the following to your `settings.json`:
```json
{
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
}
```

## Release process

To release a new version, follow these steps:

1. Make sure all relevant PRs are merged and that all necessary QA testing is complete
1. Make sure release notes are up to date and accurate
1. In one commit on the `main` branch:
   - Update the version number in `pyproject.toml` to the release version
   - Replace the UNRELEASED header in `CHANGELOG.md` with the release version and date
1. Tag the release commit with the release version (for example, `v0.2.1` for version `0.2.1`)
1. Push the release commit and tag
1. In one commit on the `main` branch:
   - Update the version number in `pyproject.toml` to the next pre-release version
   - Add a new UNRELEASED header in `CHANGELOG.md`
1. Push the post-release commit

## Running in Production
There is a [docker image](https://github.com/praekeltfoundation/mc-intent-classifier/pkgs/container/mc-intent-classifier) that can be used to easily run this service. It uses the following environment variables for configuration:

| Variable      | Description | Required |
| ----------    | ----------- | -------- |
| NLU_USERNAME  | The username used for API requests | Yes (for /nlu/ endpoint) |
| NLU_PASSWORD  | The password used for API requests | Yes (for /nlu/ endpoint) |
| SENTRY_DSN    | Where to send exceptions to | No |
| CELERY_BROKER_URL | RabbitMQ connection URL (e.g., amqp://user:pass@host:5672/) | Yes (for async processing) |
| CELERY_RESULT_BACKEND | Result backend URL (e.g., rpc://) | No (defaults to rpc://) |
| CELERY_TASK_ALWAYS_EAGER | Set to "true" for synchronous task execution (testing only) | No |
| TURN_API_BASE_URL | Turn API base URL (e.g., https://whatsapp.turn.io) | Yes (for Turn integration) |
| TURN_API_TOKEN | Turn API authentication token (Bearer token) | Yes (for Turn integration) |

**Note:** You need to run both the Flask app (webhook receiver) and Celery worker (task processor) containers for full functionality.
