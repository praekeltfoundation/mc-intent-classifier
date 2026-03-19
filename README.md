# MomConnect Intent Classifier

Model that classifies the intent of inbound messages. It is not intended to be exposed to the outside world; we only have it accessible inside the cluster.

## Development

This project uses [Poetry](https://python-poetry.org/docs/#installation) for packaging and dependency management, so install that first.

Ensure you're also running at least python 3.11, `python --version`.

Then you can install the dependencies:

```bash
poetry install
```

### Local Stack with Docker Compose

This project ships with a local async stack:

- `api`: Flask + Gunicorn on port `5000`
- `worker`: Celery worker
- `rabbitmq`: internal broker (no host ports exposed)

1. Create your local env file:

```bash
cp .env.docker.example .env.docker
```

1. Fill in required values in `.env.docker`:

- `TURN_HMAC_SECRET`
- `TURN_API_BASE_URL`
- `TURN_API_TOKEN`

1. Spin up the stack:

```bash
docker compose --env-file .env.docker up --build
```

1. Verify the API is up:

```bash
curl http://localhost:5000/metrics
```

1. Expose your local API with ngrok (host machine):

```bash
ngrok http 5000
```

Then point your webhook sender at:

```text
https://<your-ngrok-subdomain>.ngrok-free.app/nlu/
```

1. Quick local endpoint check (without signature, should return `401`):

```bash
curl -i -X POST http://localhost:5000/nlu/ \
  -H "Content-Type: application/json" \
  -d '{"messages":[]}'
```

### Local Python-Only Run (No Docker)

To run Flask directly:

```bash
poetry run flask --app src.application run
```

To run Celery directly (requires RabbitMQ or another broker):

```bash
poetry run celery -A src.celery_app worker --loglevel=info --concurrency=4
```

For synchronous local testing (no broker):

```bash
export CELERY_TASK_ALWAYS_EAGER=true
```

To run the autoformatting and linting:

```bash
poetry run ruff format && poetry run ruff check && poetry run mypy --install-types
```

For the test runner, we use [pytest](https://docs.pytest.org/):

```bash
poetry run pytest
```

## Regenerating the embeddings JSON file

1. Delete the JSON embeddings file in `src/data/`.
1. Update `nlu.yaml` with your changes.
1. Run the Flask app. This should regenerate the embeddings file:

```bash
poetry run flask --app src.application run
```

## Editor configuration

If you'd like your editor to handle linting and/or formatting for you, here's how to set it up.

### Visual Studio Code

1. Install the Python and Ruff extensions.
1. In settings, check the "Python > Linting: Mypy Enabled" box.
1. In settings, set the "Python > Formatting: Provider" to "black" (apparently `ruff format` is not supported by the Python extension yet, and `black` is probably close enough).
1. If you want formatting to apply automatically, in settings, check the "Editor: Format On Save" checkbox.

Alternatively, add the following to your `settings.json`:

```json
{
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true
}
```

## Release process

To release a new version, follow these steps:

1. Make sure all relevant PRs are merged and that all necessary QA testing is complete.
1. Make sure release notes are up to date and accurate.
1. In one commit on the `main` branch:
   - Update the version number in `pyproject.toml` to the release version
   - Replace the UNRELEASED header in `CHANGELOG.md` with the release version and date
1. Tag the release commit with the release version (for example, `v0.2.1` for version `0.2.1`).
1. Push the release commit and tag.
1. In one commit on the `main` branch:
   - Update the version number in `pyproject.toml` to the next pre-release version
   - Add a new UNRELEASED header in `CHANGELOG.md`
1. Push the post-release commit.

## Running in Production

There is a [docker image](https://github.com/praekeltfoundation/mc-intent-classifier/pkgs/container/mc-intent-classifier) that can be used to run this service. It uses the following environment variables for configuration:

| Variable | Description | Required |
| --- | --- | --- |
| TURN_HMAC_SECRET | Shared secret used to validate the `X-Turn-Hook-Signature` header on `/nlu/` requests | Yes |
| SENTRY_DSN | Where to send exceptions | No |
| CELERY_BROKER_URL | RabbitMQ connection URL (for example, `amqp://user:pass@host:5672/`) | Yes (for async processing) |
| CELERY_TASK_ALWAYS_EAGER | Set to `true` for synchronous task execution (testing only) | No |
| TURN_API_BASE_URL | Turn API base URL (for example, `https://whatsapp.turn.io`) | Yes (for Turn integration) |
| TURN_API_TOKEN | Turn API authentication token (Bearer token) | Yes (for Turn integration) |

**Note:** You need to run both the Flask app (webhook receiver) and Celery worker (task processor) containers for full functionality.
