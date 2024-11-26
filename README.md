# MomConnect Intent Classifier

Model that classifies the intent of inbound messages.

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

| Variable      | Description |
| ----------    | ----------- |
| NLU_USERNAME  | The username used for API requests |
| NLU_PASSWORD  | The password used for API requests |
| SENTRY_DSN    | Where to send exceptions to |
