# MomConnect Intent Classifier

The **MomConnect Intent Classifier** labels inbound messages from mothers into high-level intents.  
It is used by the Department of Health to triage and route messages appropriately (e.g. *service feedback*, *sensitive exits like baby loss or opt-outs*, etc.).

⚠️ **Note:** This service is **internal only** and not exposed outside the cluster.

---

## Features

- **Sentence embeddings** (via [SentenceTransformers](https://www.sbert.net/))
- **Linear classifier head** trained on embeddings
- **Threshold-based decisioning** per intent
- **Policy-driven biasing**:
  - *Sensitive exits (baby loss, opt-out)* → prioritise **recall**
  - *Service feedback* → prioritise **precision**
  - *Noise/Spam* → strict filtering
- **Automatic enrichment**:
  - Service feedback → sentiment polarity (positive/negative/neutral)
  - Sensitive exit → bereavement vs generic opt-out
- **Review band**: low-confidence predictions are flagged as `NEEDS_REVIEW`

---

## Development

This project uses [Poetry](https://python-poetry.org/docs/#installation) for packaging and dependency management.

1. Ensure you’re running **Python 3.11+**:
   ```bash
   python --version
Install dependencies:

bash
Copy
Edit
poetry install
Running the Flask worker
Set the environment variables and start the app:

bash
Copy
Edit
export NLU_USERNAME=your-username
export NLU_PASSWORD=your-password

poetry run flask --app src.application run
Code Quality
Autoformat + Linting:

bash
Copy
Edit
poetry run ruff format .
poetry run ruff check .
poetry run mypy --install-types
Tests:

bash
Copy
Edit
poetry run pytest
Regenerating Embeddings
Delete the existing embeddings JSON in src/data/

Update training examples in nlu.yaml

Run the Flask app:

bash
Copy
Edit
poetry run flask --app src.application run
→ Embeddings will be regenerated.

Threshold Tuning
Thresholds live in artifacts/thresholds.json.
They can be tuned with validation data via:

bash
Copy
Edit
poetry run python src/fit_thresholds.py \
  --model-dir artifacts/ \
  --nlu-path src/data/nlu.yaml \
  --validation-path src/data/validation.yaml
Policy:

Sensitive exits → low threshold (recall focus)

Service feedback → higher threshold (precision focus)

Noise → strict threshold

Other → balanced

Review band → learned dynamically from disagreement zone

Editor Configuration
For VS Code:

Install the Python and Ruff extensions.

Settings:

"python.linting.mypyEnabled": true

"python.formatting.provider": "black"

"editor.formatOnSave": true

Release Process
Merge all PRs & complete QA.

Update release notes in CHANGELOG.md.

On main branch:

Update version in pyproject.toml

Replace UNRELEASED with release version + date in CHANGELOG.md

Commit + tag:

bash
Copy
Edit
git tag v0.2.1
git push origin main --tags
Post-release:

Increment version in pyproject.toml (e.g. 0.2.2.dev0)

Add new UNRELEASED header to CHANGELOG.md

Running in Production
A Docker image is published for deployment.

Required environment variables:

Variable	Description
NLU_USERNAME	Username for API requests
NLU_PASSWORD	Password for API requests
SENTRY_DSN	Sentry DSN for error reporting

License
MIT

yaml
Copy
Edit

---

Do you also want me to add an **API Quickstart** section (sample `curl` + example JSON response) so QA/SxD can test it directly without spinning up notebooks?
