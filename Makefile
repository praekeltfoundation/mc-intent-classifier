# ======================================
# MomConnect Intent Classifier v2.0
# ======================================

# Using 'poetry run python3' ensures we use the venv's Python
PYTHON := poetry run python3
SRC := src
TESTS := tests
DATA := $(SRC)/data
MAPPED_DATA := $(SRC)/mapped_data
ARTIFACTS := $(SRC)/artifacts

VENV := .venv
UV := uv

# Fix for transformers/tiktoken loading issue on some systems
export TOKENIZERS_PARALLELISM=false

.PHONY: help install lint typecheck test build-jsonl migrate-legacy train tune-thresholds evaluate serve serve-dev clean all

help:
	@echo "MomConnect Intent Classifier v2.0"
	@echo "--------------------------------"
	@echo "make install          - Install dependencies"
	@echo "make lint             - Run ruff linter and formatter with auto-fix"
	@echo "make typecheck        - Run mypy type checker"
	@echo "make test             - Run pytest for unit tests"
	@echo "make all              - Run build-jsonl, train, tune, and evaluate in sequence"
	@echo ""
	@echo "--- Model Workflow ---"
	@echo "make build-jsonl      - (Safe) Generate JSONL files from manually-edited mapped YAMLs."
	@echo "make train            - Train a new model on the training set."
	@echo "make tune-thresholds  - Tune confidence thresholds on the validation set."
	@echo "make evaluate         - Evaluate the final model on the test set."
	@echo ""
	@echo "--- Data Migration (Use with Caution) ---"
	@echo "make migrate-legacy   - (Destructive) Overwrite mapped YAMLs from original legacy data."
	@echo ""
	@echo "--- Service ---"
	@echo "make serve-dev        - Run Flask app for LOCAL development"
	@echo "make serve            - Run Flask app with Gunicorn (for production)"
	@echo "make clean            - Remove build artifacts"

install:
	poetry install

lint:
	poetry run ruff check $(SRC) $(TESTS) --fix
	poetry run ruff format $(SRC) $(TESTS)

typecheck:
	poetry run mypy $(SRC) $(TESTS)

test:
	poetry run pytest -vv --disable-warnings

# New SAFE command for generating JSONL from the mapped YAMLs
build-jsonl:
	$(PYTHON) $(SRC)/data/build_datasets.py --jsonl-only \
		--data-dir $(MAPPED_DATA) \
		--files "nlu.yaml" "validation.yaml" "test.yaml" \
		--out-dir $(MAPPED_DATA)

# Renamed DESTRUCTIVE command with a safeguard
migrate-legacy:
	@echo "🔴 WARNING: This is a destructive operation."
	@echo "It will overwrite the manually-edited files in '$(MAPPED_DATA)'"
	@echo "with newly generated content from the legacy '$(DATA)' directory."
	@read -p "Do you genuinely want to proceed? (y/n) " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "Proceeding with migration..."; \
		$(PYTHON) $(SRC)/data/build_datasets.py \
			--data-dir $(DATA) \
			--emit-jsonl \
			--out-dir $(MAPPED_DATA); \
	else \
		echo "Migration cancelled."; \
	fi

train:
	$(PYTHON) $(SRC)/train_model.py --data-path $(MAPPED_DATA)/samples.train.jsonl --artifacts-dir $(ARTIFACTS)

tune-thresholds:
	$(PYTHON) $(SRC)/evaluate_model.py tune --data-path $(MAPPED_DATA)/samples.validation.jsonl --artifacts-dir $(ARTIFACTS)

evaluate:
	$(PYTHON) $(SRC)/evaluate_model.py report --data-path $(MAPPED_DATA)/samples.test.jsonl --artifacts-dir $(ARTIFACTS)

# Full pipeline now uses the safe build command
all: build-jsonl train tune-thresholds evaluate

# Use this for local development on macOS to avoid MPS/forking issues
serve-dev:
	FLASK_APP=src/application.py poetry run flask run -p 5001

# This is for production-like environments (e.g., Linux containers)
serve:
	OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES poetry run gunicorn --workers 2 --bind 0.0.0.0:5001 --preload src.application:app

clean:
	rm -rf $(ARTIFACTS) __pycache__ .pytest_cache .mypy_cache
