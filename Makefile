# ======================================
# MomConnect Intent Classifier v2.0
# ======================================

PYTHON := python3
SRC := src
DATA := $(SRC)/data
MAPPED_DATA := $(SRC)/mapped_data
ARTIFACTS := $(SRC)/artifacts

VENV := .venv
UV := uv

.PHONY: help install lint typecheck test datasets train serve serve-dev clean

help:
	@echo "MomConnect Intent Classifier v2.0"
	@echo "--------------------------------"
	@echo "make install      - install deps"
	@echo "make lint         - run ruff lint + format check"
	@echo "make typecheck    - run mypy type checker"
	@echo "make test         - run pytest"
	@echo "make datasets     - emit JSONL from source YAML"
	@echo "make train        - train new model"
	@echo "make serve-dev    - run Flask app for LOCAL development"
	@echo "make serve        - run Flask app with Gunicorn (for production)"
	@echo "make clean        - remove build artifacts"

install:
	$(UV) pip install -r requirements.txt

lint:
	ruff check $(SRC)
	ruff format $(SRC)

typecheck:
	mypy $(SRC)

test:
	pytest -vv --disable-warnings

datasets:
	$(PYTHON) $(SRC)/data/build_datasets.py --data-dir $(DATA) --emit-jsonl --out-dir $(MAPPED_DATA)

train:
	$(PYTHON) $(SRC)/train_model.py --data-path $(MAPPED_DATA)/samples.train.jsonl --artifacts-dir $(ARTIFACTS)

# Use this for local development on macOS to avoid MPS/forking issues
serve-dev:
	FLASK_APP=src/application.py poetry run flask run -p 5001

# This is for production-like environments (e.g., Linux containers)
serve:
	OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES gunicorn --workers 2 --bind 0.0.0.0:5001 --preload src.application:app

clean:
	rm -rf $(ARTIFACTS) __pycache__ .pytest_cache .mypy_cache

