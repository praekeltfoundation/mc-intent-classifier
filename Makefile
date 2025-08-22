# ======================================
# MomConnect Intent Classifier v2.0
# ======================================

PYTHON := python3
SRC := src
DATA := $(SRC)/data
ARTIFACTS := $(SRC)/artifacts

VENV := .venv
UV := uv

# Default encoder (can override: make train ENCODER=all-MiniLM-L6-v2)
ENCODER ?= BAAI/bge-m3

.PHONY: help venv install lint typecheck test train thresholds serve clean

help:
	@echo "MomConnect Intent Classifier v2.0"
	@echo "--------------------------------"
	@echo "make venv         - create virtualenv with uv"
	@echo "make install      - install deps"
	@echo "make lint         - run ruff lint + format check"
	@echo "make typecheck    - run mypy type checker"
	@echo "make test         - run pytest"
	@echo "make train        - train new model"
	@echo "make thresholds   - tune thresholds"
	@echo "make serve        - run Flask app (dev mode)"
	@echo "make clean        - remove build artifacts"

venv:
	$(UV) venv $(VENV)

install:
	$(UV) pip install -r requirements.txt

lint:
	ruff check $(SRC)
	ruff format --check $(SRC)

typecheck:
	mypy $(SRC)

test:
	pytest -q --disable-warnings

train:
	$(PYTHON) $(SRC)/train.py --data-dir $(DATA) --artifacts-dir $(ARTIFACTS) --encoder $(ENCODER)

thresholds:
	$(PYTHON) $(SRC)/fit_thresholds.py --model-dir $$(ls -td $(ARTIFACTS)/mcic-* | head -1)

serve:
	FLASK_APP=$(SRC)/application.py FLASK_ENV=development $(PYTHON) -m flask run --host=0.0.0.0 --port=8000

clean:
	rm -rf $(ARTIFACTS)/*.pkl $(ARTIFACTS)/*.json __pycache__ .pytest_cache .mypy_cache
