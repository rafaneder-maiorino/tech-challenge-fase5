.PHONY: install download audit prepare mlflow-server train tune serve lint format test all build-manual eval-retrieval eval-guardrails

install:  ## create the environment (uv sync)
	uv sync

download:  ## fetch the raw CSV and verify its SHA-256
	uv run python -m support_copilot.data.source

audit:  ## dataset acceptance audit -> reports/dataset_audit.md
	uv run python scripts/dataset_audit.py --files v5_multi v4_multi_20k

prepare:  ## clean + paraphrase-family split + German drift pool -> data/processed/
	uv run python -m support_copilot.data.prepare

mlflow-server:  ## local MLflow on :5001 (5000 is AirPlay on macOS)
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5001

train:  ## model selection + champion export (needs `make mlflow-server` in another terminal)
	uv run python -m support_copilot.model.train

tune:  ## linear-model sweep under grouped CV -> reports/tuning_linear.md
	uv run python scripts/tune_linear.py

serve:  ## API on :8000, Swagger at /docs
	uv run uvicorn support_copilot.api.main:app --port 8000

lint:  ## ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -q

all: download prepare lint test

build-manual:  ## rebuild the OPNsense PDF (needs Google Chrome)
	uv run python scripts/build_manual.py

eval-retrieval:  ## chunk-size comparison + refusal threshold -> reports/retrieval_eval.md
	uv run python scripts/eval_retrieval.py

eval-guardrails:  ## attack suite + holdout + external set -> reports/guardrails_eval.md
	uv run python scripts/eval_guardrails.py
