.PHONY: install download audit prepare lint format test all

install:  ## create the environment (uv sync)
	uv sync

download:  ## fetch the raw CSV and verify its SHA-256
	uv run python -m support_copilot.data.source

audit:  ## dataset acceptance audit -> reports/dataset_audit.md
	uv run python scripts/dataset_audit.py --files v5_multi v4_multi_20k

prepare:  ## clean + stratified split + German drift pool -> data/processed/
	uv run python -m support_copilot.data.prepare

lint:  ## ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -q

all: download prepare lint test
