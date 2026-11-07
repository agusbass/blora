.PHONY: install dev-install test test-cov lint format typecheck check clean help

# ─── Installation ─────────────────────────────────────────────────
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

# ─── Testing ──────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v --tb=short $(ARGS)

test-cov:
	python -m pytest tests/ --cov=src/blora --cov-report=term-missing -v

# ─── Code Quality ─────────────────────────────────────────────────
lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

typecheck:
	mypy src/

# Run everything CI runs, locally.
check: lint typecheck test

# ─── Cleanup ──────────────────────────────────────────────────────
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache *.egg-info dist build htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ─── Help ─────────────────────────────────────────────────────────
help:
	@echo "BloRA Makefile"
	@echo "  make dev-install   — editable install + dev tools + pre-commit hooks"
	@echo "  make test          — run pytest"
	@echo "  make test-cov      — run pytest with coverage"
	@echo "  make lint          — ruff check"
	@echo "  make format        — ruff format"
	@echo "  make typecheck     — mypy"
	@echo "  make check         — lint + typecheck + test (what CI runs)"
	@echo "  make clean         — remove caches and build artifacts"
