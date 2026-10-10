.PHONY: setup check test build example plots

setup:
	@python3 scripts/fix_macos_pth.py
	uv sync --extra lse --extra plots
	@python3 scripts/fix_macos_pth.py

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run --extra plots pytest

test:
	uv run --extra plots pytest

build:
	uv build

example:
	uv run python examples/quickstart.py
	uv run python examples/features.py
	uv run python examples/linear_features.py
	uv run python examples/macro_features.py

plots:
	uv run --extra plots python examples/plots.py
