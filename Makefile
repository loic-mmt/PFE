.PHONY: setup check test build example plots

setup:
	@python3 scripts/fix_macos_pth.py
	uv sync --extra lse --extra plots --extra calendars --extra commodities
	@python3 scripts/fix_macos_pth.py

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run --extra plots --extra calendars --extra commodities pytest

test:
	uv run --extra plots --extra calendars --extra commodities pytest

build:
	uv build

example:
	uv run --extra calendars python examples/daily_market.py
	uv run python examples/context_features.py
	uv run python examples/document_features.py
	uv run python examples/international_features.py
	uv run python examples/model_datasets.py
	uv run python examples/data_qualification.py
	uv run python examples/quickstart.py
	uv run python examples/features.py
	uv run python examples/linear_features.py
	uv run python examples/macro_features.py
	uv run python examples/option_features.py

plots:
	uv run --extra plots python examples/plots.py
