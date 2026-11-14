PYTHON ?= python
SRC    := src tests

.PHONY: help install lint format typecheck test check run clean

help:
	@echo "Targets:"
	@echo "  install    Install the package in editable mode with dev tools"
	@echo "  lint       Check imports (isort), formatting (black) and types (pyright)"
	@echo "  format     Sort imports (isort) and format the code (black)"
	@echo "  typecheck  Run pyright"
	@echo "  test       Run tests with coverage (terminal, htmlcov/, coverage.xml)"
	@echo "  check      Run lint and test (everything CI should run)"
	@echo "  run        Train a demo model on synthetic data"
	@echo "  clean      Remove caches and build artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint: typecheck
	$(PYTHON) -m pylint $(SRC)
	$(PYTHON) -m isort --check-only --diff $(SRC)
	$(PYTHON) -m black --check $(SRC)

format:
	$(PYTHON) -m isort $(SRC)
	$(PYTHON) -m black $(SRC)

typecheck:
	$(PYTHON) -m pyright $(SRC)

test:
	$(PYTHON) -m coverage run -m unittest discover -s tests -v
	$(PYTHON) -m coverage report
	$(PYTHON) -m coverage html
	$(PYTHON) -m coverage xml

check: lint test

run:
	$(PYTHON) -m wldstk_ml

clean:
	rm -rf build dist *.egg-info src/*.egg-info .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
