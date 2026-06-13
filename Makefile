# Makefile for the Buddy Jr RL Lab
# Learner-friendly targets for install, lint, type-check, test, docs, sim, and viz.
#
# Usage:
#   make          — print this help
#   make install  — create .venv and install rl-lab[dev]
#   make lint     — run ruff
#   make type     — run mypy
#   make test     — run pytest
#   make docs     — serve MkDocs docs locally
#   make sim      — launch the hello-sim via the CLI
#   make foxglove — launch the Foxglove bridge via the CLI
#   make clean    — remove all generated artefacts

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Override with: make VENV=/path/to/venv install
VENV ?= .venv

# Python to bootstrap the venv with.  Requires python3.12 on the PATH.
PYTHON ?= python3.12

# Derived paths — all tools are run from inside the venv so that the correct
# package versions are always used, regardless of what is active globally.
VENV_BIN  := $(VENV)/bin
PIP       := $(VENV_BIN)/pip
RUFF      := $(VENV_BIN)/ruff
MYPY      := $(VENV_BIN)/mypy
PYTEST    := $(VENV_BIN)/pytest
MKDOCS    := $(VENV_BIN)/mkdocs
RL_LAB    := $(VENV_BIN)/rl-lab

# ---------------------------------------------------------------------------
# .PHONY declarations — these targets never produce a file of the same name
# ---------------------------------------------------------------------------

.PHONY: help install lint type test docs sim foxglove clean

# ---------------------------------------------------------------------------
# help  (default goal)
# ---------------------------------------------------------------------------

# Prints every target that has a '## ' comment on the same or preceding line.
# Pattern: look for lines like "target: ... ## description" in this Makefile.
.DEFAULT_GOAL := help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

install: $(VENV)/bin/activate  ## Create .venv with python3.12 (if missing) and pip install -e .[dev]

# The activate script is the canonical sentinel for a complete venv.
$(VENV)/bin/activate:
	@echo ">>> Creating virtual environment in $(VENV) using $(PYTHON)"
	$(PYTHON) -m venv $(VENV)
	@echo ">>> Installing rl-lab[dev] in editable mode"
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo ">>> Installation complete.  Run 'source $(VENV)/bin/activate' to activate."

# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------

lint: $(VENV)/bin/activate  ## Run ruff check on the whole codebase
	$(RUFF) check .

# ---------------------------------------------------------------------------
# type
# ---------------------------------------------------------------------------

type: $(VENV)/bin/activate  ## Run mypy on the rl_lab package
	$(MYPY) rl_lab

# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

test: $(VENV)/bin/activate  ## Run pytest in quiet mode
	$(PYTEST) -q

# ---------------------------------------------------------------------------
# docs
# ---------------------------------------------------------------------------

docs: $(VENV)/bin/activate  ## Serve the MkDocs documentation site locally (Ctrl-C to stop)
	$(MKDOCS) serve

# ---------------------------------------------------------------------------
# sim
# ---------------------------------------------------------------------------

# Note: scripts/launch_hello_sim.py is an M2 deliverable (not yet landed).
# Until it exists the CLI stub (rl-lab sim) is the runnable entry point.
sim: $(VENV)/bin/activate  ## Launch the Buddy Jr hello-sim via the rl-lab CLI
	$(RL_LAB) sim

# ---------------------------------------------------------------------------
# foxglove
# ---------------------------------------------------------------------------

# Note: scripts/launch_foxglove_bridge.py is an M2 deliverable (not yet landed).
# Until it exists the CLI stub (rl-lab viz) is the runnable entry point.
foxglove: $(VENV)/bin/activate  ## Launch the Foxglove bridge via the rl-lab CLI
	$(RL_LAB) viz

# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------

clean:  ## Remove all generated artefacts (__pycache__, caches, build dirs, recordings)
	@echo ">>> Removing Python bytecode and tool caches"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	@echo ">>> Removing build and packaging artefacts"
	rm -rf build dist
	find . -maxdepth 3 -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	@echo ">>> Removing training runs and MCAP recordings"
	rm -rf runs/
	find . -name '*.mcap' -delete
	@echo ">>> Done."
