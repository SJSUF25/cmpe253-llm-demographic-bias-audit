.DEFAULT_GOAL := help

UV ?= uv
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
PYTHONPATH := $(CURDIR)/src
export UV_CACHE_DIR
export PYTHONPATH
PROJECT_CLI := $(UV) run --frozen python -m project_cli

.PHONY: help check-uv setup setup-colab capture-environment smoke-model governance-lock storage-init storage-verify sync-upstream \
	download-data build-data validate-data pilot \
	infer-baseline infer-mitigation score annotate-export analyze explain \
	privacy test lint demo reproduce-small

help: ## Show available project commands
	@awk 'BEGIN {FS = ":.*## "; printf "Available targets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

check-uv: ## Verify that uv is installed
	@command -v $(UV) >/dev/null 2>&1 || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }

setup: check-uv ## Create the Python 3.11 environment and install development dependencies
	$(UV) sync --group dev

setup-colab: check-uv ## Install the exact Colab/GPU dependency group from uv.lock
	$(UV) sync --frozen --group dev --group colab

capture-environment: ## Write a hardware/software manifest (OUTPUT optional)
	$(UV) run --frozen --group colab python -m environment_tools capture --output "$(or $(OUTPUT),results/manifests/environment.json)"

smoke-model: ## Run deterministic GPU smoke test (MODEL=M1 or M2; OUTPUT required)
	@test -n "$(MODEL)" || { echo "Usage: make smoke-model MODEL=M1 OUTPUT=/path/result.json"; exit 2; }
	@test -n "$(OUTPUT)" || { echo "Usage: make smoke-model MODEL=M1 OUTPUT=/path/result.json"; exit 2; }
	$(UV) run --frozen --group colab python -m inference.smoke_test --model-key "$(MODEL)" --output "$(OUTPUT)"

governance-lock: ## Hash the approved PRD and frozen governance/config inputs
	$(UV) run --frozen python -m governance_tools lock --output results/manifests/governance_lock.json

storage-init: ## Initialize and probe shared storage (ROOT and AUTHOR are required)
	@test -n "$(ROOT)" || { echo "Usage: make storage-init ROOT=/path AUTHOR=name"; exit 2; }
	@test -n "$(AUTHOR)" || { echo "Usage: make storage-init ROOT=/path AUTHOR=name"; exit 2; }
	$(UV) run --frozen python -m storage_tools init --root "$(ROOT)" --actor "$(AUTHOR)"

storage-verify: ## Verify a shared-storage probe (ROOT and REVIEWER are required)
	@test -n "$(ROOT)" || { echo "Usage: make storage-verify ROOT=/path REVIEWER=name"; exit 2; }
	@test -n "$(REVIEWER)" || { echo "Usage: make storage-verify ROOT=/path REVIEWER=name"; exit 2; }
	$(UV) run --frozen python -m storage_tools verify --root "$(ROOT)" --actor "$(REVIEWER)"

sync-upstream: ## Verify pinned revisions and refresh upstream metadata snapshots
	$(UV) run --frozen python -m upstream_tools sync

download-data: ## Download immutable upstream datasets
	$(PROJECT_CLI) download-data

build-data: ## Build deterministic processed datasets
	$(PROJECT_CLI) build-data

validate-data: ## Validate dataset schemas, counts, hashes, and split integrity
	$(PROJECT_CLI) validate-data

pilot: ## Run the preregistered model and data pilot
	$(PROJECT_CLI) pilot

infer-baseline: ## Run locked baseline inference
	$(PROJECT_CLI) infer-baseline

infer-mitigation: ## Run locked fairness-control inference
	$(PROJECT_CLI) infer-mitigation

score: ## Run automated scorers and BBQ likelihood scoring
	$(PROJECT_CLI) score

annotate-export: ## Export the blinded human-annotation sample
	$(PROJECT_CLI) annotate-export

analyze: ## Run the locked statistical analysis
	$(PROJECT_CLI) analyze

explain: ## Run local/global explanation and validation analyses
	$(PROJECT_CLI) explain

privacy: ## Run synthetic-PII and controlled-membership experiments
	$(PROJECT_CLI) privacy

test: check-uv ## Run unit and integration tests
	$(UV) run --frozen pytest

lint: check-uv ## Run static checks and formatting verification
	$(UV) run --frozen ruff check .
	$(UV) run --frozen ruff format --check .

demo: ## Start the cached Streamlit presentation
	$(PROJECT_CLI) demo

reproduce-small: ## Run the clean small end-to-end reproduction
	$(PROJECT_CLI) reproduce-small
