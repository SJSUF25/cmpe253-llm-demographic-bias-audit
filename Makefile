.DEFAULT_GOAL := help

UV ?= uv
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
PYTHONPATH := $(CURDIR)/src
export UV_CACHE_DIR
export PYTHONPATH
PROJECT_CLI := $(UV) run --frozen python -m project_cli

.PHONY: help check-uv setup setup-colab capture-environment smoke-model governance-lock storage-init storage-verify sync-upstream build-synthetic-pii build-membership run-cab score-cab annotation-packet annotation-review annotation-freeze analysis-plan-check run-pii score-pii score-h5 privacy-audit-packet score-helpfulness score-adapter-utility compare-adapter-utility train-canary attack-canary token-length-check cab-quality-check privacy-controls-smoke \
	download-data build-data build-bbq validate-data pilot \
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

download-data: ## Download immutable upstream datasets (ROOT required)
	@test -n "$(ROOT)" || { echo "Usage: make download-data ROOT=/path [VERIFY_REMOTE=1]"; exit 2; }
	$(PROJECT_CLI) download-data --root "$(ROOT)" $(if $(VERIFY_REMOTE),--verify-remote,)

build-data: ## Build deterministic processed datasets (ROOT required)
	@test -n "$(ROOT)" || { echo "Usage: make build-data ROOT=/path"; exit 2; }
	$(PROJECT_CLI) build-data --root "$(ROOT)"

build-bbq: ## Build the frozen BBQ sample and option permutations (ROOT required)
	@test -n "$(ROOT)" || { echo "Usage: make build-bbq ROOT=/path"; exit 2; }
	$(PROJECT_CLI) build-bbq --root "$(ROOT)"

build-synthetic-pii: ## Build deterministic private synthetic PII records (ROOT required)
	@test -n "$(ROOT)" || { echo "Usage: make build-synthetic-pii ROOT=/path"; exit 2; }
	$(UV) run --frozen python -m privacy.synthetic_data --root "$(ROOT)"

privacy-controls-smoke: ## Check pinned Presidio redaction on a development fixture
	$(UV) run --frozen --group colab python -m privacy.controls --smoke

build-membership: ## Build the sealed ordinary and canary corpus (ROOT required)
	@test -n "$(ROOT)" || { echo "Usage: make build-membership ROOT=/path"; exit 2; }
	$(UV) run --frozen python -m privacy.membership_data --root "$(ROOT)"

run-cab: ## Run one registered CAB model/condition (ROOT, RUN_ID, SPLIT required)
	@test -n "$(ROOT)" || { echo "Usage: make run-cab ROOT=/path RUN_ID=CAB-EXP-M1-B0 SPLIT=development"; exit 2; }
	@test -n "$(RUN_ID)" || { echo "RUN_ID is required"; exit 2; }
	@test -n "$(SPLIT)" || { echo "SPLIT=development or test is required"; exit 2; }
	$(UV) run --frozen --group colab python -m inference.runner --root "$(ROOT)" --run-id "$(RUN_ID)" --split "$(SPLIT)"

score-cab: ## Score one completed CAB run with pinned secondary scorers
	@test -n "$(ROOT)" || { echo "Usage: make score-cab ROOT=/path RUN_ID=CAB-EXP-M1-B0 SPLIT=development"; exit 2; }
	@test -n "$(RUN_ID)" || { echo "RUN_ID is required"; exit 2; }
	@test -n "$(SPLIT)" || { echo "SPLIT=development or test is required"; exit 2; }
	$(UV) run --frozen --group colab python -m scoring.cab --root "$(ROOT)" --run-id "$(RUN_ID)" --split "$(SPLIT)"

annotation-packet: ## Build a blinded CAB review packet after four runs (ROOT, SPLIT, ANNOTATOR)
	@test -n "$(ROOT)" && test -n "$(SPLIT)" && test -n "$(ANNOTATOR)" || { echo "Usage: make annotation-packet ROOT=/path SPLIT=development ANNOTATOR=vineet"; exit 2; }
	$(UV) run --frozen python -m scoring.annotation --root "$(ROOT)" --split "$(SPLIT)" --annotator "$(ANNOTATOR)"

annotation-review: ## Resume blinded CAB ratings (PACKET required)
	@test -n "$(PACKET)" || { echo "PACKET is required"; exit 2; }
	$(UV) run --frozen python -m scoring.annotation_review review --packet "$(PACKET)"

annotation-freeze: ## Export completed ratings and a checksum manifest (PACKET required)
	@test -n "$(PACKET)" || { echo "PACKET is required"; exit 2; }
	$(UV) run --frozen python -m scoring.annotation_review export --packet "$(PACKET)"

analysis-plan-check: ## Validate the exact 48-test preregistered analysis plan
	$(UV) run --frozen python -m analysis.plan

run-pii: ## Run one registered private PII condition (ROOT, RUN_ID, SPLIT required)
	@test -n "$(ROOT)" || { echo "Usage: make run-pii ROOT=/path RUN_ID=PII-M1-P0 SPLIT=development"; exit 2; }
	@test -n "$(RUN_ID)" || { echo "RUN_ID is required"; exit 2; }
	@test -n "$(SPLIT)" || { echo "SPLIT=development or test is required"; exit 2; }
	$(UV) run --frozen --group colab python -m privacy.evaluation --root "$(ROOT)" --run-id "$(RUN_ID)" --split "$(SPLIT)"

score-pii: ## Aggregate privacy exposure after all six test runs (ROOT required)
	@test -n "$(ROOT)" || { echo "Usage: make score-pii ROOT=/path"; exit 2; }
	$(UV) run --frozen python -m privacy.metrics --root "$(ROOT)"

score-h5: ## Evaluate preregistered paired privacy exposure and utility (ROOT required)
	@test -n "$(ROOT)" || { echo "Usage: make score-h5 ROOT=/path"; exit 2; }
	$(UV) run --frozen --group colab python -m privacy.h5 --root "$(ROOT)"

privacy-audit-packet: ## Prepare 50-record blinded privacy helpfulness review
	@test -n "$(ROOT)" || { echo "Usage: make privacy-audit-packet ROOT=/path"; exit 2; }
	$(UV) run --frozen python -m privacy.utility --root "$(ROOT)" audit-packet

score-helpfulness: ## Aggregate 300 private blinded helpfulness ratings
	@test -n "$(ROOT)" || { echo "Usage: make score-helpfulness ROOT=/path"; exit 2; }
	$(UV) run --frozen python -m privacy.utility --root "$(ROOT)" score-helpfulness

score-adapter-utility: ## Score ordinary tasks before/after adapter (MODE required)
	@test -n "$(ROOT)" || { echo "Usage: make score-adapter-utility ROOT=/path MODE=base"; exit 2; }
	@test -n "$(MODE)" || { echo "MODE=base or MODE=adapter is required"; exit 2; }
	$(UV) run --frozen --group colab python -m privacy.utility --root "$(ROOT)" adapter-utility --mode "$(MODE)"

compare-adapter-utility: ## Compare the two ordinary utility run summaries
	@test -n "$(ROOT)" || { echo "Usage: make compare-adapter-utility ROOT=/path"; exit 2; }
	$(UV) run --frozen python -m privacy.utility --root "$(ROOT)" compare-adapter-utility

train-canary: ## Train or resume the controlled Phi QLoRA adapter (ROOT required)
	@test -n "$(ROOT)" || { echo "Usage: make train-canary ROOT=/path"; exit 2; }
	$(UV) run --frozen --group colab python -m privacy.train_canary --root "$(ROOT)"

attack-canary: ## Score controlled canary attacks (ROOT and MODE=base/adapter required)
	@test -n "$(ROOT)" || { echo "Usage: make attack-canary ROOT=/path MODE=base"; exit 2; }
	@test -n "$(MODE)" || { echo "MODE=base or MODE=adapter is required"; exit 2; }
	$(UV) run --frozen --group colab python -m privacy.canary_attack --root "$(ROOT)" --mode "$(MODE)"

validate-data: ## Reverify all built datasets and create the private manual-QA packet
	@test -n "$(ROOT)" || { echo "Usage: make validate-data ROOT=/path"; exit 2; }
	$(PROJECT_CLI) validate-data --root "$(ROOT)"

cab-quality-check: ## Freeze and verify implicit CAB exclusions before inference
	@test -n "$(ROOT)" || { echo "Usage: make cab-quality-check ROOT=/path"; exit 2; }
	$(UV) run --frozen python -m data.cab_quality --root "$(ROOT)"

token-length-check: ## Check pinned chat-template lengths for inference and training
	@test -n "$(ROOT)" || { echo "Usage: make token-length-check ROOT=/path"; exit 2; }
	$(UV) run --frozen --group colab python -m data.token_lengths --root "$(ROOT)"

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
