# CMPE 253 LLM Demographic Bias Audit

Reproducible, offline pipeline for evaluating privacy, fairness, explainability, and
robustness in two open-weight conversational language models.

## Development setup

Requirements:

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/)
- GNU Make or compatible `make`

Run:

```bash
make setup
make test
make lint
```

`make reproduce-small` runs a clean, offline CPU contract fixture through CAB
pair expansion, BBQ option-order scoring, aggregation, and an exact paired test.
It writes `build/reproduce_small/report.json`. The fixture uses a deterministic
stand-in likelihood function, so its numbers verify code wiring and are **not**
model findings or a substitute for the pinned GPU experiments.

Use `make help` to list the complete project workflow. Experiment commands that have
not yet been implemented exit with a clear status instead of silently succeeding.

## Governance setup

Pinned public upstream metadata, cards, licenses, and configuration snapshots are
refreshed and verified with:

```bash
make sync-upstream
```

After the private shared Drive folder has been created and mounted, initialize it and
verify its contents. Record the actual reviewer; a repeat by Vineet is a reproducibility
check, not an independent human review:

```bash
make storage-init ROOT="/mounted/shared/path" AUTHOR="Vineet Malewar"
make storage-verify ROOT="/mounted/shared/path" REVIEWER="Vineet Malewar (repeat check)"
```

The mount path belongs in the ignored local environment, never in repository files.
See `docs/data_handling.md` for the storage layout and handling rules.

After the governance gate is approved, download or re-verify the pinned CAB and BBQ
sources without overwriting existing raw bytes:

```bash
make download-data ROOT="$AUDIT_SHARED_ROOT"
make download-data ROOT="$AUDIT_SHARED_ROOT" VERIFY_REMOTE=1
make build-data ROOT="$AUDIT_SHARED_ROOT"
make build-bbq ROOT="$AUDIT_SHARED_ROOT"
make build-synthetic-pii ROOT="$AUDIT_SHARED_ROOT"
make build-membership ROOT="$AUDIT_SHARED_ROOT"
make validate-data ROOT="$AUDIT_SHARED_ROOT"
make token-length-check ROOT="$AUDIT_SHARED_ROOT"
make cab-quality-check ROOT="$AUDIT_SHARED_ROOT"
make privacy-controls-smoke
```

The command checks pinned revisions, byte sizes, SHA-256 values, schemas, row counts,
CAB race/gender inventory, and all three required BBQ categories before writing an
immutable raw-data manifest.
`make build-data` expands the linked CAB explicit/implicit families into a private,
deterministic prompt manifest; it refuses a source-hash mismatch or changed existing
output. The tokenizer-length check uses the pinned official chat templates for every
CAB and BBQ input and the Phi adapter training sequences. It produces a separate
private receipt; the data review gate must also pass before confirmatory inference.
The CAB quality check freezes a source-preserving exclusion of 23 implicit families
with answer-like source content. It retains all explicit primary prompts and
requires a separate quality-manifest hash in the data-gate receipt.
The accelerated AI review is documented in `docs/data_qa_ai_review.md`. After
the quality check passes and the current PRD is copied to Drive, run
`make ai-review-gate ROOT="$AUDIT_SHARED_ROOT"` on a CPU Colab runtime. This
writes a hash-bound receipt labeled `ai_assisted_no_human_review`; the final
report must disclose that no independent human semantic review was performed.
`notebooks/e1_t06_data_gate_colab.ipynb` remains available if Abhinand later
performs a separate human review. Its old 85-pass log is not approval evidence.
The privacy-controls smoke command checks the pinned Presidio and spaCy models
against one synthetic development record; see `docs/privacy_controls.md`.
The six registered PII inference cells and aggregate exposure scorer are described
in `docs/privacy_evaluation.md`. Test runs require the private data-gate receipt.
The controlled Phi QLoRA training/resume procedure is in
`docs/canary_training.md`.
The controlled canary extraction and membership attacks are documented in
`docs/canary_attack.md`.

The gated E2-T04 model/data pilot is documented in `docs/pilot.md`. GPU execution
and its AI-assisted feasibility decision are recorded in `docs/pilot_v2_review.md`.
The full E2-T05/T06 CAB and BBQ execution is prepared in
`notebooks/e2_t05_t06_full_colab.ipynb`; it requires the revised Drive PRD,
an L4 runtime, and a renewed v3 data-gate receipt.
After that full run is finished, `notebooks/e6_privacy_colab.ipynb` executes the
original proposal's PII and controlled-canary experiments under a newly hash-bound
v4 data gate. Do not pull its newer commit into an interrupted CAB/BBQ run.
Pinned CAB secondary diagnostics can be run with `make score-cab` after a CAB
inference cell completes; see `docs/cab_scoring.md`.
After all eight test scoring manifests exist, run
`make report-cab-secondary ROOT=/mounted/project` to verify their private outputs
and write aggregate-only
diagnostic tables. Classifier scores remain uncalibrated secondary measures.
After all full CAB/BBQ cells complete, `make score-fairness-descriptive` produces
hash-verified subgroup and intersectional tables with scenario/template-cluster
intervals; see `docs/fairness_descriptive.md`. It does not replace the locked
confirmatory analysis or human ratings. The full analysis sequence is in
`notebooks/e3_e4_analysis_colab.ipynb`.
The bounded local/global BBQ input-ablation workflow is in
`notebooks/e5_explanations_colab.ipynb`; see `docs/explanation_ablation.md`.
The CAB robustness60 source selection and neutral/adversarial input preparation
are documented in `docs/robustness_subset.md`; GPU execution is in
`notebooks/e7_robustness_colab.ipynb`.
The H5 privacy decision, blinded review packet, and adapter utility commands are
documented in `docs/privacy_h5.md`.

The completed E6 outputs can be checked without a GPU. The first command
regenerates the sealed synthetic input hashes and recomputes canary, adapter
utility, and PII exposure totals from the saved row scores. The second computes
the 12 BBQ portions of the frozen H1/H2 test inventory. Its p values are
unadjusted until the corresponding CAB ratings complete each 18-test family.

```bash
make audit-privacy-receipts ROOT="$AUDIT_SHARED_ROOT"
make analyze-bbq-partial ROOT="$AUDIT_SHARED_ROOT"
```

The audit outputs are `results/manifests/e6_receipt_audit.json` and
`results/manifests/bbq_partial_h1_h2.json`. Their scope and interpretation are
recorded in `docs/interim_privacy_evidence.md` and
`docs/interim_results_review.md`. A concise claim-to-evidence summary and
remaining review steps are in `docs/review_handoff.md`.
Run `make audit-evidence ROOT=/mounted/project` to check the fairness aggregate
against all 12 full-run source manifests and list measured versus pending claims.
For the offline presentation, `make demo-assets ROOT=/mounted/project` checks that
provenance before exporting aggregate-only data, and `make demo ROOT=/mounted/project`
opens the cached dashboard. See `docs/demo_cached.md`.

## Colab environment gate

Open `notebooks/e0_t04_colab_smoke_test.ipynb` in a fresh Colab Pro GPU runtime. It
installs the exact `colab` dependency group from `uv.lock`, records the GPU/CUDA/OS
environment, and tests M1 and M2 sequentially. Both generated manifests must pass
before confirmatory runs begin. See `docs/system_boundary.md` and
`docs/experiment_registry.md` for the frozen scope and run IDs.

## Repository policy

Synthetic/raw data, model files, adapters, private generations, credentials, and local
Drive paths must not be committed. Small manifests, aggregate results, reviewed
redacted examples, code, and report assets may be committed after review.
