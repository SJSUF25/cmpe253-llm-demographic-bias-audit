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

Use `make help` to list the complete project workflow. Experiment commands that have
not yet been implemented exit with a clear status instead of silently succeeding.

## Governance setup

Pinned public upstream metadata, cards, licenses, and configuration snapshots are
refreshed and verified with:

```bash
make sync-upstream
```

After the private shared Drive folder has been created and mounted, initialize it from
Vineet's account and verify it independently from Abhinandan's account:

```bash
make storage-init ROOT="/mounted/shared/path" AUTHOR="Vineet Malewar"
make storage-verify ROOT="/mounted/shared/path" REVIEWER="Abhinandan"
```

The mount path belongs in the ignored local environment, never in repository files.
See `docs/data_handling.md` for the storage layout and handling rules.

After the governance gate is approved, download or re-verify the pinned CAB and BBQ
sources without overwriting existing raw bytes:

```bash
make download-data ROOT="$AUDIT_SHARED_ROOT"
make download-data ROOT="$AUDIT_SHARED_ROOT" VERIFY_REMOTE=1
```

The command checks pinned revisions, byte sizes, SHA-256 values, schemas, row counts,
CAB race/gender inventory, and all three required BBQ categories before writing an
immutable raw-data manifest.

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
