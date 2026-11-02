# Configuration Files

Versioned experiment, model, scorer, annotation, and analysis configurations belong
here. Final run configurations must use immutable upstream revisions.

- `upstream_registry.json` is the reviewed immutable input to `make sync-upstream`.
- `model_manifest.yaml` records model, tokenizer, and scorer revisions and snapshots.
- `data_manifest.yaml` records CAB and BBQ revisions and governance snapshots.
- `storage.example.yaml` defines the shared private-storage contract.
- `storage.local.yaml` is reserved for ignored machine-specific mount configuration.
- `raw_data_sources.json` pins the exact CAB/BBQ source files, byte sizes, hashes,
  schemas, row counts, and required category inventory for `make download-data`.
