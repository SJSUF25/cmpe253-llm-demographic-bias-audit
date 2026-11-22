# Configuration Files

Versioned experiment, model, scorer, annotation, and analysis configurations belong
here. Final run configurations must use immutable upstream revisions.

- `upstream_registry.json` is the reviewed immutable input to `make sync-upstream`.
- `model_manifest.yaml` records model, tokenizer, and scorer revisions and snapshots.
- `data_manifest.yaml` records CAB and BBQ revisions and governance snapshots.
- `storage.example.yaml` defines the shared private-storage contract.
- `storage.local.yaml` is reserved for ignored machine-specific mount configuration.
- `raw_data_sources.json` pins the exact CAB/BBQ source files, byte sizes, hashes,
  schemas, row counts, official BBQ supplemental metadata, and required category
  inventory for `make download-data`.
- `cab_build.json` freezes CAB expansion, identity ordering, split namespace, expected
  counts, normalization version, and private processed-output locations.
- `bbq_build.json` freezes the 500-per-category selection and private output paths for
  `make build-bbq ROOT=/mounted/shared/path`.
- `membership_data.json` freezes the controlled canary corpus, repetition strata,
  calibration/evaluation split, and private output paths for `make build-membership`.
- `inference.json` freezes model keys, system instructions, token caps, precision,
  attention, and greedy/sensitivity decoding for the model adapter.
