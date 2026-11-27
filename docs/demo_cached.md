# Cached offline demo

The demo bundle contains only named aggregate metrics. Generate it after the
fairness summary and its 12 source manifests exist:

```bash
make demo-assets ROOT=/mounted/project
make demo ROOT=/mounted/project
```

Install the pinned `demo` dependency group once while online. The running app
reads only `ROOT/demo_assets/bundle.json` and does not load models or call the
network. The `demo-assets` target first verifies the fairness source hashes. The
exporter copies allowlisted fields from the fairness, secondary CAB, explanation,
robustness, and privacy aggregate files and records each source SHA-256. It does
not copy raw prompts, model outputs, canaries, PII values, or private paths.
`demo_assets/manifest.json` records the bundle hash and pending sections.

The eight tabs track the PRD demo views. Dataset examples and side-by-side raw
outputs stay unavailable until individual examples have been reviewed and
redacted. Missing experimental sections show explicit pending messages. The
fairness view keeps confidence intervals and denominators visible, and the
mitigation view distinguishes BBQ accuracy from the joint utility decision.

The bundle is a presentation artifact. The signed aggregate files in Drive
remain the evidence source. Rebuild the bundle when a new aggregate section is
completed; do not hand-edit its JSON.
