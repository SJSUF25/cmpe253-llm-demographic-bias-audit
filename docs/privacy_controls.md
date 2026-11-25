# E6-T01 Synthetic PII Controls

The controls operate only on the 400 entirely fictional records defined in
`configs/privacy_data.json`. `configs/privacy_controls.json` freezes entity names,
placeholder tokens, entity priority, regex patterns, the spaCy model, and the P2
normalization rule. The Colab lock pins `presidio-analyzer==2.2.366` and the
`en_core_web_sm` 3.8.0 wheel by URL and SHA-256.

| Condition | Input sent to the model | Model output returned |
| --- | --- | --- |
| P0 | Raw synthetic record | Raw output |
| P1 | Presidio plus known-value and format-pattern redaction | Raw output |
| P2 | Same redacted input as P1 | Presidio plus known-value output filtering |

Call `prepare_input(record, condition, analyzer=analyzer)` before inference and
`filter_output(output, record, condition, analyzer=analyzer)` after inference.
Create one analyzer per process with `make_analyzer()` and reuse it for all records.
Both functions return text and an audit object containing condition, typed
replacement counts, and detection-source counts. The audit does not contain raw
PII values. Store the full record and raw model outputs only in private Drive.

P1's field-level synthetic truth map is complete by construction. Generic Presidio
detections are admitted only when they lie entirely inside a matching known field;
this prevents false-positive redaction of ticket IDs and task facts. Exact known
values take precedence on equal-length spans; other overlaps use longest span,
then the configured entity priority. P2 runs generic Presidio detections on model
output and also catches any known value after NFKC, casefolding, and removal of
formatting separators while preserving source offsets. The P2 detector can still
miss paraphrased or novel sensitive data and may over-redact novel text; those error
rates must be measured on the held-out synthetic test set, not assumed away.

Local verification:

```bash
make test
make privacy-controls-smoke
```

The smoke command exercises the real pinned Presidio/spaCy engine on one
development record. The E6-T01 development check also compared all 200
development P1 outputs with the generator's exact redacted references and found
zero mismatches. No privacy test record or model output was used for this tuning.

Sources for the pinned engine and model:

- [Presidio analyzer installation](https://microsoft.github.io/presidio/installation/)
- [Presidio custom recognizers](https://microsoft.github.io/presidio/analyzer/adding_recognizers/)
- [spaCy English model](https://spacy.io/models/en/)
