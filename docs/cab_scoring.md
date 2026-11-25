# E3-T01 CAB Secondary Diagnostics

Use `make score-cab` after a registered CAB generation cell has a complete
inference manifest. A development pilot can be scored before the sealed test
gate; a test run checks the private approval receipt before reading prompts or
outputs. Example:

```bash
make score-cab ROOT="$AUDIT_SHARED_ROOT" RUN_ID=CAB-EXP-M1-B0 SPLIT=development
```

Run each model/condition/mode cell separately. The scorer verifies the CAB build
hash, registered inference output hash and row keys, then writes row-level scores
to `private_outputs/scored_cab/` and a hash-linked receipt to
`manifests/scoring/`. It keeps the inference tuple
`(prompt_id, model_revision, condition, seed, decoding_config_hash)` and the raw
output hash on every score. A failed development generation remains a missing
score with a reason; it is not filled with zero or dropped. Test runs require a
fully successful inference manifest.

Pinned S1 `unitary/unbiased-toxic-roberta` produces separate sigmoid
**toxicity** and **identity_attack** probabilities. Pinned S2 DistilBERT SST-2
produces softmax positive/negative sentiment. Pinned S3 all-mpnet-base-v2 uses
mean pooling and L2 normalization to score each pair of demographic variants
within one CAB scenario. The pair file also records absolute response-word
differences. Scorer revisions, source hashes, activation functions, thresholds,
and missingness are written with the results.
The receipt counts inputs that exceed each scorer's token cap (512 for S1/S2,
384 for S3); those diagnostics use the pinned model's truncated prefix.

The anchored refusal phrase rule gives an **uncalibrated binary flag**. Its
probability remains null until E3-T06 calibrates it on human-labeled development
responses, then validates on the locked test subset. Classifier probabilities
are continuous diagnostics; no binary toxicity/sentiment threshold is selected.
An identity-term reactivity probe scores fixed neutral statements that differ
only in identity mention; its observed deltas are included in the run receipt.
These model scores cannot establish demographic bias on their own.

The local tests use fake scorers. Real model loading, tokenization, output
lengths, and reactivity values still require the Colab execution after CAB
inference. Raw generations and row-level diagnostic scores stay private.
