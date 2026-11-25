# E6-T02 Private PII Evaluation Runner

The registered `PII-M1/M2-P0/P1/P2` cells each contain 200 synthetic records in
the sealed test split. All three conditions use the same pinned baseline system
instruction and greedy decoding. P1 and P2 use the same input redaction; P2 adds
output filtering. Running all six cells yields 1,200 model generations. A separate
private P2 filtered file is derived from each P2 run.

`make run-pii` verifies the synthetic corpus checksum and 200-record split. Test
runs check the data-gate receipt and clean Git worktree **before reading the test
partition or loading a model**. The shared resumable runner then verifies every
partition and key on restart. Input audit manifests bind the source-file hash,
control-config hash, and inference manifest hash. P2 filter manifests bind the raw
inference file, filtered file, and per-row output hashes. Raw PII and generations
remain in `private_outputs` on the mounted Drive.

After all six test cells are complete, `make score-pii` verifies their hashes and
writes only aggregate counts to `aggregate_results/privacy_exposure_summary.json`:
40 records per task, exact and normalized record-level exposure, and exposure by
PII type. The input-redaction precision/recall/F1 table measures the five *known
synthetic fields* against their exact reference placeholders. It does not estimate
accuracy for unknown real-world PII. P2 residual exposure is measured on its
filtered output. Utility comparison and statistical H5 decisions remain E6-T05.

Development runs can be used to check integration before the sealed test gate.
Once the data gate is approved, run one model at a time in the locked Colab GPU
environment:

```bash
make run-pii ROOT="$AUDIT_SHARED_ROOT" RUN_ID=PII-M1-P0 SPLIT=test
make run-pii ROOT="$AUDIT_SHARED_ROOT" RUN_ID=PII-M1-P1 SPLIT=test
make run-pii ROOT="$AUDIT_SHARED_ROOT" RUN_ID=PII-M1-P2 SPLIT=test
make run-pii ROOT="$AUDIT_SHARED_ROOT" RUN_ID=PII-M2-P0 SPLIT=test
make run-pii ROOT="$AUDIT_SHARED_ROOT" RUN_ID=PII-M2-P1 SPLIT=test
make run-pii ROOT="$AUDIT_SHARED_ROOT" RUN_ID=PII-M2-P2 SPLIT=test
make score-pii ROOT="$AUDIT_SHARED_ROOT"
```

Do not run the test commands while the CAB quality manifest and truthful
data-gate approval receipt are pending. The local unit tests use fictional
fixtures and fake model outputs; they do not constitute a GPU run.
