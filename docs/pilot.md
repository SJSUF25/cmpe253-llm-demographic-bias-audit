# E2-T04 Pilot

The pilot is a gated feasibility run over 50 CAB development families and a frozen,
stratified sample of 100 BBQ items. It runs both models under baseline and fairness
conditions. Its outputs are diagnostic and never enter confirmatory tests.

Run the selection plan before loading a model:

```bash
make pilot ROOT="$AUDIT_SHARED_ROOT" ACTION=plan
```

This writes immutable private selection artifacts and
`manifests/pilot/selection.json`. CAB selects 25 gender and 25 race development
families and includes every explicit group variant in those families. BBQ uses
Hamilton allocation across category, context condition, and question polarity, then
orders rows by a purpose-specific SHA-256. The selected 100 items retain all three
cyclic option orders.

Model execution requires a clean Git checkout and a valid current
`manifests/data_gate_approval.json`. Run one model at a time on the same L4 GPU class:

```bash
make pilot ROOT="$AUDIT_SHARED_ROOT" ACTION=run-model MODEL=M1
make pilot ROOT="$AUDIT_SHARED_ROOT" ACTION=run-model MODEL=M2
```

Each command loads one pinned FP16 model and runs four cells: CAB baseline, BBQ
baseline, CAB mitigation, and BBQ mitigation. CAB uses the general resumable runner.
BBQ checkpoints every ten completed items, with nine full-answer likelihood calls per
item. Before those cells, a small model preflight verifies exact greedy output under
reversed prompt execution order and repeated likelihood agreement within `1e-4`.
Repeating a completed command verifies and reuses its immutable receipt.

After both model receipts exist, reconcile the eight cells:

```bash
make pilot ROOT="$AUDIT_SHARED_ROOT" ACTION=report
```

The report requires zero errors and exact row counts. It records runtime per row,
peak allocated GPU memory, CAB token maxima and length finishes, BBQ option-order
disagreement, and projected full-run time. It also creates a private 20-row CAB output
packet. The report status is `ready_for_operator_review`: Vineet must read that packet,
record any anomalies, confirm the projected workload is feasible, and make the pilot
gate decision before sealed CAB or full BBQ execution.

The pilot does not replace the data review. Planning is safe while data gate
checks run; model execution remains blocked until a valid receipt matches the
current governance lock. The accelerated AI review route must be disclosed as
such in the final report.
