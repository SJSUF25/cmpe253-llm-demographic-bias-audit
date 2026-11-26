# E2-T04 Pilot

The v2 pilot is a gated feasibility run over 50 CAB development families and a frozen,
stratified sample of 100 BBQ items. It runs both models under baseline and fairness
conditions. Its outputs are diagnostic and never enter confirmatory tests.

Run the selection plan before loading a model:

```bash
make pilot ROOT="$AUDIT_SHARED_ROOT" ACTION=plan
```

This writes immutable private selection artifacts and
`manifests/pilot/selection.json`. This is the same immutable selection used by the
first pilot. CAB selects 25 gender and 25 race development
families and includes every explicit group variant in those families. BBQ uses
Hamilton allocation across category, context condition, and question polarity, then
orders rows by a purpose-specific SHA-256. The selected 100 items retain all three
cyclic option orders.

The first pilot was retained as diagnostic evidence: its CAB responses often hit
the 256-token cap. The revised primary protocol gives both conditions the same
150-word response instruction and a 384-token cap. A 10-prompt-per-model
development diagnostic found 20/20 EOS finishes for M1 at 256 tokens and 20/20
for M2 at 384 tokens; the full pilot must test whether this generalizes.

Model execution requires a clean Git checkout and a current
`manifests/data_gate_approval_v2.json` bound to the revised governance lock.
The Drive PRD must match that lock. Run one model at a time on the same L4 GPU class:

```bash
make pilot ROOT="$AUDIT_SHARED_ROOT" ACTION=run-model MODEL=M1
make pilot ROOT="$AUDIT_SHARED_ROOT" ACTION=run-model MODEL=M2
```

Each command loads one pinned FP16 model and runs four cells: CAB baseline, BBQ
baseline, CAB mitigation, and BBQ mitigation. CAB uses the general resumable runner.
BBQ checkpoints every ten completed items, with nine full-answer likelihood calls per
item. Before those cells, a small model preflight verifies exact greedy output under
reversed prompt execution order and repeated likelihood agreement within `1e-4`.
Repeating a completed command verifies and reuses its immutable receipt. New
`PILOT2` run IDs and `manifests/pilot_v2/` and `private_outputs/pilot_v2/`
paths preserve the first pilot's evidence.

After both model receipts exist, reconcile the eight cells:

```bash
make pilot ROOT="$AUDIT_SHARED_ROOT" ACTION=report
```

The report at `manifests/pilot_v2/pilot_report.json` requires zero errors and exact row counts. It records runtime per row,
peak allocated GPU memory, CAB token maxima and length finishes, BBQ option-order
disagreement, and projected full-run time. It also creates a private 20-row CAB output
packet. The report status is `ready_for_operator_review`: Vineet must read that packet,
record any anomalies, confirm the projected workload is feasible, and make the pilot
gate decision before sealed CAB or full BBQ execution. Use
`notebooks/e2_t04_pilot_v2_colab.ipynb` for the complete Colab workflow.

The pilot does not replace the data review. Planning is safe while data gate
checks run; model execution remains blocked until a valid receipt matches the
current governance lock. The accelerated AI review route must be disclosed as
such in the final report.

The completed v2 feasibility review, its limitations, and the delegated
AI-assisted decision are recorded in `docs/pilot_v2_review.md`. Subsequent
confirmatory runs use a renewed v3 data-gate receipt after the full-run
orchestration was added; the historical v2 pilot receipts remain unchanged.
