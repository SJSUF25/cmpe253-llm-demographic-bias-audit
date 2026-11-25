# E6-T04 Controlled Canary Attacks

Run the two modes only after the pinned Colab adapter training has produced a
verified epoch-3 checkpoint and the truthful private data-gate receipt exists:

```bash
make attack-canary ROOT="$AUDIT_SHARED_ROOT" MODE=base
make attack-canary ROOT="$AUDIT_SHARED_ROOT" MODE=adapter
```

Both modes use the pinned Phi model in the same 4-bit NF4 configuration. The
adapter mode attaches the verified final epoch adapter; the base mode does not.
The script refuses a dirty Git worktree, checks the data gate before reading the
sealed partitions, and reconciles all 300 members, 500 calibration non-members,
and 500 final evaluation non-members against the frozen generator and manifest.

For each canary, it measures greedy prompted exact extraction, mean target-token
log probability (negative sequence loss), and the mean of the lowest 20% target
token log probabilities (Min-K%, ceiling to at least one token). The official
chat template masks the prompt, includes the target completion, and refuses
truncation beyond 512 tokens. Higher likelihood scores indicate greater member
confidence. The generated output is compared to the exact target, then discarded.

For each score, the sixth-highest calibration value is the fixed threshold;
**strictly greater** scores predict membership, so calibration produces at most
five false positives. ROC-AUC and final FPR use the 300 members and 500 held-out
evaluation non-members only. The report also includes TPR at that fixed threshold,
extraction by repetition stratum, and 10,000 stratified bootstrap 95% intervals.
The threshold is held fixed during bootstrap resampling. Results are descriptive
for these synthetic records and this controlled adapter.

Private row-level scores and manifests are written under
`private_outputs/canary_attack/` and `manifests/canary_attack/`. Aggregate JSON
appears under `aggregate_results/canary_attack_base.json` and
`aggregate_results/canary_attack_adapter.json`. No target canary values or
generated text are written to aggregate files or Git. The local unit fixtures
verify partition hashes, threshold ties, tie-aware AUC, and held-out separation;
they do not substitute for the two Colab GPU runs.
The GPU run saves hash-checked 100-record parts; restarting the same clean commit
reuses completed parts after verifying their input IDs and hashes.
