# Evidence review handoff

The implementation and measured outputs support a build-and-measure project on
two open-weight assistants. The project succeeds as a reproducible assurance
pipeline; the fixed fairness instruction and PII controls do **not** pass their
prespecified joint utility gates. Keep both findings visible in the demo.

## Results ready to present

| Claim | Evidence | Bound |
|---|---|---|
| Two-model paired audit on CAB and BBQ | `fairness_descriptive.json`; 8 CAB and 4 BBQ inference manifests | English, single-turn, tested identities only |
| Fixed fairness instruction changes BBQ performance | M1 accuracy −5.47 points; M2 −6.47; one-sided lower 95% bounds −6.61 and −9.32 points | Both fail the frozen −3-point accuracy margin; no joint mitigation success |
| Known synthetic PII exposure falls under redaction | P0: 200/200 M1 and 199/200 M2; P1/P2: 0/200 each | All four joint H5 cells fail the frozen content-similarity margin |
| Controlled Phi adapter creates measurable memorization risk | Membership AUC 1.000 after adapter versus 0.469–0.479 at base; exact extraction 2%, 99%, 100% after 1/5/10 presentations | Fixed synthetic canaries and held-out non-members; no claim about real people |
| Adapter improves measured ordinary-task accuracy | 24.25% base to 92.25% adapter on 400 held-out synthetic records | Four simple task formats, not general assistant usefulness |
| Local explanations and robustness probes run | 240 ablation rows and 32 automated-refusal robustness cells | Exact identity marker absent from 148/240 ablation rows; robustness lacks calibrated harmful-compliance labels |

The privacy evidence source hashes and recomputed totals are in
`results/manifests/e6_receipt_audit.json`. The 12 computed BBQ H1/H2 statistics
are in `results/manifests/bbq_partial_h1_h2.json`; their p values are unadjusted.
No H1/H2 family conclusion is available until the paired CAB ratings and the
full 18-test corrections are complete. One unadjusted BBQ comparison has
`p = 0.0084` while its separate bootstrap interval includes zero; do not use it
as the headline finding.

## Reproduce the local evidence checks

With the completed Drive folder mounted as `$AUDIT_SHARED_ROOT`:

```bash
make audit-privacy-receipts ROOT="$AUDIT_SHARED_ROOT"
make analyze-bbq-partial ROOT="$AUDIT_SHARED_ROOT"
make test
make lint
```

The first two commands read measured outputs and write local hash inventories;
they do not call the model or modify the measured Drive artifacts. The cached
dashboard is started with `make demo ROOT="$PWD"` after placing the matching
`demo_assets/bundle.json` and `manifest.json` locally. The fallback loader
verifies the bundle SHA-256 before displaying it.

## Items needing the team's judgment

- Complete and sign off the blinded CAB differential-treatment and helpfulness
  ratings using the fixed codebooks. Automated scores and the assistant's review
  cannot be represented as independent human judgments.
- Inspect a few representative outputs privately before screen sharing, then
  choose redacted examples. Keep raw prompts, PII, canaries, and row-level
  outputs out of public demo assets.
- Reconcile the final H1/H2 family corrections and complete the explanation
  faithfulness and robustness labels if those claims will appear in the report.
- Time two five-minute demo rehearsals, including a forced switch from the
  optional Colab live pair to the checked cached dashboard.

The cached dashboard currently opens all eight views. Its SHA-verified fallback
loaded all nine aggregate sections in about 0.003 seconds locally. The live
generation path was already exercised in Colab; the timed forced-failure drill
and final example signoff remain to be recorded.
