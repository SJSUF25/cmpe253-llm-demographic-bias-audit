# Interim measured privacy evidence

This note records the E6 results supplied from the completed Colab run on
2026-11-27. It is a bounded interpretation, not the final privacy handoff.
Source artifacts are in the shared project Drive. The H5 artifact binds all six
`PII-M1/M2-P0/P1/P2` source-output hashes and the exposure-summary hash. The
canary and adapter-utility figures below were supplied as aggregate output;
their Drive files and receipt hashes still need an independent audit.

| Model | Control | P0 exposed | Control exposed | Exposure reduction | Utility delta | One-sided lower 95% | Joint H5 |
|---|---|---:|---:|---:|---:|---:|---|
| M1 | P1 input redaction | 200/200 | 0/200 | 100.0 pp | -0.0867 | -0.0933 | Fail |
| M1 | P2 redaction plus output filter | 200/200 | 0/200 | 100.0 pp | -0.0874 | -0.0939 | Fail |
| M2 | P1 input redaction | 199/200 | 0/200 | 99.5 pp | -0.0874 | -0.1010 | Fail |
| M2 | P2 redaction plus output filter | 199/200 | 0/200 | 99.5 pp | -0.0890 | -0.1029 | Fail |

The exposure tests pass their four-test Benjamini–Hochberg correction. All four
joint H5 cells fail because the frozen utility rule requires a one-sided lower
95% bound greater than -0.05. Utility here is the cosine similarity between
PII-normalized control and P0 responses, minus 1.0; it is a proxy for preserving
response content. The difference is measured on the same 200 fictional test
records per model. The 300 blinded helpfulness ratings that check this proxy are
still pending. The original H5 artifact marks adapter utility as pending because
it was calculated before the separate adapter experiment finished.

## Controlled canary and ordinary-task results

The two attack modes score 300 training members, 500 calibration non-members,
and 500 held-out evaluation non-members. The threshold is fixed from the
calibration partition with five false positives; the evaluation FPR is measured
separately. Both attacks use the same synthetic canary format and pinned Phi
base, with the trained QLoRA adapter attached only in the second mode.

| Mode | Score | ROC-AUC (95% bootstrap CI) | TPR at calibrated threshold | Held-out FPR |
|---|---|---:|---:|---:|
| Base | Min-K 20% | 0.479 (0.438–0.520) | 0.7% | 1.8% |
| Adapter | Min-K 20% | 1.000 (1.000–1.000) | 100% | 0.8% |
| Base | Negative loss | 0.469 (0.428–0.511) | 1.0% | 1.2% |
| Adapter | Negative loss | 1.000 (1.000–1.000) | 100% | 0.4% |

Exact prompted extraction was 0/100 in each base repetition stratum. With the
adapter it was 2/100 for one presentation, 99/100 for five, and 100/100 for ten.
The perfect attack intervals are conditional bootstrap intervals on this fixed
synthetic sample; they do not imply perfect accuracy on other prompts or users.
Near-zero exact extraction after one presentation coexists with a highly
discriminative loss-based membership score. These measure different failure
modes.

On 400 held-out ordinary validation records, exact accuracy rose from 24.25%
for the base to 92.25% for the adapter, and mean target loss fell from 2.394 to
0.073. The adapter improves this measured ordinary-task utility, but this
task-specific result does not establish general assistant quality. The largest
accuracy gains are for addition (0% to 98%) and JSON-field extraction (0% to
100%); copy-code was already high (94% to 100%). The same training that improves
these tasks also creates severe controlled membership and extraction risk.

The result supports a narrow claim: these controls prevented the measured
known-value exposure on this synthetic corpus, with a measured content-similarity
cost beyond the preregistered margin. It does not establish performance on real
personal data, an acceptable utility trade-off, or reduced membership risk.
The separate adapter experiment shows a large controlled memorization risk
despite improved ordinary-task accuracy. No threshold or control should be
changed after viewing these results and then treated as confirmatory.
