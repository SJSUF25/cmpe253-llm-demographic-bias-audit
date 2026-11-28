# Interim results review

This review uses the aggregate artifacts currently saved in the shared project
Drive. It does not substitute for the preregistered H1/H2 tests, human ratings,
or final evidence signoff. The fixed methods and success margins remain those in
`LLM_Bias_Audit_PRD.md` Sections 3.3 and 9.3. Null and failed-control results
remain findings.

## Artifact checks

| Aggregate artifact | Reconciled sources | Scope |
|---|---:|---|
| `fairness_descriptive.json` | 8 CAB and 4 BBQ manifests | 36 BBQ cells, 48 CAB refusal cells, 16 identity-signal shift cells; 10,000 cluster bootstrap draws |
| `cab_secondary_summary.json` | 8 scoring manifests | 48 group cells, 56 pair cells; uncalibrated secondary metrics |
| `explain_summary.json` | 4 explanation manifests | 240 model-condition-item rows, 60 feature cells, 36 order-stability cells |
| `robustness_refusal.json` | 8 robustness receipts | 32 automated-refusal cells for whitespace and adversarial variants |
| `privacy_exposure_summary.json` / `privacy_h5.json` | 6 PII source-output hashes | 1,200 generations; 4 frozen H5 exposure/utility comparisons |
| `canary_attack_base.json` / `canary_attack_adapter.json` | Both source-output SHA-256 links and complete 1,300-row summaries independently recomputed in the E6 receipt audit | 2 modes × 1,300 synthetic records, with held-out attack metrics |
| `adapter_utility_comparison.json` | Both 400-row source hashes and task-level totals independently recomputed in the E6 receipt audit | 400 held-out ordinary task records |

These inventory and hash fields show that the aggregate builders consumed the
intended files. They do not independently certify model behavior or human label
quality. The full-run BBQ likelihood spot check remains an independent review item.
The E6 receipt audit verified 46 files and recomputed the 30 task-level synthetic
PII exposure cells. Its hash inventory is in
`results/manifests/e6_receipt_audit.json`.

The 12 available planned H1/H2 BBQ tests were computed with 10,000 seeded
template-cluster permutations. Their result is in
`results/manifests/bbq_partial_h1_h2.json`. These are unadjusted partial
statistics: the frozen H1 and H2 families each contain 18 tests, so their
Benjamini–Hochberg corrections and decisions await the CAB ratings. One H2
comparison, M1 `Race_x_gender`, has an unadjusted one-sided permutation
`p = 0.0084`; its separate cluster-bootstrap 95% interval includes zero.
That mixed uncertainty and the missing family correction preclude a success
claim for this comparison.

## Findings against the original expectations

| Area | Measured result | Interpretation and next check |
|---|---|---|
| BBQ baseline bias | All six ambiguous-context baseline category intervals include zero. Baseline scores range from -1.6 to +4.0 points across the two models and three categories. | No preregistered H1 claim can be made from these descriptive intervals. CAB human differential-treatment ratings and the planned formal tests remain open. This does not prove absence of bias. |
| Fairness instruction | Pooled BBQ accuracy changes are -5.47 percentage points for M1 and -6.47 for M2. One-sided lower 95% bounds are -6.61 and -9.32 points, below the frozen -3-point margin. | The BBQ component of H3 fails for both models, so the joint mitigation-success rule cannot pass. Some category bias point estimates improve, but every ambiguous-context absolute-bias-reduction interval includes zero. Keep null/mixed results visible. |
| CAB secondary scoring | Pairwise response similarity means range from 0.842 to 0.938; the largest absolute mean response-length difference is 20.2 words. | These are diagnostics, not a blinded differential-treatment judgment. The anchored refusal rule is not calibrated against human labels. |
| Explanation | Answer-order feature-rank Spearman means range from 0.695 to 0.970 across 36 cells. Exact identity-marker effects are missing for 148/240 rows, including all 80 race-by-gender rows. | Section ablation and cyclic-order stability are measured, but the identity-marker claim is unsupported for the missing rows. The exact-text matcher does not find the target-group label in those prompts. Token faithfulness and paraphrase stability remain open. |
| Robustness | Among 32 cells, 7 adversarial refusal-gap point changes are nonzero; none of their 95% intervals excludes zero. All whitespace gap changes are zero. | This is only an uncalibrated anchored-refusal diagnostic. It cannot establish adversarial safety or harmful-compliance rates. Human labels, paraphrase, position, and stochastic tests remain open. |
| Synthetic PII | P1/P2 reduce measured normalized known-value exposure to 0/200 per model. All four corrected exposure tests pass. All four joint H5 cells fail: utility lower bounds range from -0.0933 to -0.1029, below the -0.05 margin. | The controls block these known fictional values but do not meet the prespecified content-preservation rule. The redaction detector's 1.0 precision/recall is against exact known synthetic fields, not unfamiliar real PII. Human helpfulness remains open; the separate adapter utility result is reported below. See `docs/interim_privacy_evidence.md`. |
| Controlled memorization | All three training receipts exist for the pinned Phi adapter, each with 3,200 presentations and 200 optimizer steps. Mean microbatch loss fell from 2.104 to 0.380 to 0.108. Base loss and Min-K membership AUCs were 0.469 and 0.479; both adapter AUCs were 1.000 on 300 members and 500 held-out non-members. Adapter exact extraction was 2%, 99%, and 100% after one, five, and ten presentations; base extraction was 0% in all strata. | This is a strong controlled memorization finding. The bootstrap [1.0, 1.0] AUC intervals describe this fixed synthetic sample, not an assurance of perfect detection on new data. The threshold was calibrated on a separate 500 non-members; held-out adapter FPRs were 0.4% (loss) and 0.8% (Min-K). Source hashes and aggregate scores were checked in the E6 receipt audit. |
| Adapter ordinary-task utility | On 400 held-out synthetic ordinary records, exact accuracy increased from 24.25% to 92.25% and mean target loss fell from 2.394 to 0.073. | The adapter improves the four measured task formats while substantially increasing canary leakage. This does not establish general-purpose assistant utility. |

### Where the fairness-instruction accuracy loss occurs

The BBQ item table separates ambiguous from disambiguated contexts. Averaging
the three equally sized 250-item category cells gives the following descriptive
changes (F1 minus B0):

| Model | Ambiguous accuracy change | Disambiguated accuracy change |
|---|---:|---:|
| M1 | -2.67 percentage points | -8.27 percentage points |
| M2 | +3.87 percentage points | -16.80 percentage points |

The largest category cell is M2 `Gender_identity` disambiguated: -29.2 points,
with a template-cluster 95% interval from -35.3 to -23.0 points. This is a
substantive *exploratory* finding: the fixed fairness instruction is associated
with lower accuracy especially when the benchmark supplies disambiguating
evidence. These aggregate scores alone do not reveal whether the model shifted
toward the unknown option or another incorrect option; that requires an
item-level answer breakdown. The frozen pooled H3 accuracy gate
remains failed regardless of the diagnostic breakdown.

The H5 scorer also reports secondary similarity to structured task references.
For M1, P0/P1/P2 means are 0.678/0.653/0.647; for M2 they are
0.671/0.660/0.659. These secondary differences are smaller than the primary
P0-output similarity losses, so blinded helpfulness review may clarify how
much practical task utility changed. It cannot retroactively make the frozen
H5 utility gate pass.

## Decision for the current build

The pipeline is producing substantial, hash-linked measurements from the chosen
CAB and BBQ benchmarks and the synthetic privacy corpus. Several hoped-for
positive-control outcomes are **not** present: fairness-instruction utility and
privacy-control utility miss their frozen margins, and the current explanation
method lacks most exact identity-marker effects. Report those failures directly;
do not retune the frozen prompts, thresholds, or sample after seeing test results
and then present the new version as confirmatory.

The remaining evidence work is the blinded CAB and helpfulness ratings,
calibration of the refusal rule, remaining H1/H2 tests, and bounded presentation
examples. The cached demo fallback was checked locally against its SHA-256
manifest and loaded in 0.003 seconds; all nine aggregate sections were present.
This note remains an interim interpretation until the remaining evidence is
completed and signed off.
