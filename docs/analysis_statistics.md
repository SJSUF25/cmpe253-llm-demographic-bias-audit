# Statistical primitives for E4-T02

`src/analysis/statistics.py` implements the methods needed by the frozen H1–H5
plan. All default resampling uses seed 253 and 10,000 draws/replicates. These
functions consume paired **cluster-level** inputs; they do not infer independent
units from raw prompt rows.

## Pairing and clusters

For CAB, compute each scenario-family statistic first, then pass one value per
`scenario_id` to cluster resampling. For BBQ, average each item's three answer
orders first, group items under `template_id`, and pass every item value within
that template together. A bootstrap draw samples whole clusters, preserving
their internal observations. A sign-flip draw changes the sign of every paired
difference in a cluster together. Sorted cluster IDs make results invariant to
input dictionary order. The sign-flip test uses the standard Monte Carlo
`(extreme + 1) / (draws + 1)` p-value, including ties as extreme.

The two-sided cluster sign-flip function supplies the paired permutation
comparison against zero used for BBQ. For nonlinear outcomes such as the
**absolute** official bias score, `paired_cluster_permutation_pvalue` swaps
each whole baseline/mitigation cluster and reruns a pure caller-supplied statistic
on every draw. The E4-T03 caller must use that function to recompute the official
score; treating independently computed row scores as permutation units is invalid.

## Intervals and decisions

Two-sided confidence intervals use the nearest-rank 2.5th and 97.5th percentiles
of the 10,000 sorted bootstrap estimates. One-sided non-inferiority uses the
nearest-rank fifth percentile. A bound passes only when it is **strictly
greater** than the frozen margin; equality fails. The observed effect should
be reported separately from the interval. `cluster_mean` pools pre-aggregated
item values after cluster grouping, so larger BBQ templates contribute their
actual item count.

The exact binomial function returns the upper tail for H1's 5% material-treatment
null. Exact two-sided McNemar doubles the smaller Binomial(0.5) discordance tail,
capped at 1; the one-sided function tests improvement. Both return 1 with zero
discordant pairs. The matched odds ratio is `first_only / second_only` and is
reported as undefined (`null`) when its denominator is zero; no silent
pseudocount is added. Paired standardized mean difference uses the sample SD of
paired differences and is undefined at zero variance.

`missing_mean_bounds` assigns every missing outcome first the prespecified
minimum and then the maximum. This is a conservative bound for a **mean** of
bounded outcomes, not a substitute for rerunning nonlinear gap or ratio analyses.
The E4-T03/E4-T04 runners must show missing counts and apply their separate
complete-case and conservative sensitivity policies.

The unit tests include hand-calculated exact tails, risk differences, zero
discordances, zero variance, ties, all-zero sign flips, whole-template bootstrap
behavior, invalid/missing values, and 10,000-draw repeatability.
