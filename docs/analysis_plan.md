# Confirmatory analysis plan

`configs/confirmatory_analysis.json` enumerates all 48 tests in PRD Section 9.3.
It contains 18 H1 fairness tests, 18 H2 mitigation tests, four H3 utility tests,
four H5 privacy exposure tests, and four H5 privacy utility tests. Each row fixes
the dataset, model, cell, contrast, comparison, test, cluster, effect, alternative,
correction family, and margin. The top level fixes seed 253, 10,000 permutation
draws, 10,000 bootstrap resamples, 95% confidence, and BH q=0.05.

Run `make analysis-plan-check` to validate the complete inventory and print its
SHA-256. The validator rejects any changed or additional test. The plan and its
validator are included in `results/manifests/governance_lock.json`.

This is a **prepared plan**, not an approved preregistration receipt. Before
opening aggregate confirmatory results, Vineet must review the plan against PRD
Section 9.3, record its exact hash and review date in the private project records,
and complete the data and pilot gates. Changes after outcome inspection require
versioning and exploratory labeling; they cannot silently enter H1–H5.

The `H3` and `H5_utility` rows use one-sided lower bounds and margins of -0.25,
-0.03, and -0.05 as appropriate. Their `intersection_union` correction labels
describe the joint decision rule, not BH-adjusted p-values. The H1 and H2 rows
are separate BH families of 18; H5 exposure has its own BH family of four.
