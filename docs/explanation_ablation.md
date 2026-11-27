# BBQ explanation ablation

The deterministic `explain60` selection uses five source items from each
context/polarity stratum in each of the three BBQ categories. Selection uses
only frozen source IDs and SHA-256 sorting. The same 60 items are used for both
models and both fairness conditions.

`make run-explain ROOT=/mounted/project RUN_ID=EXPLAIN-M1-B0` runs one of the
four registered 60-item cells on NVIDIA L4. Before every ablation, the runner
rescores the full prompt's selected and runner-up answers under all three
cyclic option orders and requires the resulting mean likelihoods to reproduce
the completed full BBQ result within `1e-4`. A mismatch stops the run.

The original winner and runner-up remain fixed. For each input alteration, the
runner scores those same answer strings in all three orders. The reported
`margin_drop` is the original selected-minus-runner-up margin minus the altered
margin. Positive values mean the removed input supported the original winner.
The five alterations are exact target-group masking, context removal, question
removal, displayed option-text masking, and removal of common function words.
Absent exact markers are reported as missing, not zero effect. These are
behavioral input-sensitivity explanations; they do not reveal hidden reasoning.

Each completed item is saved separately with its source and environment hashes;
rerunning the same cell checks and reuses verified items. After all four cells,
`make report-explain ROOT=/mounted/project` checks all 240 item hashes and
reports each model × condition × category × feature mean with 95% template
cluster bootstrap intervals. It includes missing-feature counts. Each item
also retains the five effects under each of the three cyclic answer orders.
The report computes pairwise Spearman rank correlation and top-two overlap of
feature effects, with template-cluster intervals and missingness. A constant
ranking has undefined correlation and remains missing rather than being given
a perfect score. This tests answer-order stability only; token-level
faithfulness and paraphrase stability remain pending.
