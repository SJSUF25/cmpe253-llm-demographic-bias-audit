# CAB/BBQ descriptive analysis

After all 12 full CAB/BBQ cells are complete and the renewed data gate passes,
run `make score-fairness-descriptive ROOT="$AUDIT_SHARED_ROOT"`. The command
verifies all registered output hashes and keys before writing
`aggregate_results/fairness_descriptive.json` to Drive. It never publishes raw
generations.

Before attributing any BBQ item, run
`make freeze-explain-sample ROOT="$AUDIT_SHARED_ROOT"`. It deterministically
selects five items from each context/polarity stratum of each of the three
categories (60 total), with hashes of the source build manifest. The same
sample applies to both models and conditions.

For each model, condition, BBQ category, and ambiguity context, it reports
official item-level bias and accuracy. The answer-order likelihoods are averaged
within each source item before this step. It also reports paired mitigation
changes in accuracy and absolute bias. The `Race_x_gender` category remains a
separate intersectional cell. For CAB, the command reports matched refusal gaps
for female–male and each race–White contrast under explicit and implicit identity
signals, plus absolute-gap changes after the fairness instruction.
It also pairs explicit and implicit variants of the same scenario family and
reports the change in the refusal gap with a family-cluster interval. This is an
identity-signal shift diagnostic on the common families; implicit families
excluded by the frozen quality gate are not silently replaced.

Every interval resamples entire `template_id` clusters for BBQ or entire
`scenario_id` families for CAB, using 10,000 seeded draws. Repeated option orders,
demographic variants, and mitigation conditions are never treated as independent
observations. The source manifest hashes are saved with the aggregate result.
The pooled 1,500-item BBQ accuracy change also receives the preregistered
one-sided 95% template-cluster lower bound against the `-0.03` margin. This is
only the accuracy component of H3; the joint mitigation claim remains pending
the human helpfulness component.

This is a **descriptive** artifact. The CAB refusal rule is an uncalibrated
anchored-phrase detector. It can miss indirect refusals and does not substitute
for blinded human differential-treatment or helpfulness ratings. No H1/H2/H3
decision, p-value, or multiplicity-corrected claim is made by this command.
