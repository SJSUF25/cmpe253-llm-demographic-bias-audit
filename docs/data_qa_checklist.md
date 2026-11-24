# E1-T06 Data Review Checklist

Run `make validate-data ROOT=/mounted/shared/path` after the pinned raw downloads
and all four dataset builders have produced their private artifacts. The command
checks immutable source hashes and manifests, recomputes derived CAB/BBQ/PII/canary
artifacts, and writes these private files:

- `manifests/data_validation_manifest.json`
- `private_outputs/data_qa/review_packet.json`

The review packet contains 25 gender and 25 race CAB **development** scenario
families, two development PII records per task, two ordinary training records per
task, one member canary from each repetition stratum, two calibration non-members,
and one BBQ item from each category/context/polarity cell. It does not include CAB
test prompts, PII test records, or final non-member evaluation records. Keep the
packet in restricted shared storage; do not commit or share its contents publicly.

## Human review

For each of the 50 CAB families, verify that identity substitution changes only
the intended demographic marker, all variants keep the same scenario and split,
explicit/implicit variants link to the same source family, and the prompt remains
coherent. Record exceptions by scenario ID without copying harmful text into Git.

For the selected BBQ items, check that the context, question, three answer options,
correct label, target location, and target group agree with the pinned source.
For PII, check that every fictional value appears once and its redacted reference
uses the correct placeholder. For ordinary records, verify each exact answer by
hand. For canaries, verify the fixed prompt/target format and the assigned 1/5/10
repetition stratum; verify that calibration records have no training presentation.

Record the operator name, date, manifest SHA-256, reviewed item counts, exceptions,
and disposition in a private signed review receipt. Under PRD v2.5 Vineet performs
this check and marks it as an operator review. A Codex check or repeat by Vineet
must not be described as independent human QA.

## Remaining automated gate

With both pinned model tokenizers available, compute input-token lengths **after**
applying each model's official chat template and the exact condition-specific system
instruction. Reject any CAB or BBQ input above 4,096 tokens. Check the controlled
adapter prompt-plus-target sequences against the 512-token training limit. Record
the maximum and all failures in the data-gate receipt. Until this check and human
review pass, the manifest's overall status remains pending.
