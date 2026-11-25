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
packet in the designated shared storage under the accepted access exception; do not
commit its contents publicly.

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
and disposition in a private signed review receipt. Under the 2026-11-25
delegation, Abhinand performs this check as the sole operator. A Codex check or
Vineet's earlier review must not be described as Abhinand's human QA.

The 2026-11-25 AI-assisted review found six development CAB implicit prompts with
embedded answer text; see `docs/data_qa_ai_review.md`. The old 85-pass notebook log
does not resolve these exceptions. The approved exclusion policy is in PRD section
6.1. Run `make cab-quality-check ROOT=/mounted/shared/path`, verify its 23-family
exclusion count and 736 eligible implicit prompts, and bind the resulting private
manifest hash in any data-gate approval receipt. An operator review may approve
the documented exceptions and exclusion, but must not say the original packet
was exception-free. The data-gate receipt remains pending until then.

`notebooks/e1_t06_data_gate_colab.ipynb` is the current Colab operator workflow.
It creates a fresh private review log and asks for a separate explicit `x` decision
on each of the six sampled development exclusions. The other 17 flagged test
families are excluded by rule without opening their text. Its final cell checks
the complete review, Drive PRD/governance-lock match, and all data hashes before
writing the operator and inference-gate receipts. Do not use the old scratch
notebook or its 85-pass log as approval evidence.
Abhinand runs the notebook on a free CPU runtime. Its fixed reviewer name is
`Abhinand`, and the fresh operator log is `manual_review_abhinand_operator.json`.
He alone can enter the final attestation. No independent second human review is
claimed.

## Pinned tokenizer-length gate

After structural validation, run `make token-length-check ROOT=/mounted/shared/path`
in the locked Colab environment. It loads only the two pinned tokenizers, applies
their official chat templates with both exact system instructions, and checks every
CAB prompt and BBQ option-order input against 4,096 tokens. It also checks all 3,200
Phi adapter prompt-plus-target presentations against 512 tokens without truncation.
The deterministic `manifests/token_length_manifest.json` records counts, maxima,
input hashes, tokenizer revisions, and every overlength ID. A failing limit blocks
the data gate. The structural manifest remains immutable and pending human review;
the signed approval receipt binds both manifest hashes after this check and the
manual QA above pass.
