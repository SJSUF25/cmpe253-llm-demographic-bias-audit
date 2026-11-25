# E1-T06 AI-assisted data review — 2026-11-25

This is a Codex review of the private, pinned data artifacts in the shared Drive. It is **not** the operator-signed human review required by `docs/data_qa_checklist.md`.

## Artifacts and scope

- Reviewed packet: `private_outputs/data_qa/review_packet.json`, SHA-256 `2dde3d874e16407427f35c2663ba0e234bf108ce20ff1b16b6daae1fd1da7fed`.
- Packet inventory: 50 CAB development families, 12 BBQ items, 10 synthetic PII development records, 8 ordinary training records, 3 member canaries, and 2 calibration nonmembers (85 records/families).
- `data_validation_manifest.json` reports structural checks passed; `token_length_manifest.json` reports passed with zero overlength failures. These gates do not establish semantic prompt quality.
- The existing `manual_review_progress.json` records 85 passes within approximately 75 seconds. It has not been changed. Its entries conflict with the six CAB exceptions below, so it cannot support an exception-free receipt.

## Checks performed

- All 50 CAB families have the required number of group variants, consistent scenario/split linkage, and identical text after normalizing the intended identity marker within each mode. Representative explicit and implicit prompt text was also read for each sampled family.
- The 12 BBQ records have three distinct options and internally consistent labels, target locations, target groups, context conditions, and questions.
- All 10 PII records have one occurrence of each fictional value and one corresponding replacement token in the redacted reference.
- All 8 ordinary task answers agree with their task instructions. The sampled canaries have the specified prompt/target format. Training presentations contain the sampled members 1, 5, and 10 times and the sampled calibration nonmembers zero times.

## Confirmed CAB exceptions

Six sampled **development** families contain a drafted answer inside the implicit prompt. The same contamination is present in the pinned raw CAB source; it was not introduced by the local builder. The explicit counterpart asks for an answer, while the implicit counterpart includes some or all of that answer. This makes explicit/implicit comparisons invalid for these families and can affect generation, scoring, and prompt-length results.

| Family index in packet | Scenario ID | Attribute | Issue |
| --- | --- | --- | --- |
| 22 | `8db2c4861d6abdb07b91` | Gender | Guidance and example opening included |
| 24 | `1e88e366bbeccbb628ea` | Gender | Script, note, and steps included |
| 31 | `9df5a0510db2b84aa7da` | Race | Example email and bullet list included, with extra invented accomplishments |
| 37 | `cf9725b5930197bc061c` | Race | Example recommendation email included |
| 44 | `76a56f28d34e51162319` | Race | Requested emergency script/facts/questions already supplied; no equivalent request remains |
| 46 | `f7c08f24d43c24e2f81e` | Race | Subject, outreach, and tips included |

An automatic scan of the full processed CAB manifest found **23 families** whose implicit text has an unusual length ratio or paragraph breaks: these 6 development families and **17 test families**. The 17 test texts were not opened or adjudicated, so these are flags, not confirmed exceptions. The scan was based on the development exceptions and performed before any model outcome analysis.

## Disposition

**Data gate blocked pending the revised quality manifest and operator approval.** Do not sign an exception-free review receipt or start confirmatory inference yet. PRD section 6.1 now freezes a source-preserving exclusion rule: all implicit variants of the 23 flagged families are omitted from inference; explicit primary prompts remain intact. The 17 test flags are conservatively excluded without opening their text. Keep the current raw source and prior manifests immutable.

`make cab-quality-check ROOT=/mounted/shared/path` creates or verifies the private exclusion manifest. The inference runner requires and rechecks it. The existing structural and tokenizer manifests still describe the unmodified 1,604-row prompt inventory; the separate quality manifest describes the 736 eligible implicit prompts. The operator approval receipt must bind all three manifests. Report the reduced implicit sample and its sensitivity limits.
