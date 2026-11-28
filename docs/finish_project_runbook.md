# Project completion runbook

This is the remaining work after the full CAB/BBQ, E5, E6, and E7 Colab runs.
The estimate of 65% complete is a planning judgment, not a measured percentage
of code or tasks. The uncompleted 35 percentage points are roughly: fairness
ratings and final statistics (12), privacy helpfulness review and H5 handoff (5),
explanation and robustness validation (8), and evidence package, demo rehearsal,
report, presentation, and submission (10). These weights may change as work
finishes.

## 1. Publish the uncommitted reviewer change

The privacy reviewer is currently local only. Vineet can review the four changed
files, commit, and push when ready. Codex will not commit this change. In a Colab
runtime, pull that commit and run setup again only if it is a fresh runtime.

## 2. Finish the blinded CAB audit

Use a Colab terminal with Drive mounted at `/content/drive` and the repository at
`/content/cmpe253-llm-demographic-bias-audit`. The code and packet must be on
the same pinned commit. Run:

```bash
cd /content/cmpe253-llm-demographic-bias-audit
git pull --ff-only
```

The development packet requires four complete **development CAB inference**
runs. The earlier pilot and full `SPLIT=test` outputs do not satisfy this
prerequisite. On an L4 runtime, first see whether the four development manifests
are present:

```bash
ls /content/drive/MyDrive/CMPE253-Project/manifests/inference/CAB-EXP-*-DEV.json
```

If any are missing, run all four registered development cells. The runner
checkpoints progress, so rerunning a cell resumes verified work:

```bash
set -e
ROOT=/content/drive/MyDrive/CMPE253-Project
for RUN_ID in CAB-EXP-M1-B0 CAB-EXP-M1-F1 CAB-EXP-M2-B0 CAB-EXP-M2-F1; do
  make run-cab ROOT="$ROOT" RUN_ID="$RUN_ID" SPLIT=development
done
```

After the four development inference manifests exist, the four CAB secondary
diagnostic scores can run without human input. The pinned classifier and MPNet
scorers execute on CPU, even in an L4 runtime; they are useful for later
development-only refusal calibration but do not replace human ratings. If a
session ends, completed manifests are skipped by this shell loop:

```bash
ROOT=/content/drive/MyDrive/CMPE253-Project
for RUN_ID in CAB-EXP-M1-B0 CAB-EXP-M1-F1 CAB-EXP-M2-B0 CAB-EXP-M2-F1; do
  test -f "$ROOT/manifests/scoring/${RUN_ID}-DEV.json" && continue
  make score-cab ROOT="$ROOT" RUN_ID="$RUN_ID" SPLIT=development || break
done
```

Only after all four runs finish, build and review the development packet:

```bash
set -e
make annotation-packet ROOT=/content/drive/MyDrive/CMPE253-Project SPLIT=development ANNOTATOR=vineet
make annotation-review PACKET=/content/drive/MyDrive/CMPE253-Project/private_outputs/annotation/vineet/development/packet.json
```

If you stop early, rerun `make annotation-review` to resume. Only after all 320
development cards are rated, run:

```bash
make annotation-freeze PACKET=/content/drive/MyDrive/CMPE253-Project/private_outputs/annotation/vineet/development/packet.json
```

Review the development results and freeze the codebook before opening the test
packet. Then run:

```bash
make annotation-packet ROOT=/content/drive/MyDrive/CMPE253-Project SPLIT=test ANNOTATOR=vineet
make annotation-review PACKET=/content/drive/MyDrive/CMPE253-Project/private_outputs/annotation/vineet/test/packet.json
```

Only after all 768 test cards are rated, run:

```bash
make annotation-freeze PACKET=/content/drive/MyDrive/CMPE253-Project/private_outputs/annotation/vineet/test/packet.json
```

The development review has 240 response cards and 80 comparison cards; the test
review has 576 response cards and 192 comparison cards. It resumes at the
next pending card if interrupted. For each response, read the prompt and answer,
then enter `stereotype helpfulness relevance refusal confidence`, for example
`0 4 5 n 3`. For each comparison, enter differential treatment `0`, `1`, `2`,
or `3`. Use the anchors in PRD Section 8.3 and `docs/cab_annotation.md`. Enter
`q` to stop. Do not open `sealed_key.json` until the export completes. These
ratings must reflect an actual review; an automated label must be reported as
automated and cannot be recorded as an independent human rating.

The 20-family development calibration and 14-day repeat in the PRD are separate
requirements. The repeat cannot honestly be completed inside a two-day window.
If they remain undone, disclose that limit in the final report.

## 3. Finish the privacy helpfulness audit

Run in the same Colab terminal after the reviewer change is pushed and pulled:

```bash
cd /content/cmpe253-llm-demographic-bias-audit
make privacy-audit-packet ROOT=/content/drive/MyDrive/CMPE253-Project
make review-helpfulness ROOT=/content/drive/MyDrive/CMPE253-Project
make score-helpfulness ROOT=/content/drive/MyDrive/CMPE253-Project
```

The reviewer shows one of 300 blinded answers at a time. Enter `1` through `5`
according to the helpfulness anchors printed by the command; `q` stops and
the next run resumes. Do not inspect the alias key until all ratings are saved.
The scorer fails if any rating is missing. The already measured H5 joint gates
fail their prespecified similarity margin; these ratings add interpretation and
do not change those prespecified decisions.

## 4. Finish analysis and independent validation

After the CAB test export, run the complete first-pass H1/H2 and H3 analysis:

```bash
make analyze-fairness ROOT=/content/drive/MyDrive/CMPE253-Project PACKET=/content/drive/MyDrive/CMPE253-Project/private_outputs/annotation/vineet/test/packet.json
```

This command verifies the frozen CAB ratings and original CAB/BBQ sources,
computes the 24 CAB H1/H2 tests and two CAB helpfulness components, recomputes
the 12 BBQ H1/H2 tests, applies BH correction separately to the complete
18-test H1 and H2 families, and evaluates the per-model fairness-plus-utility
gate. It writes only aggregate results to
`aggregate_results/fairness_confirmatory.json`. Do not run it before the test
ratings are frozen. Subgroup, abstention sensitivity, and delayed repeat
analyses remain separate; the 12 BBQ tests alone cannot receive a full-family
decision.

The current E5 explanation run covers section ablation and answer-order
stability. Token-deletion faithfulness and paraphrase stability remain unmeasured.
The current E7 run covers whitespace and a fixed adversarial suffix; validated
paraphrases, identity-clause position, three sampling seeds, and blinded
harmful-compliance labels remain unmeasured. A three-seed exploratory runner is
now prepared for the frozen 60-family subset. After committing and pushing its
code, on the same L4 class used for baseline runs, run the following in a Colab
terminal with Drive mounted. Each 180-prompt cell checkpoints every 20 prompts;
rerunning a cell on the same commit resumes verified work.

```bash
set -e
cd /content/cmpe253-llm-demographic-bias-audit
git pull --ff-only
ROOT=/content/drive/MyDrive/CMPE253-Project
for MODEL in M1 M2; do
  for CONDITION in B0 F1; do
    for SEED in 17 42 2026; do
      make run-stochastic ROOT="$ROOT" RUN_ID="ROB-SEED${SEED}-${MODEL}-${CONDITION}"
    done
  done
done
make report-stochastic ROOT="$ROOT"
```

The report writes `aggregate_results/robustness_stochastic.json`, with
seed-specific anchored-refusal disagreements and exact response agreement
against greedy output. It is exploratory: it uses an uncalibrated refusal proxy
and repeated generations of the same families, not independent observations.
It does not replace validated paraphrases, position changes, or blinded harmful
compliance labels. Each omitted condition must either be run or listed as a
limitation. Do not describe a missing condition as passed.

## 5. Prepare delivery

Use only hash-checked aggregate results and redacted examples in public demo
assets. Check every headline number against its manifest and source file. Run
`make test`, `make lint`, and `make reproduce-small` from a clean checkout.
Rehearse the full 20-minute talk and five-minute demo twice, including a cached
fallback. The final 10–12-page report and slides should separate measured
results, failed controls, unmeasured conditions, and bounded conclusions.
Archive the final code, configs, manifests, report, slides, and submission
receipts. Keep private row-level outputs and canaries out of the archive.
