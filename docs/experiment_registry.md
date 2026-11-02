# Experiment Registry and Change Control

Registry version 1.0.0 is the human-readable companion to
`configs/experiment_registry.json`. It becomes frozen when Vineet Malewar and
Abhinandan approve its governance-lock hash after both complete E0-T04.

## Locked condition vocabulary

| ID | Meaning | Primary use |
|---|---|---|
| B0 | Baseline instruction, greedy FP16 | CAB and BBQ baseline |
| F1 | Fixed PRD fairness instruction, greedy FP16 | Required fairness ablation |
| P0 | Raw deterministic synthetic PII input | Privacy baseline |
| P1 | Input-side Presidio and exact-rule redaction | Privacy control 1 |
| P2 | P1 plus deterministic output filtering | Privacy control 2 |

M1 and M2 always resolve through `configs/upstream_registry.json`; floating branches
are invalid. Each run uses seed 253 unless the registry/PRD explicitly assigns the
robustness seeds 17, 42, and 2026. The full run ID list and private output location are
machine-readable in `configs/experiment_registry.json`.

## Run families

- `CAB-EXP-*`: all locked explicit CAB prompts, both models, B0 and F1.
- `CAB-IMP-*`: matched implicit identity-signal evaluation, both models and conditions.
- `BBQ-*`: the deterministic 1,500-item sample with all three answer orders.
- `ROB-*`: the frozen 60-family perturbation/adversarial matrix and stochastic seeds.
- `PII-*`: the sealed 200-record privacy test under P0, P1, and P2.
- `CANARY-M2-QLORA`: separate 4-bit controlled-adapter membership experiment.
- `EXPLAIN-BBQ-60`: preregistered local/global explanation sample and validation.

Every inference record must contain the run ID, model/revision, prompt/scenario ID,
condition, seed, decoding parameters, input hash, output, status, runtime, GPU and
software-manifest hashes. Resuming a run skips records only when the complete key and
input/config hashes match. Failures remain explicit records and are never silently
dropped.

## Freeze and approval procedure

1. Run both E0-T04 model smoke tests and store their passing JSON manifests.
2. Run `make governance-lock`; inspect the listed files and SHA-256 values.
3. Vineet and Abhinandan independently compare the lock with the approved PRD.
4. Both record approval, date, lock SHA-256, and E0-T04 manifest hashes in the task
   tracker before generating confirmatory outputs.

After freeze, any semantic change requires a new registry version, a new governance
lock, both-member approval, and rerunning every affected experiment. Typographical
documentation fixes that do not alter execution or interpretation are logged but do
not force reruns. No results may be viewed before deciding whether a change is
semantic. Exploratory work must use a visibly different ID and cannot be promoted to
the confirmatory family.

## Output and release controls

Raw generations, human-annotation exports, privacy records, canary membership labels,
and adapters remain in their named private Drive folders. Only aggregate results,
reviewed redacted examples, manifests without sensitive fields, and figures may move
into Git. Checksums verify transfers and backups; a checksum never substitutes for
access control.
