# Resumable CAB Inference Runner

`make run-cab ROOT=/mounted/shared/path RUN_ID=CAB-EXP-M1-B0 SPLIT=development`
runs a registered CAB explicit or implicit cell with one pinned model. `RUN_ID`
must match `configs/experiment_registry.json`; supported conditions are B0
(baseline) and F1 (fairness mitigation). Development outputs are stored under
`private_outputs/pilot/`. A test run uses `SPLIT=test` and writes to the exact
registered private output path. The runner selects only the requested CAB
explicit/implicit mode and split from the frozen prompt manifest. Implicit runs
exclude the 23 source-quality-flagged families according to the verified private
`manifests/cab_implicit_quality_manifest.json`; all explicit primary prompts remain.

The runner writes successive 100-row JSONL files under a private `parts/`
directory. Each part is first written and synced as a temporary file, then
renamed into place. After each full part it writes a checksum manifest under
`manifests/inference/`. If the process is interrupted partway through a block,
only the completed, verified parts are reused; the uncheckpointed prompts are
rerun. A final part may have fewer than 100 rows. When all prompts have a row,
the registered output file is assembled from the parts and checked against
their hashes. On resume, every stored row is checked against its prompt ID,
model revision, condition, seed, decoding hash, code commit, config hash, and
raw-output hash. Conflicting or repeated keys fail visibly.

The unique key is `(prompt_id, model_revision, condition, seed,
decoding_config_hash)`. Infrastructure errors receive one immediate retry with
the identical prompt and decode settings. Other errors become explicit terminal
rows with `status=error`, an error type, and null output fields. A confirmatory
run containing any such row fails after saving its evidence. The manifest holds
row counts, partition hashes, output hash, UTC times, accumulated elapsed time,
environment/GPU data, package-lock hash, code commit, worktree state, input and
config hashes, seed, and error counts. Raw outputs stay in the designated Drive
folder under the documented access exception.

`SPLIT=test` refuses to load a model unless the Git worktree is clean and a
private versioned data-gate receipt exists with the exact fields below. The CAB/BBQ
full-run notebook uses v3; subsequent privacy runs renew it as v4 after the next
governance-lock change.
The full-run Colab notebook first runs `make smoke-model` for each model in that
runtime. Its frozen 20-prompt fixture is generated twice, with reverse execution
order on the second pass. It requires byte-identical output hashes, unchanged
likelihood-selected answers, and a maximum repeated-likelihood delta of `1e-4`.
The preflight receipt is saved under `manifests/full_preflight/`. If it fails,
stop that model's full run.
The historical v1/v2/v3 receipts are retained as evidence of earlier protocol
versions; the runner prefers the newest receipt and validates it against the
current governance lock.
The generic runner checks the same gate for every confirmatory call.
The accelerated route uses `make ai-review-gate ROOT=/mounted/shared/path`; its
receipt names `Codex AI-assisted audit` and declares
`review_method=ai_assisted_no_human_review`. The runner verifies the exact reviewed
packet and evidence hashes. Reports must disclose that no human review occurred.
The AI command writes `data_ai_review_receipt_v4.json` and
`data_gate_approval_v4.json` without overwriting v1/v2/v3. Create the AI receipt only
after the tokenizer-length and quality checks pass.
The human receipt shown below is a separate option after Abhinand's manual QA.
The named approval is process evidence; the program checks receipt
contents and hashes, not a cryptographic signature. Before accepting the receipt,
the runner rehashes the CAB, BBQ, and training-presentation inputs listed in the
token-length manifest, so a changed data file cannot reuse a stale length check.

```json
{
  "schema_version": 1,
  "validation_manifest_sha256": "<SHA-256 of manifests/data_validation_manifest.json>",
  "token_length_manifest_sha256": "<SHA-256 of manifests/token_length_manifest.json>",
  "cab_implicit_quality_manifest_sha256": "<SHA-256 of manifests/cab_implicit_quality_manifest.json>",
  "governance_lock_sha256": "<SHA-256 of results/manifests/governance_lock.json>",
  "token_length_status": "passed",
  "human_review_status": "approved",
  "approved_by": ["Abhinand"]
}
```

The generic `run_inference` function is also usable by later controlled-run
wrappers. Its integration tests deliberately interrupt execution, resume,
verify no omissions or duplicates, exercise one retry, and reject tampered
outputs. The local tests use a fake adapter. Actual Qwen/Phi generation and
storage-mount behavior still require the Colab gate before E2-T02 is Done.
