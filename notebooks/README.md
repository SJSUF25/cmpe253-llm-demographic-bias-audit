# Notebooks

`e0_t04_colab_smoke_test.ipynb` is the required fresh-runtime environment and model
smoke test. Vineet runs it on an L4 after the commit containing its matching `uv.lock`
is pushed. The setup cell selects `/content/drive/MyDrive/CMPE253-Project` when that
shortcut exists. Otherwise copy the folder path from Colab's Files pane and enter a
filesystem path under `/content/drive/`, not the browser's Drive URL. Notebook cells are
orchestration only; reusable logic lives under `src/`.

Vineet's 2026-11-02 environment, M1, and M2 JSON manifests already exist in the
shared Drive `manifests/` folder. Their L4 environment fields agree, and both model
fixtures passed (20 prompts, exact output match, likelihood delta 0.0). A later
2026-11-23 local Colab run produced another passing trio in `/content/manifests`;
attempting to copy it over the older Drive files failed because the capture timestamps
and therefore the bytes differ. The Drive trio is sufficient evidence for E0-T04.

The notebook is now cleared of execution output for clean reproduction. Future runs
write directly to a unique `manifests/e0_t04_runs/<run_id>/` folder in mounted Drive,
so they cannot replace a previous run. Do not rerun expensive model smoke tests just
to reproduce a timestamp or overwrite a passing manifest.

Notebooks may call importable project modules for exploration and presentation. They
must not contain transformations or analysis logic unavailable from project commands.

`e2_t04_pilot_colab.ipynb` orchestrates the frozen pilot. It plans the selection,
runs M1 and M2 in separate restart-safe cells, and builds the combined report. The
reusable selection, checkpoint, and reporting logic lives in `inference.pilot`.

`e3_e4_analysis_colab.ipynb` verifies the completed full CAB/BBQ inventory,
renews the versioned data gate, freezes the explanation sample before attribution,
produces the clustered descriptive fairness
summary, and runs the pinned secondary CAB scorers on CPU. Use a fresh CPU runtime;
no GPU is needed for this notebook. It starts only after the
full-run notebook finishes and its analysis commit is pushed.
That notebook also freezes the CAB robustness60 source selection and builds the
neutral-format and fixed-adversarial input variants before aggregate outcomes
are shown. See `docs/robustness_subset.md`.

`e5_explanations_colab.ipynb` runs the same 60 BBQ items under both models and
conditions, rechecks the original answer likelihoods, saves per-item ablation
effects, and aggregates global feature sensitivity. See
`docs/explanation_ablation.md` for its bounded interpretation.

`e7_robustness_colab.ipynb` runs the frozen whitespace and adversarial CAB
subset under both models and conditions, then summarizes automated refusal-gap
changes. It records the unmeasured paraphrase, position, seed, and human-label
conditions explicitly.

`e6_privacy_colab.ipynb` orchestrates the six measured PII cells, H5 exposure and
utility scoring, the three-epoch Phi QLoRA canary adapter, base/adapter attacks,
and ordinary-task adapter utility. Run it after the full CAB/BBQ notebook finishes
and the matching code is committed. It renews the v4 data gate without overwriting
the v3 receipt used by the earlier full run. Reusable methods live under `src/privacy/`.

`e8_live_demo_colab.ipynb` is an optional presentation notebook. Run its setup,
cached fallback, and frozen development-pair cells first. The last cell loads only
the pinned FP16 Qwen model on L4 and displays four greedy responses for one
female/male identity pair under baseline and fairness instructions. It does not
save generated text. Inspect the outputs privately before screen sharing. If the
GPU step fails, interrupt it and rerun the already prepared cached fallback cell.
The cached results are the measured study evidence; the live pair is illustrative.

