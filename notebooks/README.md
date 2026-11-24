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

