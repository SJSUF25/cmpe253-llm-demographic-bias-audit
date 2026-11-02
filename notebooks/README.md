# Notebooks

`e0_t04_colab_smoke_test.ipynb` is the required fresh-runtime environment and model
smoke test. Run it independently from both team members' Colab Pro accounts after the
commit containing its matching `uv.lock` is pushed. Notebook cells are orchestration
only; reusable logic lives under `src/`.

Notebooks may call importable project modules for exploration and presentation. They
must not contain transformations or analysis logic unavailable from project commands.

