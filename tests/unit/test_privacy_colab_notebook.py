"""Keep the proposal-aligned privacy Colab workflow executable and output-free."""

import json
from pathlib import Path


def test_privacy_notebook_covers_all_six_cells_and_memorization_runs() -> None:
    path = Path(__file__).resolve().parents[2] / "notebooks/e6_privacy_colab.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), f"privacy-cell-{index}", "exec")
    content = path.read_text()
    for marker in (
        "data_gate_approval_v4.json",
        "PII-M1-P0",
        "PII-M2-P2",
        "score-h5",
        "train-canary",
        "attack-canary",
        "compare-adapter-utility",
    ):
        assert marker in content
