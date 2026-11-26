"""Keep the full CAB/BBQ Colab runner executable and free of saved outputs."""

import json
from pathlib import Path


def test_full_run_notebook_is_clean_and_covers_registered_cells():
    path = Path(__file__).resolve().parents[2] / "notebooks/e2_t05_t06_full_colab.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), f"full-cell-{index}", "exec")
    text = path.read_text()
    for marker in (
        "data_gate_approval_v3.json",
        "batch_order_reversed_on_second_pass",
        "likelihood_selected_answers_match",
        "CAB-EXP-M1-B0",
        "CAB-IMP-M2-F1",
        "BBQ-M1-B0",
        "BBQ-M2-F1",
        "report-bbq",
    ):
        assert marker in text
