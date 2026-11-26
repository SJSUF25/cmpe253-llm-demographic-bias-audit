"""Keep the Colab pilot notebook reviewable and free of saved execution data."""

import json
from pathlib import Path


def test_pilot_notebook_cells_compile_and_have_no_outputs():
    path = Path(__file__).resolve().parents[2] / "notebooks/e2_t04_pilot_colab.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 13
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            compile("".join(cell["source"]), f"pilot-cell-{index}", "exec")
    text = path.read_text()
    assert "ACTION=plan" in text
    assert "MODEL=M1" in text and "MODEL=M2" in text
    assert "ACTION=report" in text


def test_pilot_v2_notebook_is_clean_and_renews_gate():
    path = Path(__file__).resolve().parents[2] / "notebooks/e2_t04_pilot_v2_colab.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            compile("".join(cell["source"]), f"pilot-v2-cell-{index}", "exec")
    text = path.read_text()
    assert "ai-review-gate" in text
    assert "data_gate_approval_v2.json" in text
    assert "pilot_v2/pilot_report.json" in text
