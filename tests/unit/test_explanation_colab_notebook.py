"""Check the explanation Colab handoff keeps its source gate and four cells."""

import json
from pathlib import Path


def test_explanation_notebook_compiles_in_order() -> None:
    path = Path(__file__).resolve().parents[2] / "notebooks/e5_explanations_colab.ipynb"
    notebook = json.loads(path.read_text())
    code = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    for cell in code:
        compile(cell, str(path), "exec")
    assert len(code) == 5
    assert "ai-review-gate" in code[1] and "freeze-explain-sample" in code[1]
    assert all(run_id in code[2] for run_id in ("EXPLAIN-M1-B0", "EXPLAIN-M1-F1"))
    assert all(run_id in code[3] for run_id in ("EXPLAIN-M2-B0", "EXPLAIN-M2-F1"))
    assert "report-explain" in code[4]
