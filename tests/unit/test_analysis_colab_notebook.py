"""Check the analysis handoff notebook remains runnable in cell order."""

import json
from pathlib import Path


def test_analysis_notebook_has_gate_before_scoring() -> None:
    path = Path(__file__).resolve().parents[2] / "notebooks/e3_e4_analysis_colab.ipynb"
    notebook = json.loads(path.read_text())
    code = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    for cell in code:
        compile(cell, str(path), "exec")
    assert len(code) == 5
    assert all(
        command in code[1]
        for command in (
            "ai-review-gate",
            "freeze-explain-sample",
            "build-robustness-perturbations",
            "report-bbq",
        )
    )
    assert "score-fairness-descriptive" in code[2]
    assert "score-cab" in code[3]
    assert "report-cab-secondary" in code[4]
    assert all(
        cell["outputs"] == []
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
