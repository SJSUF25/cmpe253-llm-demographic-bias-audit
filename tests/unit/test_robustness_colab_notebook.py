"""Check the robustness notebook's gate and eight registered run cells."""

import json
from pathlib import Path


def test_robustness_notebook_compiles_and_lists_all_cells() -> None:
    path = Path(__file__).resolve().parents[2] / "notebooks/e7_robustness_colab.ipynb"
    notebook = json.loads(path.read_text())
    code = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    for cell in code:
        compile(cell, str(path), "exec")
    assert len(code) == 5
    assert "build-robustness-perturbations" in code[1]
    for model, cell in (("M1", code[2]), ("M2", code[3])):
        for kind in ("SPACE", "ADV"):
            for condition in ("B0", "F1"):
                assert f"ROB-{kind}-{model}-{condition}" in cell
    assert "report-robustness" in code[4]
