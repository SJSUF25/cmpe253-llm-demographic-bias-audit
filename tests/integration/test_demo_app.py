"""Render every cached dashboard tab from a minimal aggregate-only bundle."""

import json
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_cached_app_renders_without_private_outputs(tmp_path, monkeypatch) -> None:
    bundle = {
        "schema_version": 1,
        "mode": "cached_offline",
        "limitations": ["Human ratings pending."],
        "sections": {
            "fairness": {
                "bbq_cells": [
                    {
                        "model": "M1",
                        "condition": "B0",
                        "category": "Gender_identity",
                        "context": "ambig",
                        "accuracy": 0.8,
                    }
                ],
                "bbq_accuracy_gate_components": [
                    {"model": "M1", "accuracy_component_passes": False}
                ],
                "cab_refusal_cells": [],
            }
        },
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    monkeypatch.setenv("DEMO_BUNDLE", str(path))
    app_path = Path(__file__).resolve().parents[2] / "src/demo/app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception
    assert len(app.tabs) == 8
    assert "cached" in app.caption[0].value.lower()
    assert len(app.dataframe) >= 2
