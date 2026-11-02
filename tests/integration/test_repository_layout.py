"""Integration checks for the implementation-ready repository skeleton."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_required_directories_exist() -> None:
    required = (
        "configs",
        "docs",
        "data/raw",
        "data/processed",
        "data/manifests",
        "src/data",
        "src/inference",
        "src/scoring",
        "src/analysis",
        "src/explain",
        "src/privacy",
        "src/demo",
        "tests/unit",
        "tests/integration",
        "results/manifests",
        "reports",
        "notebooks",
    )

    assert all((ROOT / path).is_dir() for path in required)


def test_planning_documents_are_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/PROJECT_TASKS.md" in gitignore
    assert "/LLM_Bias_Audit_PRD.md" in gitignore


def test_makefile_exposes_required_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    required = (
        "setup",
        "download-data",
        "build-data",
        "validate-data",
        "pilot",
        "infer-baseline",
        "infer-mitigation",
        "score",
        "annotate-export",
        "analyze",
        "explain",
        "privacy",
        "test",
        "lint",
        "demo",
        "reproduce-small",
        "storage-init",
        "storage-verify",
        "sync-upstream",
        "setup-colab",
        "capture-environment",
        "smoke-model",
        "governance-lock",
    )

    assert all(f"{target}:" in makefile for target in required)


def test_colab_notebook_and_registry_are_valid() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/e0_t04_colab_smoke_test.ipynb").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (ROOT / "configs/experiment_registry.json").read_text(encoding="utf-8")
    )
    run_ids = [run["id"] for run in registry["runs"]]

    assert notebook["nbformat"] == 4
    assert len(run_ids) == len(set(run_ids))
    assert {run["model"] for run in registry["runs"]} >= {"M1", "M2"}
