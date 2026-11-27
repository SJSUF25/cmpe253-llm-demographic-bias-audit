"""The cached demo bundle must never copy row-level text or private paths."""

import json

import pytest

from demo.assets import SOURCES, DemoAssetError, build_bundle


def _row(fields: tuple[str, ...]) -> dict:
    row = {field: 0 for field in fields}
    row.update({"model": "M1", "condition": "B0", "category": "Gender_identity"})
    return row


def test_bundle_projects_aggregate_fields_only(tmp_path) -> None:
    relative, tables = SOURCES["fairness"]
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    artifact = {
        "schema_version": 1,
        "status": "descriptive_only_no_confirmatory_decision",
    }
    for name, fields in tables.items():
        artifact[name] = [_row(fields) for _ in range(36 if name == "bbq_cells" else 1)]
        artifact[name][0]["raw_output"] = "private fictional email: test@example.com"
        artifact[name][0]["private_path"] = "/content/drive/secret"
    path.write_text(json.dumps(artifact))
    output = build_bundle(tmp_path)
    bundle = json.loads(output.read_text())
    manifest = json.loads((tmp_path / "demo_assets/manifest.json").read_text())
    serialized = output.read_text()
    assert bundle["mode"] == "cached_offline"
    assert len(bundle["sections"]["fairness"]["bbq_cells"]) == 36
    assert "test@example.com" not in serialized
    assert "private_path" not in serialized
    assert "/content/drive" not in serialized
    assert "privacy" not in bundle["sections"]
    assert manifest["reviewed_examples_included"] is False
    assert manifest["included_sections"] == ["fairness"]


def test_bundle_rejects_unsafe_aggregate_label(tmp_path) -> None:
    relative, tables = SOURCES["fairness"]
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    artifact = {"schema_version": 1}
    for name, fields in tables.items():
        artifact[name] = [_row(fields) for _ in range(36 if name == "bbq_cells" else 1)]
    artifact["bbq_cells"][0]["category"] = "test@example.com"
    path.write_text(json.dumps(artifact))
    with pytest.raises(DemoAssetError, match="unsuitable"):
        build_bundle(tmp_path)
