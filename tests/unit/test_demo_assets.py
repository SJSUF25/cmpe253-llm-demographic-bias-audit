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


def test_bundle_includes_only_allowlisted_canary_and_utility_metrics(tmp_path) -> None:
    relative, tables = SOURCES["fairness"]
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    fairness = {"schema_version": 1}
    for name, fields in tables.items():
        fairness[name] = [_row(fields) for _ in range(36 if name == "bbq_cells" else 1)]
    path.write_text(json.dumps(fairness))

    for mode in ("base", "adapter"):
        path = tmp_path / SOURCES[f"canary_{mode}"][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        score = {
            "auc": 1.0,
            "tpr_at_calibrated_1pct_fpr": 1.0,
            "evaluation_fpr": 0.008,
            "calibration_false_positives": 5,
            "ci95": {"auc": [1.0, 1.0], "tpr": [1.0, 1.0], "fpr": [0.002, 0.016]},
            "private_path": "/content/drive/secret",
        }
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "mode": mode,
                    "score_attacks": {"negative_loss": score, "min_k20": score},
                    "member_extraction_by_repetition": {
                        str(repetition): {
                            "records": 100,
                            "exact_rate": 0.02,
                            "ci95": [0.0, 0.05],
                            "canary": "CANARY-PRIVATE",
                        }
                        for repetition in (1, 5, 10)
                    },
                }
            )
        )
    path = tmp_path / SOURCES["adapter_utility"][0]
    utility = {
        "records": 400,
        "base_exact_accuracy": 0.2425,
        "adapter_exact_accuracy": 0.9225,
        "adapter_minus_base_exact_accuracy": 0.68,
        "base_mean_target_loss": 2.394,
        "adapter_mean_target_loss": 0.073,
        "adapter_minus_base_target_loss": -2.321,
        "private_path": "/content/drive/secret",
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "overall": utility,
                "tasks": {
                    task: {**utility, "records": 100}
                    for task in ("addition", "copy_code", "json_field", "uppercase")
                },
            }
        )
    )
    output = build_bundle(tmp_path)
    bundle = json.loads(output.read_text())
    assert bundle["sections"]["canary_adapter"]["scores"][0]["auc"] == 1.0
    assert (
        bundle["sections"]["adapter_utility"]["cells"][0]["adapter_exact_accuracy"]
        == 0.9225
    )
    assert "CANARY-PRIVATE" not in output.read_text()
    assert "/content/drive/secret" not in output.read_text()
