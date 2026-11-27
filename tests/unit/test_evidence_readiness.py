"""Claim readiness keeps failed utility components distinct from missing ratings."""

import hashlib
import json

import pytest

from analysis.descriptive import BBQ_IDS, CAB_IDS
from analysis.evidence import EvidenceError, audit


def _write(path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_evidence_audit_links_sources_and_flags_accuracy_failure(tmp_path) -> None:
    bbq = {
        run_id: _write(tmp_path / "manifests/bbq" / f"{run_id}.json", run_id.encode())
        for run_id in BBQ_IDS
    }
    cab = {
        run_id: _write(
            tmp_path / "manifests/inference" / f"{run_id}.json", run_id.encode()
        )
        for run_id in CAB_IDS
    }
    fairness = {
        "status": "descriptive_only_no_confirmatory_decision",
        "bbq_cells": [{}] * 36,
        "cab_refusal_cells": [{}] * 48,
        "bbq_manifest_sha256": bbq,
        "cab_manifest_sha256": cab,
        "bbq_accuracy_gate_components": [
            {
                "model": model,
                "accuracy_change": -0.05,
                "one_sided_lower95": -0.07,
                "margin": -0.03,
                "accuracy_component_passes": False,
            }
            for model in ("M1", "M2")
        ],
    }
    _write(
        tmp_path / "aggregate_results/fairness_descriptive.json",
        json.dumps(fairness).encode(),
    )
    result = json.loads(audit(tmp_path).read_text())
    by_claim = {row["claim"]: row["status"] for row in result["claims"]}
    assert by_claim["H3 fairness instruction preserves BBQ accuracy"] == (
        "accuracy_component_failed"
    )
    assert by_claim["H1 and H2 human differential-treatment outcomes"] == (
        "not_evaluable_from_this_aggregate"
    )
    assert by_claim["Robustness automated refusal shift"] == "pending"
    scoring = {
        run_id: _write(
            tmp_path / "manifests/scoring" / f"{run_id}.json", run_id.encode()
        )
        for run_id in CAB_IDS
    }
    _write(
        tmp_path / "aggregate_results/cab_secondary_summary.json",
        json.dumps(
            {
                "schema_version": 1,
                "status": "descriptive_secondary_uncalibrated",
                "source_scoring_manifest_sha256": scoring,
            }
        ).encode(),
    )
    measured = json.loads(audit(tmp_path).read_text())
    assert {row["claim"]: row["status"] for row in measured["claims"]}[
        "Secondary CAB diagnostic tables"
    ] == "measured_descriptive"
    (tmp_path / "manifests/scoring" / f"{CAB_IDS[0]}.json").write_text("changed")
    with pytest.raises(EvidenceError, match="source manifest changed"):
        audit(tmp_path)
    (tmp_path / "manifests/scoring" / f"{CAB_IDS[0]}.json").write_bytes(
        CAB_IDS[0].encode()
    )
    (tmp_path / "manifests/bbq" / f"{BBQ_IDS[0]}.json").write_text("tampered")
    with pytest.raises(EvidenceError, match="source manifest changed"):
        audit(tmp_path)
