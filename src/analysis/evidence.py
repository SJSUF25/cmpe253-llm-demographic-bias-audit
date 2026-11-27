"""Audit which project claims have measured, hash-linked evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from analysis.descriptive import BBQ_IDS, CAB_IDS


class EvidenceError(ValueError):
    """An aggregate result or one of its recorded sources differs."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_sources(root: Path, recorded: dict, subdir: str, expected: set[str]) -> None:
    if not isinstance(recorded, dict) or set(recorded) != expected:
        raise EvidenceError(f"{subdir} source inventory differs")
    for run_id, digest in recorded.items():
        path = root / "manifests" / subdir / f"{run_id}.json"
        if not path.is_file() or _sha(path) != digest:
            raise EvidenceError(f"{run_id} source manifest changed")


def audit(root: Path) -> Path:
    """Separate completed measurements, failed gates, and unmeasured claims."""
    root = root.expanduser().resolve()
    fairness_path = root / "aggregate_results/fairness_descriptive.json"
    if not fairness_path.is_file():
        raise EvidenceError("Fairness descriptive aggregate is missing")
    fairness = json.loads(fairness_path.read_text())
    if (
        fairness.get("status") != "descriptive_only_no_confirmatory_decision"
        or len(fairness.get("bbq_cells", [])) != 36
        or len(fairness.get("cab_refusal_cells", [])) != 48
        or len(fairness.get("bbq_accuracy_gate_components", [])) != 2
    ):
        raise EvidenceError("Fairness descriptive inventory differs")
    _check_sources(root, fairness.get("bbq_manifest_sha256"), "bbq", set(BBQ_IDS))
    _check_sources(root, fairness.get("cab_manifest_sha256"), "inference", set(CAB_IDS))
    accuracy = fairness["bbq_accuracy_gate_components"]
    by_model = {row["model"]: row for row in accuracy}
    if set(by_model) != {"M1", "M2"}:
        raise EvidenceError("BBQ utility model inventory differs")
    claims = [
        {
            "claim": "Two-model CAB and BBQ full evaluation",
            "status": "measured",
            "basis": "Twelve source manifests match the fairness aggregate hashes.",
        },
        {
            "claim": "BBQ bias and accuracy with template uncertainty",
            "status": "measured_descriptive",
            "basis": "Thirty-six source-item cells and clustered intervals exist.",
        },
        {
            "claim": "H1 and H2 human differential-treatment outcomes",
            "status": "not_evaluable_from_this_aggregate",
            "basis": "Human ratings and calibration are separate pending evidence.",
        },
        {
            "claim": "H3 fairness instruction preserves BBQ accuracy",
            "status": "accuracy_component_failed"
            if any(not row["accuracy_component_passes"] for row in accuracy)
            else "accuracy_component_passed_only",
            "basis": {
                model: {
                    "accuracy_change": row["accuracy_change"],
                    "lower95": row["one_sided_lower95"],
                    "margin": row["margin"],
                    "component_passes": row["accuracy_component_passes"],
                }
                for model, row in sorted(by_model.items())
            },
        },
    ]
    optional = {
        "Secondary CAB diagnostic tables": (
            "aggregate_results/cab_secondary_summary.json",
            "descriptive_secondary_uncalibrated",
        ),
        "Explanation section ablation and answer-order stability": (
            "aggregate_results/explain_summary.json",
            "descriptive_section_ablation_and_order_stability",
        ),
        "Robustness automated refusal shift": (
            "aggregate_results/robustness_refusal.json",
            "descriptive_automated_refusal_only",
        ),
        "Synthetic PII exposure and utility": (
            "aggregate_results/privacy_h5.json",
            None,
        ),
    }
    optional_hashes = {}
    for claim, (relative, expected_status) in optional.items():
        path = root / relative
        if path.is_file():
            value = json.loads(path.read_text())
            if value.get("schema_version") != 1 or (
                expected_status and value.get("status") != expected_status
            ):
                raise EvidenceError(f"Optional aggregate schema differs: {claim}")
            if claim == "Secondary CAB diagnostic tables":
                _check_sources(
                    root,
                    value.get("source_scoring_manifest_sha256"),
                    "scoring",
                    set(CAB_IDS),
                )
                status = "measured_descriptive"
            else:
                status = "aggregate_present_source_not_reverified"
            optional_hashes[relative] = _sha(path)
        else:
            status = "pending"
        claims.append({"claim": claim, "status": status, "source": relative})
    payload = {
        "schema_version": 1,
        "status": "evidence_readiness_not_final_hypothesis_decisions",
        "fairness_aggregate_sha256": _sha(fairness_path),
        "optional_aggregate_sha256": optional_hashes,
        "claims": claims,
        "limitations": [
            "The BBQ accuracy failure is one measured component of H3.",
            "Automated refusal is uncalibrated against human labels.",
            "No pending experiment is treated as a null or successful result.",
        ],
    }
    output = root / "aggregate_results/evidence_readiness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(audit(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
