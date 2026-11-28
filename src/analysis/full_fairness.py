"""Join frozen human CAB and verified BBQ tests into complete H1/H2 families."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from analysis import bbq_partial, cab_human
from analysis.family_correction import correct_h1_h2
from scoring import cab


class FullFairnessError(ValueError):
    """The source inventories cannot support a full fairness decision."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combine(cab_result: dict, bbq_result: dict, fairness: dict) -> dict:
    """Join exact test IDs and require both utility components per model."""
    if (
        cab_result.get("status")
        != "cab_human_first_pass_unadjusted_h1_h2_and_h3_component"
        or bbq_result.get("status") != "partial_h1_h2_bbq_only_no_family_decision"
    ):
        raise FullFairnessError("Unverified CAB or BBQ analysis status")
    cab_tests, bbq_tests = cab_result["tests"], bbq_result["tests"]
    if len(cab_tests) != 26 or len(bbq_tests) != 12:
        raise FullFairnessError("The frozen CAB/BBQ test inventories differ")
    corrected = correct_h1_h2(
        [row for row in [*cab_tests, *bbq_tests] if row["family"] in {"H1", "H2"}]
    )
    helpfulness = {row["model"]: row for row in cab_tests if row["family"] == "H3"}
    accuracy = {row["model"]: row for row in fairness["bbq_accuracy_gate_components"]}
    if set(helpfulness) != {"M1", "M2"} or set(accuracy) != {"M1", "M2"}:
        raise FullFairnessError("Missing model utility component")
    model_decisions = []
    for model in ("M1", "M2"):
        improving = [
            row["test_id"]
            for row in corrected
            if row["family"] == "H2"
            and row["model"] == model
            and row.get("estimate", row.get("estimate_points")) > 0
            and row["q_bh18"] < 0.05
        ]
        helpful = helpfulness[model]["noninferiority_passes"]
        accurate = accuracy[model]["accuracy_component_passes"]
        model_decisions.append(
            {
                "model": model,
                "significant_fairness_improvements": improving,
                "helpfulness_noninferiority_passes": helpful,
                "bbq_accuracy_noninferiority_passes": accurate,
                "joint_mitigation_success": bool(improving) and helpful and accurate,
            }
        )
    return {
        "schema_version": 1,
        "status": "full_h1_h2_bh18_first_pass_single_rater",
        "h1_h2_tests": corrected,
        "h3_helpfulness_components": list(helpfulness.values()),
        "h3_bbq_accuracy_components": list(accuracy.values()),
        "model_decisions": model_decisions,
        "limitations": [
            "First-pass ratings come from one reviewer; "
            "no inter-rater reliability claim.",
            "Delayed blinded repeat remains separate from these results.",
        ],
    }


def analyze(root: Path, packet_path: Path) -> dict:
    root = root.expanduser().resolve()
    cab_result = cab_human.analyze(packet_path)
    source_hashes = cab_result["sources_sha256"]["cab_source_hashes"]
    expected_runs = {
        f"CAB-EXP-{model}-{condition}"
        for model in ("M1", "M2")
        for condition in ("B0", "F1")
    }
    if set(source_hashes) != expected_runs:
        raise FullFairnessError("CAB source run inventory differs")
    for run_id in sorted(expected_runs):
        _, _, source = cab._inputs(root, run_id, "test")
        if source != source_hashes[run_id]:
            raise FullFairnessError(f"CAB source changed after packet freeze: {run_id}")
    bbq_result = bbq_partial.audit(root)
    fairness_path = root / "aggregate_results/fairness_descriptive.json"
    fairness = json.loads(fairness_path.read_text())
    if fairness.get(
        "status"
    ) != "descriptive_only_no_confirmatory_decision" or bbq_result[
        "fairness_descriptive_sha256"
    ] != _sha(fairness_path):
        raise FullFairnessError("Fairness descriptive source differs")
    result = combine(cab_result, bbq_result, fairness)
    result["sources_sha256"] = {
        "fairness_descriptive": _sha(fairness_path),
        "cab_packet": cab_result["sources_sha256"]["packet"],
        "cab_ratings_manifest": cab_result["sources_sha256"]["ratings_manifest"],
        "cab_source_manifests": source_hashes,
        "bbq_source_manifests": bbq_result["source_sha256"],
        "analysis_plan": bbq_result["analysis_plan_sha256"],
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.root / "aggregate_results/fairness_confirmatory.json"
    result = analyze(args.root, args.packet)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
