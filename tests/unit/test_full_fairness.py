import pytest

from analysis.full_fairness import FullFairnessError, combine
from analysis.plan import expected_tests


def _inputs():
    cab, bbq = [], []
    for test in expected_tests():
        if test["dataset"] not in {"CAB", "BBQ"} or test["family"] not in {
            "H1",
            "H2",
            "H3",
        }:
            continue
        if test["dataset"] == "BBQ" and test["family"] == "H3":
            continue
        row = {
            "test_id": test["id"],
            "family": test["family"],
            "model": test["model"],
            "p_unadjusted": 0.001,
            "estimate": 1.0,
            "noninferiority_passes": True,
        }
        (cab if test["dataset"] == "CAB" else bbq).append(row)
    return (
        {
            "status": "cab_human_first_pass_unadjusted_h1_h2_and_h3_component",
            "tests": cab,
        },
        {"status": "partial_h1_h2_bbq_only_no_family_decision", "tests": bbq},
        {
            "bbq_accuracy_gate_components": [
                {"model": "M1", "accuracy_component_passes": False},
                {"model": "M2", "accuracy_component_passes": True},
            ]
        },
    )


def test_complete_family_and_joint_utility_gate():
    result = combine(*_inputs())
    assert len(result["h1_h2_tests"]) == 36
    assert {row["q_bh18"] for row in result["h1_h2_tests"]} == {0.001}
    assert result["model_decisions"][0]["joint_mitigation_success"] is False
    assert result["model_decisions"][1]["joint_mitigation_success"] is True


def test_missing_test_cannot_be_corrected():
    cab, bbq, fairness = _inputs()
    cab["tests"].pop()
    with pytest.raises(FullFairnessError, match="inventories"):
        combine(cab, bbq, fairness)
