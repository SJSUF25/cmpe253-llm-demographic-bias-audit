"""Exact confirmatory family membership and frozen statistical choices."""

import json

import pytest

from analysis.plan import CONFIG, COUNTS, PlanError, expected_plan, validate_plan


def test_locked_plan_has_all_and_only_prespecified_tests():
    plan = validate_plan(CONFIG)
    assert plan == expected_plan()
    assert len(plan["tests"]) == 48
    assert plan["test_counts"] == COUNTS
    assert len({test["id"] for test in plan["tests"]}) == 48
    assert {test["margin"] for test in plan["tests"] if test["family"] == "H3"} == {
        -0.25,
        -0.03,
    }
    assert {
        test["margin"] for test in plan["tests"] if test["family"] == "H5_utility"
    } == {-0.05}


def test_plan_rejects_extra_or_changed_tests(tmp_path):
    plan = expected_plan()
    plan["tests"].append(plan["tests"][0])
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(PlanError):
        validate_plan(path)
    plan = expected_plan()
    plan["tests"][0]["alternative"] = "two_sided"
    path.write_text(json.dumps(plan))
    with pytest.raises(PlanError):
        validate_plan(path)
