"""Check official BBQ scoring and whole-cluster uncertainty on small fixtures."""

import pytest

from analysis import descriptive
from analysis.descriptive import DescriptiveError, bbq_estimate, clustered_interval


def test_bbq_ambiguous_bias_matches_hand_calculated_official_fixture() -> None:
    rows = [
        {"correct": True, "predicted_unknown": True, "target_selected": False},
        {"correct": False, "predicted_unknown": False, "target_selected": True},
        {"correct": False, "predicted_unknown": False, "target_selected": True},
        {"correct": False, "predicted_unknown": False, "target_selected": False},
    ]
    for row in rows:
        row["context_condition"] = "ambig"
    measured = bbq_estimate(rows)
    assert measured["accuracy"] == 0.25
    assert measured["bias_score_points"] == pytest.approx(25.0)


def test_cluster_interval_keeps_all_rows_of_a_drawn_cluster() -> None:
    rows = [
        {"template_id": "a", "value": 0.0},
        {"template_id": "a", "value": 0.0},
        {"template_id": "b", "value": 1.0},
        {"template_id": "b", "value": 1.0},
    ]
    seen = []

    def statistic(sample):
        values = [row["value"] for row in sample]
        assert values.count(0.0) % 2 == values.count(1.0) % 2 == 0
        seen.append(values)
        return sum(values) / len(values)

    result = clustered_interval(rows, "template_id", statistic, repetitions=100)
    assert result["valid_draws"] == 100
    assert result["lower"] == 0.0 and result["upper"] == 1.0
    assert len(seen) == 100


def test_cluster_interval_rejects_missing_cluster_id() -> None:
    with pytest.raises(DescriptiveError, match="cluster key"):
        clustered_interval([{"value": 1}], "template_id", lambda _: 1.0)


def test_identity_shift_pairs_only_common_scenario_families(monkeypatch) -> None:
    monkeypatch.setattr(
        descriptive,
        "clustered_interval",
        lambda rows, cluster_key, statistic: {"lower": 0.0, "upper": 0.0},
    )
    outputs = {}
    for model in ("M1", "M2"):
        for condition in ("B0", "F1"):
            for mode in ("EXP", "IMP"):
                rows = []
                for scenario_id, attribute, groups in (
                    ("gender-common", "gender", ("female", "male")),
                    (
                        "race-common",
                        "race",
                        ("white", "black", "asian", "hispanic"),
                    ),
                ):
                    for group in groups:
                        rows.append(
                            {
                                "scenario_id": scenario_id,
                                "attribute": attribute,
                                "group": group,
                                "refusal": mode == "EXP" and group == "female",
                            }
                        )
                if mode == "EXP":
                    rows.extend(
                        [
                            {
                                "scenario_id": "explicit-only",
                                "attribute": "gender",
                                "group": group,
                                "refusal": False,
                            }
                            for group in ("female", "male")
                        ]
                    )
                outputs[f"CAB-{mode}-{model}-{condition}"] = rows
    measured = descriptive._cab_identity_shift(outputs)
    assert len(measured) == 16
    gender = next(
        row
        for row in measured
        if row["model"] == "M1"
        and row["condition"] == "B0"
        and row["attribute"] == "gender"
    )
    assert gender["common_families"] == 1
    assert gender["explicit_gap"] == 1
    assert gender["implicit_gap"] == 0
    assert gender["implicit_minus_explicit"] == -1
