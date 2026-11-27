"""Check family-paired robustness gaps and their subgroup inventory."""

from robustness.report import paired_gaps


def test_paired_gaps_keep_family_as_the_unit(monkeypatch) -> None:
    monkeypatch.setattr(
        "robustness.report.clustered_interval",
        lambda rows, key, statistic: {
            "lower": statistic(rows),
            "upper": statistic(rows),
        },
    )
    rows = []
    for attribute, groups in (
        ("gender", ("female", "male")),
        ("race", ("white", "black", "asian", "hispanic")),
    ):
        for index in range(30):
            for group in groups:
                rows.append(
                    {
                        "scenario_id": f"{attribute}-{index}",
                        "attribute": attribute,
                        "group": group,
                        "original_refusal": group == "female",
                        "perturbed_refusal": False,
                    }
                )
    result = paired_gaps(rows)
    assert len(result) == 4
    female = next(row for row in result if row["attribute"] == "gender")
    assert female["families"] == 30
    assert female["original_refusal_gap"] == 1
    assert female["perturbed_refusal_gap"] == 0
    assert female["gap_change"] == -1
