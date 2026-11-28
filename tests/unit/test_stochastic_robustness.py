import pytest

from robustness.stochastic import RUN_IDS, SEEDS, StochasticRunError
from robustness.stochastic_report import summarize


def _rows():
    rows = []
    for model in ("M1", "M2"):
        for condition in ("B0", "F1"):
            for seed in SEEDS:
                for attribute, groups in (
                    ("gender", ("female", "male")),
                    ("race", ("white", "black", "asian", "hispanic")),
                ):
                    for index in range(30):
                        for group in groups:
                            rows.append(
                                {
                                    "model": model,
                                    "condition": condition,
                                    "seed": seed,
                                    "attribute": attribute,
                                    "scenario_id": f"{attribute}-{index}",
                                    "prompt_id": f"{attribute}-{index}-{group}",
                                    "sampled_refusal": seed == 17,
                                    "greedy_refusal": False,
                                    "sampled_output": "sampled",
                                    "greedy_output": "greedy",
                                }
                            )
    return rows


def test_three_frozen_seeds_have_complete_scenario_inventory():
    assert len(RUN_IDS) == 12
    cells = summarize(_rows())
    assert len(cells) == 24
    assert {row["scenario_families"] for row in cells} == {30}
    assert all(
        row["refusal_disagreement_with_greedy"] == 1
        for row in cells
        if row["seed"] == 17
    )
    assert all(
        row["refusal_disagreement_with_greedy"] == 0
        for row in cells
        if row["seed"] != 17
    )


def test_missing_prompt_fails_inventory_check():
    with pytest.raises(StochasticRunError, match="inventory"):
        summarize(_rows()[:-1])
