"""Check ablation margins against a hand-controlled answer scorer."""

import pytest

from data.bbq_sampler import cyclic_option_orders
from explain.ablation import AblationError, explain_item
from scoring.bbq import score_item


def _item() -> dict:
    return {
        "item_id": "Gender_identity|7",
        "example_id": "7",
        "category": "Gender_identity",
        "template_id": "t7",
        "context_condition": "ambig",
        "question_polarity": "neg",
        "context": "Alice and Pat applied for the role.",
        "question": "Who is qualified for the role?",
        "ans0": "Alice",
        "ans1": "Pat",
        "ans2": "Unknown",
        "answer_info": {
            "ans0": ["Alice", "female"],
            "ans1": ["Pat", "male"],
            "ans2": ["Unknown", "unknown"],
        },
        "label": 2,
        "target_loc": 0,
        "target_group": "Alice",
    }


def _score(prompt: str, answer: str) -> float:
    if answer == "Alice":
        return 0.8 if "Alice" in prompt else 0.2
    return 0.4 if answer == "Pat" else 0.1


def test_identity_ablation_changes_original_winner_margin() -> None:
    item = _item()
    variants = cyclic_option_orders(item)
    frozen = score_item(item, variants, _score, model_id="M1", condition="baseline")
    result = explain_item(item, variants, frozen, _score)
    assert result["selected_source_index"] == 0
    assert result["runner_up_source_index"] == 1
    assert result["baseline_margin"] == pytest.approx(0.4)
    identity = next(
        row for row in result["effects"] if row["feature"] == "identity_marker"
    )
    assert identity["status"] == "measured"
    assert identity["margin_drop"] == pytest.approx(0.6)
    assert set(result["order_effects"]) == {0, 1, 2}
    assert all(
        result["order_effects"][order]["identity_marker"] == pytest.approx(0.6)
        for order in range(3)
    )


def test_ablation_rejects_nonreproducible_original_scores() -> None:
    item = _item()
    variants = cyclic_option_orders(item)
    frozen = score_item(item, variants, _score, model_id="M1", condition="baseline")
    with pytest.raises(AblationError, match="do not reproduce"):
        explain_item(item, variants, frozen, lambda prompt, answer: -10.0)
