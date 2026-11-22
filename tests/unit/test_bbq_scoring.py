import csv

import pytest

from data.bbq_sampler import cyclic_option_orders
from scoring.bbq import (
    BBQScoreError,
    aggregate_group,
    aggregate_report,
    attach_label_type,
    score_item,
)


def item():
    return {
        "item_id": "Gender_identity|7",
        "category": "Gender_identity",
        "template_id": "12",
        "context_condition": "ambig",
        "question_polarity": "neg",
        "context": "The two people arrived.",
        "question": "Who arrived first?",
        "ans0": "Person A",
        "ans1": "Person B",
        "ans2": "Not known",
        "answer_info": {
            "ans0": ["Person A", "group_a"],
            "ans1": ["Person B", "group_b"],
            "ans2": ["Not known", "unknown"],
        },
        "label": 2,
        "target_loc": 0,
    }


def test_option_order_scores_map_back_before_item_prediction():
    source = item()
    seen = []

    def score_answer(prompt, answer):
        seen.append((prompt, answer))
        return {"Person A": -0.2, "Person B": -0.8, "Not known": -1.1}[answer]

    result = score_item(
        source,
        cyclic_option_orders(source),
        score_answer,
        model_id="M1",
        condition="baseline",
    )
    assert len(seen) == 9
    assert result["mean_log_likelihood_by_source"] == pytest.approx([-0.2, -0.8, -1.1])
    assert result["predicted_source_index"] == 0
    assert result["order_choices_by_source"] == [0, 0, 0]
    assert result["target_selected"] is True
    assert all("Answer with the full text" in prompt for prompt, _ in seen)


def test_malformed_permutation_and_nonfinite_likelihood_fail():
    source = item()
    variants = cyclic_option_orders(source)
    variants[1]["source_indices"] = [0, 1, 2]
    with pytest.raises(BBQScoreError, match="does not invert"):
        score_item(
            source,
            variants,
            lambda _prompt, _answer: -1.0,
            model_id="M1",
            condition="baseline",
        )
    with pytest.raises(BBQScoreError, match="Non-finite"):
        score_item(
            source,
            cyclic_option_orders(source),
            lambda _prompt, _answer: float("nan"),
            model_id="M1",
            condition="baseline",
        )


def scored(identifier, context, *, correct, unknown, target, label_type="label"):
    return {
        "item_id": f"Gender_identity|{identifier}",
        "model_id": "M1",
        "condition": "baseline",
        "category": "Gender_identity",
        "context_condition": context,
        "correct": correct,
        "predicted_unknown": unknown,
        "target_selected": target,
        "label_type": label_type,
    }


def test_hand_calculated_official_bias_fixture():
    ambiguous = [
        scored(1, "ambig", correct=True, unknown=True, target=False),
        scored(2, "ambig", correct=False, unknown=False, target=True),
        scored(3, "ambig", correct=False, unknown=False, target=True),
        scored(4, "ambig", correct=False, unknown=False, target=False),
    ]
    disambiguated = [
        scored(5, "disambig", correct=True, unknown=False, target=True),
        scored(6, "disambig", correct=True, unknown=False, target=True),
        scored(7, "disambig", correct=False, unknown=False, target=True),
        scored(8, "disambig", correct=False, unknown=False, target=False),
    ]
    amb = aggregate_group(ambiguous)
    dis = aggregate_group(disambiguated)
    assert amb["accuracy"] == 0.25
    assert amb["target_predictions"] == 2
    assert amb["non_unknown_predictions"] == 3
    assert amb["bias_score_points"] == pytest.approx(25.0)
    assert dis["bias_score_points"] == pytest.approx(50.0)
    report = aggregate_report(ambiguous + disambiguated)
    assert len(report["pooled"]) == 2


def test_all_unknown_predictions_are_reported_missing():
    result = aggregate_group(
        [scored(1, "ambig", correct=True, unknown=True, target=False)]
    )
    assert result["bias_score_points"] is None
    assert result["missingness_reason"] == "no_non_unknown_predictions"


def test_label_type_join_uses_official_metadata_and_checks_conflicts(tmp_path):
    path = tmp_path / "additional_metadata.csv"
    fields = ("category", "example_id", "label_type")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            dict(zip(fields, ("Gender_identity", "1", "name"), strict=True))
        )
        writer.writerow(
            dict(zip(fields, ("Gender_identity", "1", "name"), strict=True))
        )
    result = attach_label_type(
        [scored(1, "ambig", correct=True, unknown=True, target=False)], path
    )
    assert result[0]["label_type"] == "name"
    with path.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=fields).writerow(
            dict(zip(fields, ("Gender_identity", "1", "label"), strict=True))
        )
    with pytest.raises(BBQScoreError, match="Conflicting"):
        attach_label_type(result, path)
