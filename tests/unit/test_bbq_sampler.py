import pytest

from data.bbq_sampler import BBQSamplingError, cyclic_option_orders, select_category


def row(identifier, context, polarity, template="t", group="g"):
    return {
        "category": "Gender_identity",
        "example_id": identifier,
        "context_condition": context,
        "question_polarity": polarity,
        "template_id": template,
        "target_group": group,
        "ans0": "first",
        "ans1": "second",
        "ans2": "third",
        "label": 1,
    }


def test_balanced_selection_is_independent_of_source_order():
    rows = [
        row(1, "ambig", "neg"),
        row(2, "ambig", "neg"),
        row(3, "ambig", "nonneg"),
        row(4, "ambig", "nonneg"),
        row(5, "disambig", "neg"),
        row(6, "disambig", "neg"),
        row(7, "disambig", "nonneg"),
        row(8, "disambig", "nonneg"),
    ]
    first, report = select_category(rows, "Gender_identity", target=8)
    second, second_report = select_category(reversed(rows), "Gender_identity", target=8)
    assert first == second
    assert report == second_report
    assert set(report["realized_by_stratum"].values()) == {2}


def test_deficit_redistribution_and_conflicting_duplicate():
    rows = [
        row(1, "ambig", "neg"),
        row(2, "ambig", "nonneg"),
        row(3, "ambig", "nonneg"),
        row(4, "ambig", "nonneg"),
        row(5, "disambig", "neg"),
        row(6, "disambig", "neg"),
        row(7, "disambig", "nonneg"),
        row(8, "disambig", "nonneg"),
    ]
    selected, report = select_category(rows, "Gender_identity", target=8)
    assert len(selected) == 8
    assert report["realized_by_stratum"]["ambiguous|negative"] == 1
    with pytest.raises(BBQSamplingError, match="Conflicting duplicate"):
        select_category(
            rows + [{**rows[0], "question": "changed"}], "Gender_identity", target=8
        )


def test_cyclic_orders_preserve_and_invert_correct_label():
    variants = cyclic_option_orders(row(1, "ambig", "neg"))
    assert len(variants) == 3
    for variant in variants:
        assert variant["answers"][variant["correct_index"]] == "second"
        assert variant["source_indices"][variant["correct_index"]] == 1
