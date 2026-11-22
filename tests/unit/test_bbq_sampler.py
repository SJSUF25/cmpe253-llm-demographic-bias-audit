import pytest

from data.bbq_sampler import (
    BBQSamplingError,
    cyclic_option_orders,
    enrich_rows,
    select_category,
)


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


def test_official_metadata_join_collapses_only_agreeing_duplicates():
    source = {
        **row(7, "ambig", "neg"),
        "question_index": "12",
        "answer_info": {
            "ans0": ["first", "unknown"],
            "ans1": ["second", "women"],
            "ans2": ["third", "men"],
        },
    }
    metadata = {
        "category": "Gender_identity",
        "example_id": "7",
        "question_index": "12",
        "target_loc": "1",
        "label_type": "label",
        "Known_stereotyped_groups": "first description",
    }
    enriched, report = enrich_rows(
        [source],
        [metadata, {**metadata, "Known_stereotyped_groups": "second description"}],
        "Gender_identity",
    )
    assert enriched[0]["target_group"] == "women"
    assert report["collapsed_metadata_duplicates"] == 1
    with pytest.raises(BBQSamplingError, match="Conflicting supplemental key"):
        enrich_rows(
            [source], [metadata, {**metadata, "target_loc": "2"}], "Gender_identity"
        )


def test_metadata_join_excludes_unusable_target_fields():
    source = {
        **row(8, "ambig", "neg"),
        "question_index": "5",
        "answer_info": {
            "ans0": ["first", ""],
            "ans1": ["second", "women"],
            "ans2": ["third", "men"],
        },
    }
    metadata = {
        "category": "Gender_identity",
        "example_id": "8",
        "question_index": "5",
        "target_loc": "NA",
        "label_type": "label",
    }
    rows, report = enrich_rows([source], [metadata], "Gender_identity")
    assert rows == []
    assert report["excluded_missing_target_loc"] == 1
    rows, report = enrich_rows(
        [source], [{**metadata, "target_loc": "0"}], "Gender_identity"
    )
    assert rows == []
    assert report["excluded_missing_target_group"] == 1
