"""Check preregistered explanation selection on a shuffled source fixture."""

from collections import Counter

from explain.sample import STRATA, select_items


def test_explain60_is_balanced_and_source_order_independent() -> None:
    rows = []
    for category in ("Gender_identity", "Race_ethnicity", "Race_x_gender"):
        for context, polarity in STRATA:
            for number in range(125):
                example_id = f"{context}-{polarity}-{number}"
                rows.append(
                    {
                        "item_id": f"{category}|{example_id}",
                        "example_id": example_id,
                        "category": category,
                        "context_condition": context,
                        "question_polarity": polarity,
                        "template_id": f"template-{number // 5}",
                    }
                )
    selected = select_items(rows)
    assert selected == select_items(list(reversed(rows)))
    assert len(selected) == 60
    assert Counter(
        (row["category"], row["context_condition"], row["question_polarity"])
        for row in selected
    ) == {
        (category, context, polarity): 5
        for category in ("Gender_identity", "Race_ethnicity", "Race_x_gender")
        for context, polarity in STRATA
    }
