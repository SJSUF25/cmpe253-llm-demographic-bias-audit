"""Check that the explanation summary retains every model and category cell."""

from explain.ablation import FEATURES
from explain.report import summarize, summarize_order_stability


def test_summary_keeps_missing_identity_markers_visible(monkeypatch) -> None:
    monkeypatch.setattr(
        "explain.report.clustered_interval",
        lambda rows, key, statistic: {
            "lower": statistic(rows),
            "upper": statistic(rows),
        },
    )
    rows = []
    for model in ("M1", "M2"):
        for condition in ("B0", "F1"):
            for category in ("Gender_identity", "Race_ethnicity", "Race_x_gender"):
                for index in range(20):
                    rows.append(
                        {
                            "model": model,
                            "condition": condition,
                            "category": category,
                            "template_id": f"{category}-{index // 2}",
                            "effects": [
                                {
                                    "feature": feature,
                                    "margin_drop": None
                                    if feature == "identity_marker" and index == 0
                                    else 0.1,
                                }
                                for feature in FEATURES
                            ],
                        }
                    )
    cells = summarize(rows)
    assert len(cells) == 60
    identity = next(
        cell
        for cell in cells
        if cell["model"] == "M1"
        and cell["condition"] == "B0"
        and cell["category"] == "Gender_identity"
        and cell["feature"] == "identity_marker"
    )
    assert identity["measured_items"] == 19
    assert identity["not_present_items"] == 1


def test_order_stability_retains_all_pairs_and_missingness(monkeypatch) -> None:
    monkeypatch.setattr(
        "explain.report.clustered_interval",
        lambda rows, key, statistic: {
            "lower": statistic(rows),
            "upper": statistic(rows),
        },
    )
    rows = []
    for model in ("M1", "M2"):
        for condition in ("B0", "F1"):
            for category in ("Gender_identity", "Race_ethnicity", "Race_x_gender"):
                for index in range(20):
                    rows.append(
                        {
                            "model": model,
                            "condition": condition,
                            "category": category,
                            "template_id": f"{category}-{index // 2}",
                            "order_effects": {
                                str(order): {
                                    feature: (
                                        None
                                        if feature == "identity_marker" and index == 0
                                        else float(position)
                                    )
                                    for position, feature in enumerate(FEATURES)
                                }
                                for order in range(3)
                            },
                        }
                    )
    cells = summarize_order_stability(rows)
    assert len(cells) == 36
    assert all(cell["spearman"]["mean"] == 1 for cell in cells)
    assert all(cell["top2_overlap"]["mean"] == 1 for cell in cells)
    assert all(cell["items"] == 20 for cell in cells)
