import pytest

from analysis.statistics import (
    StatisticsError,
    benjamini_hochberg,
    bootstrap_lower_bound,
    exact_mcnemar_improvement,
)


def test_one_sided_exact_mcnemar_and_bh():
    assert exact_mcnemar_improvement(0, 0) == 1.0
    assert exact_mcnemar_improvement(5, 0) == 1 / 32
    assert exact_mcnemar_improvement(0, 5) == 1.0
    assert exact_mcnemar_improvement(1, 1) == 0.75
    assert benjamini_hochberg([0.01, 0.04, 0.03, 1.0]) == pytest.approx(
        [0.04, 0.05333333333333333, 0.05333333333333333, 1.0]
    )
    with pytest.raises(StatisticsError):
        benjamini_hochberg([float("nan")])


def test_seeded_one_sided_lower_bound():
    assert bootstrap_lower_bound([0.0] * 200) == 0.0
    assert bootstrap_lower_bound([-0.1] * 200) == pytest.approx(-0.1)
    assert bootstrap_lower_bound([0, 1], repetitions=100) == bootstrap_lower_bound(
        [0, 1], repetitions=100
    )
