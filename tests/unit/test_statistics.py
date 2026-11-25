import pytest

from analysis.statistics import (
    StatisticsError,
    benjamini_hochberg,
    bootstrap_lower_bound,
    cluster_bootstrap_interval,
    cluster_bootstrap_lower_bound,
    cluster_mean,
    cluster_sign_flip_pvalue,
    exact_binomial_greater,
    exact_mcnemar_improvement,
    exact_mcnemar_two_sided,
    matched_odds_ratio,
    missing_mean_bounds,
    noninferiority_pass,
    paired_cluster_permutation_pvalue,
    paired_risk_difference,
    paired_standardized_mean_difference,
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


def test_exact_binomial_mcnemar_and_binary_effects():
    assert exact_binomial_greater(2, 2, 0.5) == 0.25
    assert exact_binomial_greater(0, 4, 0.05) == 1.0
    assert exact_mcnemar_two_sided(4, 0) == 0.125
    assert exact_mcnemar_two_sided(2, 2) == 1.0
    assert exact_mcnemar_two_sided(0, 0) == 1.0
    assert matched_odds_ratio(3, 1) == 3.0
    assert matched_odds_ratio(3, 0) is None
    assert paired_risk_difference([1, 1, 0, 0], [0, 1, 1, 0]) == 0.0
    with pytest.raises(StatisticsError):
        exact_mcnemar_two_sided(True, 1)
    with pytest.raises(StatisticsError):
        exact_binomial_greater(3, 2)
    with pytest.raises(StatisticsError):
        paired_risk_difference([1, 0], [1])


def test_cluster_bootstrap_preserves_whole_templates():
    values = {"template-b": [1.0, 1.0], "template-a": [0.0, 0.0]}
    assert cluster_mean(values) == 0.5
    assert cluster_bootstrap_interval(values, repetitions=10000) == (0.0, 1.0)
    assert cluster_bootstrap_interval(values, repetitions=10000) == (
        0.0,
        1.0,
    )
    assert cluster_bootstrap_lower_bound(values, repetitions=10000) == 0.0
    assert cluster_bootstrap_interval({"only": [2.0, 4.0]}, repetitions=100) == (
        3.0,
        3.0,
    )
    with pytest.raises(StatisticsError):
        cluster_bootstrap_interval({"bad": [float("nan")]})


def test_cluster_sign_flip_is_seeded_and_flips_a_family_once():
    paired = {"family-a": [1.0, 1.0], "family-b": [1.0, 1.0]}
    result = cluster_sign_flip_pvalue(paired, alternative="greater", draws=10000)
    assert result == cluster_sign_flip_pvalue(
        dict(reversed(list(paired.items()))), alternative="greater", draws=10000
    )
    assert 0.2 < result < 0.3  # Two independent families, not four prompt rows.
    assert (
        cluster_sign_flip_pvalue({"zero": [0.0]}, alternative="two_sided", draws=10000)
        == 1.0
    )
    with pytest.raises(StatisticsError):
        cluster_sign_flip_pvalue(paired, alternative="less")


def test_cluster_swap_permutation_recomputes_nonlinear_metric():
    pairs = {"template-b": ([1.0, 1.0], [0.0, 0.0]), "template-a": ([1.0], [0.0])}

    def absolute_gap_reduction(swapped):
        baseline = [value for left, _right in swapped.values() for value in left]
        mitigation = [value for _left, right in swapped.values() for value in right]
        return abs(sum(baseline) / len(baseline)) - abs(
            sum(mitigation) / len(mitigation)
        )

    result = paired_cluster_permutation_pvalue(
        pairs, absolute_gap_reduction, alternative="greater", draws=10000
    )
    assert 0.2 < result < 0.3
    assert result == paired_cluster_permutation_pvalue(
        dict(reversed(list(pairs.items()))),
        absolute_gap_reduction,
        alternative="greater",
        draws=10000,
    )
    with pytest.raises(StatisticsError):
        paired_cluster_permutation_pvalue(
            pairs, lambda _rows: float("nan"), alternative="greater", draws=1
        )


def test_noninferiority_missing_bounds_and_standardized_effect():
    assert noninferiority_pass(-0.24, -0.25)
    assert not noninferiority_pass(-0.25, -0.25)
    assert missing_mean_bounds([0.0, 1.0], 1, 0.0, 1.0) == pytest.approx((1 / 3, 2 / 3))
    assert paired_standardized_mean_difference([1.0, 2.0, 3.0]) == 2.0
    assert paired_standardized_mean_difference([1.0, 1.0]) is None
    with pytest.raises(StatisticsError):
        missing_mean_bounds([2.0], 1, 0.0, 1.0)
