"""Small, deterministic statistical primitives for preregistered paired analyses."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any


class StatisticsError(ValueError):
    """A statistical input is empty, non-finite, or outside its contract."""


def _check_counts(successes: int, trials: int) -> None:
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, int)
        or not isinstance(trials, int)
        or trials < 0
        or not 0 <= successes <= trials
    ):
        raise StatisticsError(
            "Binomial counts must be integers with 0 <= successes <= trials"
        )


def exact_binomial_greater(
    successes: int, trials: int, null_rate: float = 0.05
) -> float:
    """Exact one-sided tail P[X >= successes] under Binomial(trials, null_rate)."""
    _check_counts(successes, trials)
    if trials == 0 or not math.isfinite(null_rate) or not 0 <= null_rate <= 1:
        raise StatisticsError("Binomial test needs trials and a null rate in [0, 1]")
    if successes == 0:
        return 1.0
    return min(
        1.0,
        sum(
            math.comb(trials, count)
            * null_rate**count
            * (1 - null_rate) ** (trials - count)
            for count in range(successes, trials + 1)
        ),
    )


def exact_mcnemar_two_sided(first_only: int, second_only: int) -> float:
    """Doubled smaller exact Binomial(0.5) tail for paired binary discordances."""
    _check_discordances(first_only, second_only)
    total = first_only + second_only
    if total == 0:
        return 1.0
    tail = (
        sum(
            math.comb(total, count) for count in range(min(first_only, second_only) + 1)
        )
        / 2**total
    )
    return min(1.0, 2 * tail)


def matched_odds_ratio(first_only: int, second_only: int) -> float | None:
    """Matched odds ratio; None when the denominator is zero or both are zero."""
    _check_discordances(first_only, second_only)
    return first_only / second_only if second_only else None


def _check_discordances(first_only: int, second_only: int) -> None:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (first_only, second_only)
    ):
        raise StatisticsError("Discordant counts must be nonnegative integers")


def paired_risk_difference(first: Sequence[int], second: Sequence[int]) -> float:
    """Mean(first - second) for matched 0/1 observations."""
    if (
        not first
        or len(first) != len(second)
        or any(
            value not in (0, 1) or isinstance(value, bool)
            for value in (*first, *second)
        )
    ):
        raise StatisticsError("Risk difference needs matched binary pairs")
    return sum(a - b for a, b in zip(first, second, strict=True)) / len(first)


def paired_standardized_mean_difference(differences: Sequence[float]) -> float | None:
    """Mean paired difference divided by its sample SD; None at zero variance."""
    _validate_values(differences)
    if len(differences) < 2:
        return None
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / (
        len(differences) - 1
    )
    return mean / math.sqrt(variance) if variance > 0 else None


def _validate_values(values: Sequence[float]) -> None:
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        for value in values
    ):
        raise StatisticsError("Statistic needs nonempty finite numeric values")


def _clusters(values: Mapping[str, Sequence[float]]) -> list[tuple[float, ...]]:
    if not values or any(not isinstance(key, str) or not key for key in values):
        raise StatisticsError("Cluster IDs must be nonempty strings")
    ordered = []
    for key in sorted(values):
        cluster = values[key]
        _validate_values(cluster)
        ordered.append(tuple(float(value) for value in cluster))
    return ordered


def cluster_mean(values: Mapping[str, Sequence[float]]) -> float:
    """Pool item-level values; callers first average repeated orders or variants."""
    clusters = _clusters(values)
    return sum(map(sum, clusters)) / sum(map(len, clusters))


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Nearest-rank empirical percentile, defined before confirmatory execution."""
    return sorted_values[max(0, math.ceil(probability * len(sorted_values)) - 1)]


def cluster_bootstrap_interval(
    values: Mapping[str, Sequence[float]],
    *,
    seed: int = 253,
    repetitions: int = 10000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Resample whole clusters and retain all item values within each draw."""
    clusters = _clusters(values)
    if repetitions < 1 or not 0 < confidence < 1:
        raise StatisticsError("Invalid bootstrap repetitions or confidence")
    rng = random.Random(seed)
    count = len(clusters)
    replicates = []
    for _ in range(repetitions):
        drawn = [clusters[rng.randrange(count)] for _ in range(count)]
        replicates.append(sum(map(sum, drawn)) / sum(map(len, drawn)))
    replicates.sort()
    tail = (1 - confidence) / 2
    return _percentile(replicates, tail), _percentile(replicates, 1 - tail)


def cluster_bootstrap_lower_bound(
    values: Mapping[str, Sequence[float]],
    *,
    seed: int = 253,
    repetitions: int = 10000,
    confidence: float = 0.95,
) -> float:
    """One-sided lower percentile from whole-cluster bootstrap draws."""
    clusters = _clusters(values)
    if repetitions < 1 or not 0 < confidence < 1:
        raise StatisticsError("Invalid bootstrap repetitions or confidence")
    rng = random.Random(seed)
    count = len(clusters)
    replicates = []
    for _ in range(repetitions):
        drawn = [clusters[rng.randrange(count)] for _ in range(count)]
        replicates.append(sum(map(sum, drawn)) / sum(map(len, drawn)))
    replicates.sort()
    return _percentile(replicates, 1 - confidence)


def cluster_sign_flip_pvalue(
    differences: Mapping[str, Sequence[float]],
    *,
    alternative: str,
    seed: int = 253,
    draws: int = 10000,
) -> float:
    """Flip all observations in a cluster together; include the observed draw."""
    clusters = _clusters(differences)
    if alternative not in ("greater", "two_sided") or draws < 1:
        raise StatisticsError("Invalid sign-flip alternative or draw count")
    totals = [sum(cluster) for cluster in clusters]
    observed = sum(totals)
    if all(total == 0 for total in totals):
        return 1.0
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        simulated = sum(total if rng.getrandbits(1) else -total for total in totals)
        if (
            simulated >= observed
            if alternative == "greater"
            else abs(simulated) >= abs(observed)
        ):
            extreme += 1
    return (extreme + 1) / (draws + 1)


def paired_cluster_permutation_pvalue(
    pairs: Mapping[str, tuple[Any, Any]],
    statistic: Callable[[dict[str, tuple[Any, Any]]], float],
    *,
    alternative: str,
    seed: int = 253,
    draws: int = 10000,
) -> float:
    """Swap entire paired clusters and recompute an arbitrary nonlinear statistic.

    The callback must be pure and return a finite scalar; it receives sorted IDs.
    The first member of each pair is the baseline, the second the comparison.
    """
    if (
        not pairs
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, tuple)
            or len(value) != 2
            for key, value in pairs.items()
        )
        or alternative not in ("greater", "two_sided")
        or draws < 1
    ):
        raise StatisticsError("Invalid paired-cluster permutation input")
    ordered = dict(sorted(pairs.items()))

    def measured(candidate: dict[str, tuple[Any, Any]]) -> float:
        result = statistic(candidate)
        if (
            isinstance(result, bool)
            or not isinstance(result, (int, float))
            or not math.isfinite(result)
        ):
            raise StatisticsError("Permutation statistic must be finite")
        return float(result)

    observed = measured(ordered)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        swapped = {
            key: value if rng.getrandbits(1) else (value[1], value[0])
            for key, value in ordered.items()
        }
        result = measured(swapped)
        if (
            result >= observed
            if alternative == "greater"
            else abs(result) >= abs(observed)
        ):
            extreme += 1
    return (extreme + 1) / (draws + 1)


def noninferiority_pass(lower_bound: float, margin: float) -> bool:
    """Require a finite one-sided lower bound strictly greater than the margin."""
    if not all(math.isfinite(value) for value in (lower_bound, margin)):
        raise StatisticsError("Non-inferiority bound and margin must be finite")
    return lower_bound > margin


def missing_mean_bounds(
    observed: Sequence[float], missing_count: int, low: float, high: float
) -> tuple[float, float]:
    """Worst/best mean if each missing outcome lies in a prespecified range."""
    if observed:
        _validate_values(observed)
    if (
        isinstance(missing_count, bool)
        or not isinstance(missing_count, int)
        or missing_count < 0
        or not all(math.isfinite(value) for value in (low, high))
        or low > high
        or not observed
        and missing_count == 0
        or any(value < low or value > high for value in observed)
    ):
        raise StatisticsError("Invalid missing count or outcome bounds")
    denominator = len(observed) + missing_count
    total = sum(observed)
    return (total + missing_count * low) / denominator, (
        total + missing_count * high
    ) / denominator


def exact_mcnemar_improvement(better: int, worse: int) -> float:
    """One-sided exact p-value for a reduction in a paired binary outcome."""
    _check_discordances(better, worse)
    total = better + worse
    if total == 0:
        return 1.0
    return sum(math.comb(total, k) for k in range(better, total + 1)) / 2**total


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return monotone BH-adjusted q-values in original test order."""
    if not p_values or any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise StatisticsError("BH needs finite p-values in [0, 1]")
    ordered = sorted(enumerate(p_values), key=lambda item: (item[1], item[0]))
    adjusted = [1.0] * len(p_values)
    minimum = 1.0
    for rank in range(len(ordered), 0, -1):
        index, value = ordered[rank - 1]
        minimum = min(minimum, value * len(ordered) / rank)
        adjusted[index] = minimum
    return adjusted


def bootstrap_lower_bound(
    values: Sequence[float], *, seed: int = 253, repetitions: int = 10000
) -> float:
    """Fifth percentile of seeded record bootstrap means (one-sided 95%)."""
    if (
        not values
        or repetitions < 1
        or any(not math.isfinite(value) for value in values)
    ):
        raise StatisticsError("Bootstrap needs finite values and positive repetitions")
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(rng.choice(values) for _ in range(size)) / size for _ in range(repetitions)
    )
    return means[max(0, math.ceil(0.05 * repetitions) - 1)]
