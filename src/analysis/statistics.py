"""Small, deterministic statistical primitives for preregistered paired analyses."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


class StatisticsError(ValueError):
    """A statistical input is empty, non-finite, or outside its contract."""


def exact_mcnemar_improvement(better: int, worse: int) -> float:
    """One-sided exact p-value for a reduction in a paired binary outcome."""
    if min(better, worse) < 0 or any(
        not isinstance(value, int) for value in (better, worse)
    ):
        raise StatisticsError("McNemar discordant counts must be nonnegative integers")
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
