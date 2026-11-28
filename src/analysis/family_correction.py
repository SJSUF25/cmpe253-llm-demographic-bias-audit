"""Apply the frozen H1/H2 multiplicity correction only to complete families."""

from __future__ import annotations

import math
from collections.abc import Sequence

from analysis.plan import expected_tests
from analysis.statistics import benjamini_hochberg


class FamilyCorrectionError(ValueError):
    """The results do not match the complete preregistered H1/H2 inventory."""


def correct_h1_h2(results: Sequence[dict]) -> list[dict]:
    """Return 36 results with BH q values, or fail before any family decision."""
    planned = {
        test["id"]: test for test in expected_tests() if test["family"] in {"H1", "H2"}
    }
    if len(planned) != 36 or len(results) != 36:
        raise FamilyCorrectionError("H1/H2 correction requires all 36 planned tests")
    by_id = {}
    for result in results:
        test_id = result.get("test_id")
        p = result.get("p_unadjusted")
        if (
            test_id not in planned
            or test_id in by_id
            or not isinstance(p, (int, float))
            or isinstance(p, bool)
            or not math.isfinite(p)
            or not 0 <= p <= 1
        ):
            raise FamilyCorrectionError("Unknown, duplicate, or invalid H1/H2 test")
        family = planned[test_id]["family"]
        if result.get("family") != family:
            raise FamilyCorrectionError("Result family differs from frozen plan")
        by_id[test_id] = result
    if set(by_id) != set(planned):
        raise FamilyCorrectionError("H1/H2 test inventory is incomplete")
    corrected = []
    for family in ("H1", "H2"):
        tests = [
            by_id[test["id"]] for test in expected_tests() if test["family"] == family
        ]
        if len(tests) != 18:
            raise FamilyCorrectionError("Unexpected BH family size")
        q_values = benjamini_hochberg([test["p_unadjusted"] for test in tests])
        corrected.extend(
            {**test, "q_bh18": q} for test, q in zip(tests, q_values, strict=True)
        )
    return corrected
