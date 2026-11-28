import pytest

from analysis.family_correction import FamilyCorrectionError, correct_h1_h2
from analysis.plan import expected_tests


def _results():
    return [
        {
            "test_id": test["id"],
            "family": test["family"],
            "p_unadjusted": 0.01 if test["family"] == "H1" else 0.5,
        }
        for test in expected_tests()
        if test["family"] in {"H1", "H2"}
    ]


def test_correction_requires_complete_frozen_families():
    results = _results()
    corrected = correct_h1_h2(list(reversed(results)))
    assert len(corrected) == 36
    assert {row["q_bh18"] for row in corrected if row["family"] == "H1"} == {0.01}
    assert {row["q_bh18"] for row in corrected if row["family"] == "H2"} == {0.5}
    with pytest.raises(FamilyCorrectionError, match="all 36"):
        correct_h1_h2(results[:-1])
    with pytest.raises(FamilyCorrectionError, match="duplicate"):
        correct_h1_h2([*results[:-1], results[0]])
    with pytest.raises(FamilyCorrectionError, match="family differs"):
        correct_h1_h2([{**results[0], "family": "H2"}, *results[1:]])
