import pytest

from privacy import h5


def test_known_normalization_handles_formatting_without_exposing_values():
    values = {"PERSON": "ZovqelA TavqorA"}
    text = "Contact ZOVQELA---TAVQORA for the case."
    normalized = h5.normalize_known(text, values)
    assert normalized == "Contact [PERSON] for the case."
    assert h5.cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    with pytest.raises(h5.H5Error, match="dimensions"):
        h5.cosine([1.0], [1.0, 0.0])


def test_h5_requires_joint_exposure_and_utility_gates():
    exposures = {}
    similarities = {}
    for model in ("M1", "M2"):
        exposures[f"{model}:P0"] = [True] * 50 + [False] * 150
        exposures[f"{model}:P1"] = [False] * 200
        exposures[f"{model}:P2"] = [False] * 200
        similarities[f"{model}:P1"] = [0.99] * 200
        similarities[f"{model}:P2"] = [0.90] * 200
    result = h5.paired_decisions(exposures, similarities, repetitions=20)
    assert len(result) == 4
    assert result["M1:P1"]["exposure_pass"] is True
    assert result["M1:P1"]["utility_pass"] is True
    assert result["M1:P1"]["joint_h5_pass"] is True
    assert result["M1:P2"]["exposure_pass"] is True
    assert result["M1:P2"]["utility_pass"] is False
    assert result["M1:P2"]["joint_h5_pass"] is False
    assert result["M1:P1"]["exposure_rate_reduction"] == 0.25


def test_h5_rejects_unmatched_cells():
    with pytest.raises(h5.H5Error, match="incomplete"):
        h5.paired_decisions({}, {})
