import json

import pytest

from data import cab_quality


def _prompt(scenario_id, mode, group, text, split="development"):
    return {
        "scenario_id": scenario_id,
        "explicit_implicit": mode,
        "group": group,
        "attribute": "gender",
        "split": split,
        "prompt_text": text,
    }


def test_quality_rule_quarantines_entire_implicit_family(monkeypatch):
    monkeypatch.setattr(cab_quality, "EXPECTED_EXCLUDED_FAMILIES", 1)
    monkeypatch.setattr(cab_quality, "EXPECTED_ELIGIBLE_IMPLICIT_PROMPTS", 2)
    rows = [
        _prompt("clean", mode, group, text)
        for mode, group, text in (
            ("explicit", "male", "At the local office today, what should I do next?"),
            ("explicit", "female", "At the local office today, what should I do next?"),
            (
                "implicit",
                "male",
                "I am A. At the local office today, what should I do next?",
            ),
            (
                "implicit",
                "female",
                "I am B. At the local office today, what should I do next?",
            ),
        )
    ] + [
        _prompt("leak", mode, group, text, "test")
        for mode, group, text in (
            ("explicit", "male", "Please draft a note."),
            ("explicit", "female", "Please draft a note."),
            ("implicit", "male", "Please draft a note.\n\nHere is the note."),
            ("implicit", "female", "Please draft a note.\n\nHere is the note."),
        )
    ]
    content = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    result = cab_quality.make_manifest(content, "pinned-source-hash")
    assert result["excluded_implicit_scenario_ids"] == ["leak"]
    assert result["excluded_family_counts"] == {"test|gender": 1}
    assert result["eligible_implicit_prompt_count"] == 2
    assert result["explicit_prompt_count"] == 4


def test_quality_rule_rejects_group_disagreement(monkeypatch):
    monkeypatch.setattr(cab_quality, "EXPECTED_EXCLUDED_FAMILIES", 1)
    monkeypatch.setattr(cab_quality, "EXPECTED_ELIGIBLE_IMPLICIT_PROMPTS", 0)
    rows = [
        _prompt("one", "explicit", "male", "Draft a note."),
        _prompt("one", "explicit", "female", "Draft a note."),
        _prompt("one", "implicit", "male", "Draft a note.\n\nAnswer."),
        _prompt("one", "implicit", "female", "Draft a note."),
    ]
    with pytest.raises(cab_quality.CabQualityError, match="Inconsistent"):
        cab_quality.make_manifest(
            ("\n".join(json.dumps(row) for row in rows) + "\n").encode(), "hash"
        )
