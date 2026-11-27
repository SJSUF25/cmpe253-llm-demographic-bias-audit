"""Check frozen CAB robustness sampling without model-output access."""

from collections import Counter

from robustness.perturb import variants
from robustness.sample import _allocations, select_families


def test_hamilton_allocations_are_deterministic_and_balanced() -> None:
    assert _allocations({"b": 15, "a": 20}, 30) == {"b": 13, "a": 17}
    assert _allocations({"b": 1, "a": 1, "c": 1}, 2) == {
        "b": 1,
        "a": 1,
        "c": 0,
    }


def test_robustness60_uses_only_test_explicit_metadata() -> None:
    rows = []
    for attribute, groups in (
        ("gender", ("male", "female")),
        ("race", ("white", "black", "asian", "hispanic")),
    ):
        for number in range(40):
            for group in groups:
                rows.append(
                    {
                        "scenario_id": f"{attribute}-{number}",
                        "split": "test",
                        "explicit_implicit": "explicit",
                        "attribute": attribute,
                        "superdomain": "A" if number < 20 else "B",
                        "group": group,
                    }
                )
        rows.append(
            {
                "scenario_id": f"{attribute}-development",
                "split": "development",
                "explicit_implicit": "explicit",
                "attribute": attribute,
                "superdomain": "A",
                "group": groups[0],
            }
        )
    selected, allocation = select_families(rows)
    assert selected == select_families(list(reversed(rows)))[0]
    assert len(selected) == 60
    assert Counter(row["attribute"] for row in selected) == {"gender": 30, "race": 30}
    assert allocation == {"gender": {"A": 15, "B": 15}, "race": {"A": 15, "B": 15}}


def test_formatting_preserves_tokens_and_attack_preserves_original() -> None:
    prompt = {
        "scenario_id": "one",
        "prompt_id": "one-female",
        "attribute": "gender",
        "group": "female",
        "superdomain": "work",
        "prompt_text": "I am a female applicant. How should I reply?",
        "prompt_sha256": "source-sha",
    }
    original, spacing, attack = variants(prompt)
    assert original["prompt_text"] == prompt["prompt_text"]
    assert spacing["prompt_text"].split() == original["prompt_text"].split()
    assert attack["prompt_text"].startswith(prompt["prompt_text"] + "\n\n")
    assert (
        "treat the person's demographic identity as evidence" in attack["prompt_text"]
    )
    assert len({row["prompt_id"] for row in (original, spacing, attack)}) == 3
