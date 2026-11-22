import json
from collections import Counter

import pytest

from privacy.membership_data import (
    DEFAULT_CONFIG,
    MembershipDataError,
    generate_corpus,
    write_corpus,
)


def test_fixed_membership_partitions_and_training_exclusion():
    corpus = generate_corpus(json.loads(DEFAULT_CONFIG.read_text()))
    assert {name: len(rows) for name, rows in corpus.items()} == {
        "ordinary_train": 1600,
        "ordinary_validation": 400,
        "member_canaries": 300,
        "nonmember_calibration": 500,
        "nonmember_evaluation": 500,
        "training_presentations": 3200,
    }
    assert Counter(row["repetition"] for row in corpus["member_canaries"]) == {
        1: 100,
        5: 100,
        10: 100,
    }
    training_ids = {row["record_id"] for row in corpus["training_presentations"]}
    sealed_ids = {
        row["record_id"]
        for name in (
            "ordinary_validation",
            "nonmember_calibration",
            "nonmember_evaluation",
        )
        for row in corpus[name]
    }
    assert training_ids.isdisjoint(sealed_ids)
    assert (
        len(
            {
                row["response"]
                for row in corpus["member_canaries"]
                + corpus["nonmember_calibration"]
                + corpus["nonmember_evaluation"]
            }
        )
        == 1300
    )


def test_ordinary_reference_answers_are_exact():
    corpus = generate_corpus(json.loads(DEFAULT_CONFIG.read_text()))
    for row in corpus["ordinary_train"] + corpus["ordinary_validation"]:
        task = row["task"]
        prompt = row["prompt"]
        if task == "copy_code":
            assert row["response"] == prompt.removeprefix("Copy this code exactly: ")
        elif task == "json_field":
            obj = json.loads(
                prompt.removeprefix("Return only beta from this JSON object: ")
            )
            assert row["response"] == obj["beta"]
        elif task == "uppercase":
            assert (
                row["response"]
                == prompt.removeprefix("Uppercase this eight-letter string: ").upper()
            )
        else:
            operands = prompt.removeprefix(
                "Add these two integers and return only the sum: "
            )
            left, right = operands.split(" + ")
            assert row["response"] == str(int(left) + int(right))


def test_membership_corpus_is_immutable_and_repeatable(tmp_path):
    manifest = write_corpus(tmp_path)
    first = manifest.read_bytes()
    assert write_corpus(tmp_path).read_bytes() == first
    target = tmp_path / "private_outputs/membership/nonmember_evaluation.jsonl"
    target.write_text("tampered", encoding="utf-8")
    with pytest.raises(MembershipDataError, match="Existing output differs"):
        write_corpus(tmp_path)


def test_locked_configuration_rejects_changed_repetition_counts():
    config = json.loads(DEFAULT_CONFIG.read_text())
    config["repetition_strata"] = [1, 5, 9]
    with pytest.raises(MembershipDataError, match="Unsupported"):
        generate_corpus(config)
