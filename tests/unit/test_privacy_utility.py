import json
from collections import Counter
from pathlib import Path

import pytest

from privacy import utility
from privacy.membership_data import write_corpus
from privacy.synthetic_data import DEFAULT_CONFIG, generate_records


def test_blinded_audit_sample_is_stable_and_balanced():
    records = [
        row
        for row in generate_records(json.loads(DEFAULT_CONFIG.read_text()))
        if row["split"] == "test"
    ]
    selected = utility.audit_sample(records)
    assert len(selected) == 50
    assert Counter(row["task"] for row in selected) == dict.fromkeys(
        (
            "summarization",
            "email_rewrite",
            "case_note",
            "action_items",
            "json_extraction",
        ),
        10,
    )
    assert [row["record_id"] for row in selected] == [
        row["record_id"] for row in utility.audit_sample(list(reversed(records)))
    ]
    with pytest.raises(utility.UtilityError, match="non-test"):
        utility.audit_sample([*records, {**records[0], "split": "development"}])


def test_adapter_utility_uses_only_frozen_validation(tmp_path: Path):
    write_corpus(tmp_path)
    rows, digest = utility._validation_records(tmp_path)
    assert len(rows) == 400
    assert Counter(row["task"] for row in rows) == dict.fromkeys(
        ("copy_code", "json_field", "uppercase", "addition"), 100
    )
    assert len(digest) == 64
    path = tmp_path / "private_outputs/membership/ordinary_validation.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(utility.UtilityError, match="hash differs"):
        utility._validation_records(tmp_path)


def test_blinded_rating_tally_requires_all_aliases(tmp_path: Path):
    directory = tmp_path / "private_outputs/privacy_h5"
    directory.mkdir(parents=True)
    packet = directory / "helpfulness_packet.jsonl"
    key_path = directory / "helpfulness_alias_key.jsonl"
    ratings_path = directory / "helpfulness_ratings.jsonl"
    packet.write_text("{}\n")
    keys = [
        {
            "audit_id": f"AUD-{model}-{index}",
            "model": model,
            "aliases": {"A": "P0", "B": "P1", "C": "P2"},
        }
        for model in ("M1", "M2")
        for index in range(50)
    ]
    key_path.write_text("".join(json.dumps(row) + "\n" for row in keys))
    ratings = [
        {
            "audit_id": row["audit_id"],
            "alias": alias,
            "rating": 5 if alias == "A" else 4,
            "reviewer": "Vineet Malewar",
        }
        for row in keys
        for alias in ("A", "B", "C")
    ]
    ratings_path.write_text("".join(json.dumps(row) + "\n" for row in ratings))
    manifest_path = tmp_path / "manifests/privacy/helpfulness_packet.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "packet_sha256": utility.evaluation._sha(packet.read_bytes()),
                "alias_key_sha256": utility.evaluation._sha(key_path.read_bytes()),
            }
        )
    )
    result = json.loads(utility.score_helpfulness_ratings(tmp_path).read_text())
    assert result["cells"]["M1:P0"]["mean_helpfulness"] == 5
    assert result["cells"]["M2:P2"]["mean_helpfulness"] == 4
    ratings_path.write_text("".join(json.dumps(row) + "\n" for row in ratings[:-1]))
    with pytest.raises(utility.UtilityError, match="300 blinded ratings"):
        utility.score_helpfulness_ratings(tmp_path)


def test_helpfulness_review_saves_and_resumes_without_unblinding(
    tmp_path: Path, monkeypatch
):
    directory = tmp_path / "private_outputs/privacy_h5"
    directory.mkdir(parents=True)
    packet = directory / "helpfulness_packet.jsonl"
    rows = [
        {
            "audit_id": f"AUD-{index}",
            "task": "case_note",
            "case_facts": "Fictional case facts",
            "responses": {"A": "First", "B": "Second", "C": "Third"},
        }
        for index in range(100)
    ]
    packet.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "manifests/privacy/helpfulness_packet.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"packet_sha256": utility.evaluation._sha(packet.read_bytes())})
    )
    answers = iter(("4", "q"))
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    utility.review_helpfulness(tmp_path)
    ratings = directory / "helpfulness_ratings.jsonl"
    assert [json.loads(line)["alias"] for line in ratings.read_text().splitlines()] == [
        "A"
    ]
    answers = iter(("3", "q"))
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    utility.review_helpfulness(tmp_path)
    saved = [json.loads(line) for line in ratings.read_text().splitlines()]
    assert [(row["alias"], row["rating"]) for row in saved] == [("A", 4), ("B", 3)]
