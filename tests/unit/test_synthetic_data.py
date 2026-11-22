import json

import pytest

from privacy.synthetic_data import (
    DEFAULT_CONFIG,
    SyntheticDataError,
    generate_records,
    write_corpus,
)


def test_balanced_synthetic_records_and_redacted_references():
    config = json.loads(DEFAULT_CONFIG.read_text())
    records = generate_records(config)
    assert len(records) == 400
    assert {row["split"] for row in records} == {"development", "test"}
    assert all(len(row["pii"]) == 5 for row in records)
    assert all("[PERSON]" in row["reference"]["redacted_input"] for row in records)


def test_corpus_rerun_is_identical_and_detects_existing_change(tmp_path):
    manifest = write_corpus(tmp_path)
    first = manifest.read_bytes()
    assert write_corpus(tmp_path).read_bytes() == first
    target = tmp_path / "private_outputs/synthetic_pii/test.jsonl"
    target.write_text("tampered")
    with pytest.raises(SyntheticDataError, match="Existing output differs"):
        write_corpus(tmp_path)
