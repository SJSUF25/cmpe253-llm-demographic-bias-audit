import json
from pathlib import Path

import pytest

from privacy import canary_attack
from privacy.membership_data import write_corpus


def test_attack_partitions_match_frozen_generator(tmp_path: Path):
    write_corpus(tmp_path)
    rows, hashes = canary_attack._read_partitions(tmp_path)
    assert len(rows) == 1300
    assert len(hashes) == 3
    path = tmp_path / "private_outputs/membership/nonmember_evaluation.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(canary_attack.CanaryAttackError, match="Changed private"):
        canary_attack._read_partitions(tmp_path)


def test_calibrated_threshold_and_tie_aware_auc():
    threshold = canary_attack.threshold_from_calibration([float(x) for x in range(500)])
    assert threshold == 494.0
    assert sum(float(x) > threshold for x in range(500)) == 5
    assert canary_attack.roc_auc([2.0, 1.0], [1.0, 0.0]) == 0.875
    assert canary_attack.roc_auc([1.0], [1.0]) == 0.5


def test_heldout_attack_summary_excludes_calibration_from_auc():
    rows = [
        {
            "record_id": f"MEM-{index}",
            "split": "member",
            "repetition": (1, 5, 10)[index // 100],
            "negative_loss": 2.0,
            "min_k20": 2.0,
            "extracted_exact": index >= 200,
        }
        for index in range(300)
    ]
    rows.extend(
        {
            "record_id": f"CAL-{index}",
            "split": "calibration",
            "repetition": 0,
            "negative_loss": 5.0 if index < 5 else 0.0,
            "min_k20": 5.0 if index < 5 else 0.0,
            "extracted_exact": False,
        }
        for index in range(500)
    )
    rows.extend(
        {
            "record_id": f"EVAL-{index}",
            "split": "evaluation",
            "repetition": 0,
            "negative_loss": 0.0,
            "min_k20": 0.0,
            "extracted_exact": False,
        }
        for index in range(500)
    )
    result = canary_attack.summarize(rows, bootstrap_repetitions=20)
    score = result["score_attacks"]["negative_loss"]
    assert score["auc"] == 1.0
    assert score["calibration_false_positives"] == 5
    assert score["evaluation_fpr"] == 0.0
    assert score["tpr_at_calibrated_1pct_fpr"] == 1.0
    assert result["member_extraction_by_repetition"]["10"]["exact_rate"] == 1.0
    assert result["member_extraction_by_repetition"]["1"]["exact_rate"] == 0.0
    assert "CANARY-" not in json.dumps(result)


def test_attack_parts_resume_only_when_hashes_and_ids_match(monkeypatch, tmp_path):
    records = [
        {"record_id": f"REC-{index}", "split": "evaluation", "repetition": 0}
        for index in range(200)
    ]
    called = []

    def fake_score(_model, _tokenizer, row):
        called.append(row["record_id"])
        return {
            **row,
            "negative_loss": -1.0,
            "min_k20": -2.0,
            "extracted_exact": False,
        }

    monkeypatch.setattr(canary_attack, "score_one", fake_score)
    kwargs = (
        tmp_path,
        "base",
        records,
        {"evaluation": "source"},
        {"model": "rev"},
        "commit",
    )
    first = canary_attack._score_parts(*kwargs, None, None)
    assert len(first) == 200
    assert len(called) == 200
    assert canary_attack._score_parts(*kwargs, None, None) == first
    assert len(called) == 200
    part = tmp_path / "private_outputs/canary_attack/base_parts/0001.jsonl"
    part.write_bytes(part.read_bytes() + b" ")
    with pytest.raises(
        canary_attack.CanaryAttackError, match="checkpoint part differs"
    ):
        canary_attack._score_parts(*kwargs, None, None)
