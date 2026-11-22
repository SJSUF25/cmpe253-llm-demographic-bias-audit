import pytest

from data import validate as data_gate


def test_missing_artifacts_fail_before_any_write(tmp_path):
    with pytest.raises(data_gate.DataGateError, match="Missing"):
        data_gate.validate_data(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_manual_packet_uses_development_and_calibration_only(monkeypatch, tmp_path):
    cab = [
        {
            "attribute": attribute,
            "scenario_id": f"{attribute}-{index}",
            "split": "development",
            "prompt_text": "reviewable",
        }
        for attribute in ("gender", "race")
        for index in range(25)
    ] + [
        {
            "attribute": "gender",
            "scenario_id": "sealed",
            "split": "test",
            "prompt_text": "DO-NOT-REVIEW",
        }
    ]
    pii = [
        {"record_id": f"pii-{task}-{index}", "task": task}
        for task in (
            "summarization",
            "email_rewrite",
            "case_note",
            "action_items",
            "json_extraction",
        )
        for index in range(2)
    ]
    ordinary = [
        {"record_id": f"ordinary-{task}-{index}", "task": task}
        for task in ("copy_code", "json_field", "uppercase", "addition")
        for index in range(2)
    ]
    members = [{"record_id": f"member-{rep}", "repetition": rep} for rep in (1, 5, 10)]
    calibration = [{"record_id": f"cal-{index}"} for index in range(2)]
    bbq = [
        {
            "item_id": f"{category}-{context}-{polarity}",
            "category": category,
            "context_condition": context,
            "question_polarity": polarity,
        }
        for category in ("Gender_identity", "Race_ethnicity", "Race_x_gender")
        for context in ("ambig", "disambig")
        for polarity in ("neg", "nonneg")
    ]

    def fake_jsonl(path):
        if path.name == "prompt_manifest.jsonl":
            return cab
        if path.name == "items.jsonl":
            return bbq
        if path.name == "development.jsonl":
            return pii
        if path.name == "ordinary_train.jsonl":
            return ordinary
        if path.name == "member_canaries.jsonl":
            return members
        if path.name == "nonmember_calibration.jsonl":
            return calibration
        raise AssertionError(f"Sealed or unexpected partition opened: {path}")

    monkeypatch.setattr(data_gate, "_jsonl", fake_jsonl)
    packet = data_gate.make_review_packet(tmp_path)
    assert len(packet["cab_development_families"]) == 50
    assert len(packet["bbq"]) == 12
    assert packet["sealed_test_contents_included"] is False
    assert "DO-NOT-REVIEW" not in str(packet)
