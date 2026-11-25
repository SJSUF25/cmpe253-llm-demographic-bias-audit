"""Locked human annotation sampling, masking, and immutable rating behavior."""

import json

import pytest

from scoring.annotation import (
    AnnotationError,
    _mask,
    _write_once,
    build_packet,
    select_families,
)
from scoring.annotation_review import export, validate_rating


def test_hamilton_sample_is_deterministic_and_stratified():
    rows = []
    for attribute in ("gender", "race"):
        for index in range(20):
            rows.append(
                {
                    "split": "test",
                    "explicit_implicit": "explicit",
                    "attribute": attribute,
                    "superdomain": "a" if index < 15 else "b",
                    "scenario_id": f"{attribute}-{index}",
                }
            )
    first = select_families(rows, "test", 8, "annotation48")
    assert first == select_families(list(reversed(rows)), "test", 8, "annotation48")
    assert len(first) == 16
    assert sum(item.startswith("gender-") for item in first) == 8
    assert (
        sum(item.endswith(tuple(str(n) for n in range(15, 20))) for item in first) == 4
    )


def test_mask_uses_aliases_and_preserves_other_words():
    prompt = {"group": "female", "identity_values": ["Linda"]}
    aliases = {"female": "[GROUP_A]", "male": "[GROUP_B]"}
    assert (
        _mask("Linda is a woman. Daniel is a man.", prompt, aliases)
        == "[GROUP_A] is a [GROUP_A]. Daniel is a [GROUP_B]."
    )


def test_rating_validation_and_write_once(tmp_path):
    valid = {
        "stereotype": 0,
        "helpfulness": 5,
        "relevance": 4,
        "refusal": "no",
        "confidence": 3,
    }
    validate_rating("response", valid)
    validate_rating("set", {"abstain_reason": "insufficient_context"})
    with pytest.raises(AnnotationError):
        validate_rating("response", {**valid, "helpfulness": 6})
    path = tmp_path / "card.json"
    _write_once(path, b"a")
    _write_once(path, b"a")
    with pytest.raises(AnnotationError):
        _write_once(path, b"b")


def test_export_refuses_incomplete_packet(tmp_path):
    packet = {
        "schema_version": 1,
        "split": "development",
        "response_cards": [{"record_id": str(i)} for i in range(240)],
        "set_cards": [{"set_id": str(i)} for i in range(80)],
    }
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet))
    with pytest.raises(AnnotationError, match="Pending response"):
        export(path)


def test_development_packet_blinds_four_cells_and_seals_key(tmp_path, monkeypatch):
    from hashlib import sha256

    from scoring import annotation

    prompts = []
    for attribute, groups in (
        ("gender", ("male", "female")),
        ("race", ("white", "black", "asian", "hispanic")),
    ):
        for index in range(10):
            for group in groups:
                sid = f"{attribute}-{index}"
                prompts.append(
                    {
                        "split": "development",
                        "explicit_implicit": "explicit",
                        "attribute": attribute,
                        "superdomain": "work",
                        "scenario_id": sid,
                        "prompt_id": f"{sid}-{group}",
                        "group": group,
                        "identity_values": [group],
                        "prompt_text": f"Help a {group} colleague.",
                    }
                )
    root = tmp_path
    prompt_path = root / "prompts.jsonl"
    prompt_path.write_text("\n".join(json.dumps(row) for row in prompts) + "\n")
    (root / "build.json").write_text(
        json.dumps({"output_sha256": sha256(prompt_path.read_bytes()).hexdigest()})
    )
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps({"output_path": "prompts.jsonl", "manifest_path": "build.json"})
    )
    monkeypatch.setattr(annotation.cab, "CAB_CONFIG", config_path)

    def fake_inputs(_root, run_id, split):
        assert split == "development"
        outputs = [
            {
                "scenario_id": p["scenario_id"],
                "prompt_id": p["prompt_id"],
                "raw_output": f"The {p['group']} colleague can proceed.",
                "status": "ok",
            }
            for p in prompts
        ]
        return prompts, outputs, {"run_id": run_id}

    monkeypatch.setattr(annotation.cab, "_inputs", fake_inputs)
    packet_path = build_packet(root, "development", "vineet")
    packet = json.loads(packet_path.read_text())
    key = json.loads((packet_path.parent / "sealed_key.json").read_text())
    assert len(packet["response_cards"]) == 240
    assert len(packet["set_cards"]) == 80
    assert len(key["mappings"]) == 240
    assert "source_hashes" not in packet
    assert all(
        "model" not in row and "condition" not in row and "group" not in row
        for row in packet["response_cards"]
    )
    assert all("[GROUP_" in row["prompt"] for row in packet["response_cards"])
    assert build_packet(root, "development", "vineet") == packet_path
