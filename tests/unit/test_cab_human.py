"""Verify the frozen CAB human analysis against a complete synthetic packet."""

import hashlib
import json

import pytest

from analysis.cab_human import CABHumanError, analyze


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packet(tmp_path):
    directory = tmp_path / "private_outputs/annotation/vineet/test"
    responses, sets, mappings, ratings = [], [], [], []
    for attribute, groups in (
        ("gender", ("female", "male")),
        ("race", ("white", "black", "asian", "hispanic")),
    ):
        for index in range(24):
            sid = f"{attribute}-{index:02}"
            for model in ("M1", "M2"):
                for condition in ("B0", "F1"):
                    set_id = f"{sid}-{model}-{condition}"
                    records = []
                    for group in groups:
                        record_id = f"{set_id}-{group}"
                        records.append(record_id)
                        responses.append(
                            {
                                "record_id": record_id,
                                "set_id": set_id,
                                "prompt": "Synthetic prompt",
                                "response": "Synthetic answer",
                                "output_status": "ok",
                            }
                        )
                        mappings.append(
                            {
                                "record_id": record_id,
                                "set_id": set_id,
                                "scenario_id": sid,
                                "attribute": attribute,
                                "group": group,
                                "model": model,
                                "condition": condition,
                            }
                        )
                        ratings.append(
                            {
                                "kind": "response",
                                "card_id": record_id,
                                "rating": {
                                    "stereotype": 0,
                                    "helpfulness": 4,
                                    "relevance": 5,
                                    "refusal": "yes"
                                    if group in ("female", "black")
                                    else "no",
                                    "confidence": 3,
                                },
                            }
                        )
                    sets.append({"set_id": set_id, "record_ids": records})
                    ratings.append(
                        {
                            "kind": "set",
                            "card_id": set_id,
                            "rating": {
                                "differential_treatment": 2 if condition == "B0" else 0
                            },
                        }
                    )
    packet = {
        "schema_version": 1,
        "split": "test",
        "annotator": "vineet",
        "response_cards": responses,
        "set_cards": sets,
    }
    packet_path = directory / "packet.json"
    _write(packet_path, packet)
    _write(
        directory / "sealed_key.json",
        {"mappings": mappings, "source_hashes": {}},
    )
    rating_path = directory / "frozen_ratings.jsonl"
    rating_path.write_text("".join(json.dumps(row) + "\n" for row in ratings))
    _write(
        directory / "frozen_ratings_manifest.json",
        {
            "packet_sha256": _sha(packet_path),
            "ratings_sha256": _sha(rating_path),
            "response_count": 576,
            "set_count": 192,
        },
    )
    return packet_path


def test_complete_packet_produces_frozen_cab_inventory(tmp_path):
    result = analyze(_packet(tmp_path))
    assert len(result["tests"]) == 26
    by_id = {row["test_id"]: row for row in result["tests"]}
    assert by_id["H1:cab_differential:M1:gender"]["material_count"] == 24
    assert by_id["H1:cab_differential:M1:gender"]["ci95"] == (1.0, 1.0)
    assert by_id["H2:cab_differential:M1:race"]["estimate"] == 2
    assert by_id["H1:cab_refusal:M1:female_minus_male"]["estimate"] == 1
    assert by_id["H1:cab_refusal:M1:female_minus_male"]["ci95"] == (1.0, 1.0)
    assert by_id["H2:cab_refusal:M1:female_minus_male"]["estimate"] == 0
    assert by_id["H2:cab_refusal:M1:female_minus_male"]["ci95"] == (0.0, 0.0)
    assert by_id["H3:cab_helpfulness:M1:all_attributes"]["noninferiority_passes"]


def test_changed_rating_receipt_is_rejected(tmp_path):
    packet = _packet(tmp_path)
    with (packet.parent / "frozen_ratings.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(CABHumanError, match="receipt"):
        analyze(packet)
