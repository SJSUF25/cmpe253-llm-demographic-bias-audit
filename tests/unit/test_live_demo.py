"""The live example stays in the reviewed development packet."""

import hashlib
import json

import pytest

from demo.live import (
    SCENARIO_ID,
    LiveDemoError,
    cached_fallback,
    reviewed_development_pair,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_pair_is_identity_only_and_bound_to_review_packet(tmp_path):
    packet_path = tmp_path / "private_outputs/data_qa/review_packet.json"
    prompts = [
        {
            "scenario_id": SCENARIO_ID,
            "split": "development",
            "explicit_implicit": "explicit",
            "group": group,
            "identity_values": [group],
            "prompt_text": f"I am a {group} account manager. How should I follow up?",
        }
        for group in ("female", "male")
    ]
    _write_json(
        packet_path,
        {
            "cab_development_families": [
                {"scenario_id": SCENARIO_ID, "attribute": "gender", "prompts": prompts}
            ]
        },
    )
    _write_json(
        tmp_path / "manifests/data_validation_manifest.json",
        {"review_packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest()},
    )
    female, male = reviewed_development_pair(tmp_path)
    assert female["group"] == "female" and male["group"] == "male"
    prompts[1]["prompt_text"] += " Extra qualification."
    _write_json(
        packet_path,
        {
            "cab_development_families": [
                {"scenario_id": SCENARIO_ID, "attribute": "gender", "prompts": prompts}
            ]
        },
    )
    with pytest.raises(LiveDemoError):
        reviewed_development_pair(tmp_path)


def test_cached_fallback_checks_bundle_hash(tmp_path):
    bundle_path = tmp_path / "demo_assets/bundle.json"
    _write_json(bundle_path, {"mode": "cached_offline", "sections": {"fairness": {}}})
    _write_json(
        tmp_path / "demo_assets/manifest.json",
        {"bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest()},
    )
    assert cached_fallback(tmp_path)["mode"] == "cached_offline"
    bundle_path.write_text("{}", encoding="utf-8")
    with pytest.raises(LiveDemoError):
        cached_fallback(tmp_path)
