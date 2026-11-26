"""Verify that a full BBQ run binds to the frozen 1,500-item input inventory."""

import hashlib
import json

import pytest

from inference import bbq_runner


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_bbq_input_hashes_and_inventory(tmp_path):
    config = json.loads(bbq_runner.BUILD_CONFIG.read_text())
    item_path = tmp_path / config["items_path"]
    permutation_path = tmp_path / config["permutations_path"]
    manifest_path = tmp_path / config["manifest_path"]
    item_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    items = [
        {"item_id": f"{category}|{index}", "category": category}
        for category in config["categories"]
        for index in range(500)
    ]
    permutations = [
        {"item_id": item["item_id"], "order": order}
        for item in items
        for order in range(3)
    ]
    item_path.write_text("\n".join(json.dumps(row) for row in items) + "\n")
    permutation_path.write_text(
        "\n".join(json.dumps(row) for row in permutations) + "\n"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "config_id": config["config_id"],
                "config_sha256": _sha(bbq_runner.BUILD_CONFIG),
                "source_revision": config["source_revision"],
                "items_path": config["items_path"],
                "permutations_path": config["permutations_path"],
                "items_sha256": _sha(item_path),
                "permutations_sha256": _sha(permutation_path),
                "items_count": 1500,
                "permutations_count": 4500,
            }
        )
    )
    loaded_items, loaded_orders, source_sha = bbq_runner._inputs(tmp_path)
    assert len(loaded_items) == 1500 and len(loaded_orders) == 4500
    assert source_sha == _sha(manifest_path)
    permutation_path.write_text(permutation_path.read_text() + "tampered\n")
    with pytest.raises(bbq_runner.FullBBQError, match="differ"):
        bbq_runner._inputs(tmp_path)


def test_only_frozen_full_bbq_run_ids_are_accepted():
    assert bbq_runner._entry("BBQ-M1-B0")["model"] == "M1"
    with pytest.raises(bbq_runner.FullBBQError, match="Unregistered"):
        bbq_runner._entry("PILOT2-BBQ-M1-B0")


def test_full_bbq_rejects_t4_before_loading_model(monkeypatch, tmp_path):
    monkeypatch.setattr(bbq_runner.runner, "git_state", lambda: ("a" * 40, False))
    monkeypatch.setattr(bbq_runner.runner, "_check_test_gate", lambda _root: None)
    monkeypatch.setattr(bbq_runner, "_inputs", lambda _root: ([], [], "hash"))
    monkeypatch.setattr(
        bbq_runner,
        "collect_environment",
        lambda: {"gpu": {"name": "Tesla T4"}},
    )
    monkeypatch.setattr(
        bbq_runner,
        "load_model",
        lambda _model: pytest.fail("Loaded model before checking GPU class"),
    )
    with pytest.raises(bbq_runner.FullBBQError, match="NVIDIA L4"):
        bbq_runner.run(tmp_path, "BBQ-M1-B0")
